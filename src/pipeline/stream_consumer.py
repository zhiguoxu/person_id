"""
StreamConsumer — 服务端后台拉流消费

把"浏览器抓帧上传"改为"服务端直接拉视频流":
- 读流线程: cv2.VideoCapture (FFMPEG) 拉 FLV/HLS/RTSP, 只保留最新一帧, 断线自动重连
- 处理协程: 取最新帧 → orchestrator.process_frame → 编码 JPEG → 广播给所有观看端

观看端 (前端页面) 通过原有 /ws/vision WebSocket 接收:
- 二进制消息: 处理时所用的原始帧 (JPEG), 前端画到 canvas
- frame_result JSON: 识别结果, 前端照旧用 overlay 画框
- event JSON: 系统事件

帧节流策略: 读流线程全速消费 (避免解码器积压导致延迟), 处理协程按
stream_max_fps 上限取"最新帧"处理, 中间帧直接丢弃。

拉流自动恢复: ISS 设备推流偶发损坏后, 单纯对旧地址重连永远失败。读流线程
统计连续失败次数 (打开失败 / 连上但没读到几帧就断), 达到阈值
(stream_restream_fail_threshold, 控制台可调) 时调度
pipeline/restream.run_restream_recovery(): 查设备在线 → ISS 重新推流 →
切换到新 FLV 地址继续拉。全过程记录在 data/restream_log/, 前端可查。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from voice_agent_common.utils.context import set_device_sn
from voice_agent_common.utils.logger import logger

from src.api.iss_client import ISSEnv
from src.api.schemas import (
    StreamStatusResponse,
    build_frame_result,
    build_ws_event,
)
from src.configs.config import config
from src.configs.override import current_config
from src.pipeline.orchestrator import VisionOrchestrator
from src.pipeline.stream_probe import diagnose_open_failure
from src.pipeline.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from src.pipeline.gpu_decoder import NvdecCapture


class StreamConsumer:
    """单摄像头的后台拉流消费器。"""

    # 一次连接读到这么多帧即视为"健康会话", 连续失败计数清零
    # (真实流 15fps 下 1 秒内就能达到; 损坏的流通常一帧都读不到或只吐少量残帧)
    _HEALTHY_SESSION_FRAMES = 10

    def __init__(
        self,
        camera_id: str,
        url: str,
        orchestrator: VisionOrchestrator,
        env: ISSEnv = "test",
        auto_restream: bool = True,
    ) -> None:
        self.camera_id = camera_id
        self.url = url
        self.orchestrator = orchestrator
        # ISS 环境 (test/prod, 自动重推流时沿用) 与自动恢复开关
        self.env = env
        self.auto_restream = auto_restream

        # 最新帧 (读流线程写, 处理协程读)
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = 0
        self._frame_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._process_task: asyncio.Task | None = None

        # 状态统计
        self.running = False
        self.connected = False
        self.last_error: str | None = None
        self.frames_read = 0
        self.frames_processed = 0
        self.process_fps = 0.0  # EMA
        # 实际拉到的流分辨率 (随流动态更新, 换设备/换推流配置后可从 status 直接确认)
        self.stream_width = 0
        self.stream_height = 0

        # 拉流自动恢复状态
        self._loop: asyncio.AbstractEventLoop | None = None  # 读流线程回调恢复协程用
        self._consecutive_pull_failures = 0
        self._recovering = False            # 恢复协程进行中 (防重复调度)
        self.restream_count = 0             # 本次消费器生命周期内成功重推流次数
        self.last_restream_at: float | None = None
        self.last_restream_outcome: str | None = None

        # 拉流录像(consume 生命周期内不断线切会话; stop 时收尾上传)
        self._recorder: VideoRecorder | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, source: str) -> None:
        """启动读流线程与处理协程。

        source: 触发启动的入口(接口/启动恢复等), 只进日志, 用于追踪谁拉起了这条流。
        """
        if self.running:
            return
        self.running = True
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        # 拉流即挂 recorder: 断线重连/重推流不换, 显式 stop 才收尾。
        # video_record.enabled 是热字段, 由 maybe_write 逐帧现读门控——
        # recorder 无条件创建(纯状态对象, 不开编码器), 中途打开录像开关才能生效
        self._recorder = VideoRecorder(self.camera_id, self._loop)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"stream-reader-{self.camera_id}",
            daemon=True,
        )
        self._reader_thread.start()
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            "StreamConsumer 已启动: camera={}, url={}, source={}",
            self.camera_id, self.url, source,
        )

    async def stop(self, source: str) -> None:
        """停止拉流与处理。

        source: 触发停止的入口(接口/租约到期/应用关闭等), 只进日志, 用于追踪谁停了这条流。
        """
        try:
            if not self.running:
                return
            self.running = False
            self._stop_event.set()

            if self._process_task is not None:
                self._process_task.cancel()
                try:
                    await self._process_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._process_task = None

            if self._reader_thread is not None:
                # 读流线程是 daemon, join 超时不阻塞关闭流程
                await asyncio.to_thread(self._reader_thread.join, 5.0)
                self._reader_thread = None

            # 读流线程停稳后再收录像, 避免与 maybe_write 竞态
            recorder = self._recorder
            self._recorder = None
            if recorder is not None:
                try:
                    await recorder.stop()
                except Exception:
                    logger.exception("拉流录像收尾失败: camera={}", self.camera_id)

            self.connected = False
            # 清空运行时轨迹与注意力目标: 停止后 orchestrator 可能因为还有观看端
            # WebSocket 而不被回收(registry.maybe_release_orchestrator), 轨迹清理
            # 又只在 process_frame 里帧驱动地发生——不清的话 current_identity 会
            # 一直返回停流前最后一帧镜头前的人(身份"残影")。断流重连场景已由
            # _process_loop 的 reset_attention 覆盖, 这里补上显式停止这条路。
            self.orchestrator.reset_attention()
            logger.info(
                "StreamConsumer 已停止: camera={}, source={}", self.camera_id, source,
            )
        finally:
            # 无论停止流程中哪个清理步骤失败, 都不能让已结束实例继续占用注册表。
            from src.api.registry import unregister_stream_consumer
            unregister_stream_consumer(self.camera_id, self)

    def status(self) -> StreamStatusResponse:
        """当前状态快照。"""
        from src.api.registry import viewer_count

        return StreamStatusResponse(
            camera_id=self.camera_id,
            running=self.running,
            connected=self.connected,
            url=self.url,
            stream_width=self.stream_width,
            stream_height=self.stream_height,
            frames_read=self.frames_read,
            frames_processed=self.frames_processed,
            process_fps=round(self.process_fps, 1),
            viewers=viewer_count(self.camera_id),
            last_error=self.last_error,
            env=self.env,
            auto_restream=self.auto_restream,
            recovering=self._recovering,
            restream_count=self.restream_count,
            last_restream_at=self.last_restream_at,
            last_restream_outcome=self.last_restream_outcome,
        )

    # ------------------------------------------------------------------
    # 读流线程 (blocking IO, 独立线程)
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        # 独立线程有自己的 context: 补写 device_sn(= camera_id), 日志才带该列
        set_device_sn(self.camera_id)

        while not self._stop_event.is_set():
            # 热字段, 每次连接尝试现读
            reconnect_delay = current_config().stream_reconnect_delay
            cap = self._open_capture()
            if not cap.isOpened():
                # NvdecCapture 把探测阶段的原因放在 last_error; cv2 后端不暴露
                # 原因, 补一次 ffprobe 诊断(只在失败路径付这个代价)
                reason = getattr(cap, "last_error", None) or diagnose_open_failure(self.url)
                cap.release()
                self.connected = False
                self.last_error = f"无法打开视频流: {reason}"
                logger.warning(
                    "拉流打开失败: camera={}, url={}, 原因: {}, {}s 后重试",
                    self.camera_id, self.url, reason, reconnect_delay,
                )
                self._on_pull_failure()
                self._stop_event.wait(reconnect_delay)
                continue

            # 尽量压低解码缓冲, 降低画面延迟 (部分后端不支持, 失败无害)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.connected = True
            self.last_error = None
            logger.info("拉流已连接: camera={}, url={}", self.camera_id, self.url)

            session_frames = 0  # 本次连接读到的帧数 (区分健康会话与假连接)
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                session_frames += 1
                if session_frames == self._HEALTHY_SESSION_FRAMES:
                    # 流真正跑起来了, 之前的失败不再连续
                    self._consecutive_pull_failures = 0
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_seq += 1
                self.frames_read += 1
                if (frame.shape[1], frame.shape[0]) != (self.stream_width, self.stream_height):
                    self.stream_height, self.stream_width = frame.shape[:2]
                    logger.info(
                        "流分辨率: camera={}, {}x{}",
                        self.camera_id, self.stream_width, self.stream_height,
                    )
                recorder = self._recorder
                if recorder is not None:
                    try:
                        recorder.maybe_write(frame)
                    except Exception:
                        logger.exception(
                            "拉流录像写帧失败: camera={}", self.camera_id,
                        )

            # release 前取: NvdecCapture 在 read 失败时填好 ffmpeg 退出码 +
            # stderr 尾部; cv2 后端没有这个接口, 只能报"无详细原因"
            reason = getattr(cap, "last_error", None) or "cv2 后端无详细原因"
            cap.release()
            self.connected = False
            if not self._stop_event.is_set():
                self.last_error = "视频流中断, 正在重连"
                logger.warning(
                    "拉流中断: camera={}, 本次连接读到 {} 帧, {}s 后重连, 原因: {}",
                    self.camera_id, session_frames, reconnect_delay, reason,
                )
                self._on_pull_failure()
                self._stop_event.wait(reconnect_delay)

    def _open_capture(self) -> NvdecCapture | cv2.VideoCapture:
        """打开拉流解码器: 配置开启且本机支持时走 NVDEC(GPU 解码 ASIC,
        不占 CPU/CUDA 核心), NVDEC 环境异常时按次退回 cv2 CPU 软解
        (下次重连仍先试 GPU)。流本身不可达(设备未推流/超时)时不退回——
        CPU 解码同样打不开, 直接交给外层重连逻辑, 省一次连接超时。"""
        if current_config().stream_gpu_decode:
            from src.pipeline.gpu_decoder import NvdecCapture, nvdec_supported

            if nvdec_supported():
                cap = NvdecCapture(self.url, device=config.hardware.device)
                if cap.isOpened() or cap.stream_unreachable:
                    return cap
                cap.release()
                logger.warning(
                    "NVDEC 拉流打开失败, 本次退回 CPU 解码: camera={}, 原因: {}",
                    self.camera_id, cap.last_error,
                )
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _on_pull_failure(self) -> None:
        """(读流线程) 拉流失败的处理: 累计连续失败计数, 达到阈值时调度自动重推流。"""
        self._consecutive_pull_failures += 1
        if not self.auto_restream or self._recovering:
            return
        threshold = current_config().stream_restream_fail_threshold
        if self._consecutive_pull_failures < threshold:
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # 先置位再调度: 读流线程是唯一写入方, 恢复协程结束时复位
        self._recovering = True
        asyncio.run_coroutine_threadsafe(self._run_recovery(), loop)

    async def _run_recovery(self) -> None:
        """(事件循环) 执行自动重推流恢复并按结果切换拉流地址。"""
        from src.pipeline.restream import run_restream_recovery

        # 经 run_coroutine_threadsafe 从读流线程调度而来, 不继承其 context
        set_device_sn(self.camera_id)

        try:
            if not self.running:
                return
            fail_count = self._consecutive_pull_failures
            trigger_error = self.last_error or ""
            self.last_error = f"连续拉流失败 {fail_count} 次, 正在自动重推流..."

            attempt = await run_restream_recovery(
                camera_id=self.camera_id,
                env=self.env,
                current_url=self.url,
                fail_count=fail_count,
                last_error=trigger_error,
            )

            outcome = attempt.outcome
            self.last_restream_at = attempt.finished_at
            self.last_restream_outcome = outcome
            # 无论成败都清零计数: 失败时等再攒满一个阈值周期后重试,
            # 避免每次重连失败都打一遍 ISS / voice_server
            self._consecutive_pull_failures = 0

            if outcome == "restreamed":
                self.restream_count += 1
                self.url = attempt.new_url  # 读流线程下个循环用新地址
                self.last_error = "已自动重推流, 正在连接新直播地址..."
                # 期望状态里的地址同步换新, 服务重启恢复时直连可用地址
                from src.pipeline import stream_state
                await stream_state.update_url(self.camera_id, attempt.new_url)
            elif outcome == "device_offline":
                self.last_error = "设备不在线, 暂不重推流 (等设备上线后自动重试)"
            else:
                self.last_error = f"自动重推流失败 ({outcome}), 稍后自动重试"
        finally:
            self._recovering = False

    # ------------------------------------------------------------------
    # 处理协程 (event loop)
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        from src.api.registry import publish_to_viewers, viewer_count

        # 后台常驻任务: 显式绑定 device_sn, 不依赖创建方(REST 请求/启动恢复)的 context
        set_device_sn(self.camera_id)
        last_seq = 0
        last_done = time.perf_counter()

        while not self._stop_event.is_set():
            # 逐帧现读生效快照: 帧率/畸变矫正/限宽/预览编码参数都是热字段
            cfg = current_config()
            min_interval = 1.0 / max(cfg.stream_max_fps, 1.0)

            with self._frame_lock:
                seq = self._latest_seq
                frame = self._latest_frame

            if frame is None or seq == last_seq:
                # 断流期间没有新帧, 帧驱动的轨迹清理不会再发生——在这里把
                # 运行时轨迹/注意力目标清掉 (幂等), 避免身份查询读到断流前
                # 最后一帧的残影。cancel_vlm 取消的是 asyncio 任务, 必须在
                # 事件循环内做, 所以放处理协程而不是读流线程的重连分支。
                if not self.connected:
                    self.orchestrator.reset_attention()
                await asyncio.sleep(0.01)
                continue
            last_seq = seq

            t0 = time.perf_counter()
            has_viewers = viewer_count(self.camera_id) > 0
            try:
                # 识别路径: 默认原生分辨率 + 无 JPEG 重压缩, 不引入任何画质损失
                frame = self._prepare_frame(frame, cfg)
                result = await self.orchestrator.process_frame(frame)
                events = self.orchestrator.drain_new_events()
                # 录像的"看到人"信号: require_person 模式据此开段/收段
                # (写帧在读流线程, 识别在这里, 经 recorder 内部状态桥接)
                recorder = self._recorder
                if recorder is not None:
                    recorder.notify_person(bool(result.get("tracked_persons")))
                # 预览路径: 仅供网页观看, 可独立缩放省带宽 (不影响识别)。
                # 无观看端时跳过 JPEG 编码——它是识别推理之外最大的单路 CPU
                # 项(720p 每帧数 ms), 大规模路数下绝大多数摄像头无人在看
                jpeg = self._encode_preview(frame, cfg) if has_viewers else None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"帧处理失败: {e}"
                logger.exception("拉流帧处理失败: camera={}", self.camera_id)
                await asyncio.sleep(0.5)
                continue

            self.frames_processed += 1

            # 处理帧率 EMA
            now = time.perf_counter()
            dt = now - last_done
            last_done = now
            if dt > 0:
                self.process_fps = 0.8 * self.process_fps + 0.2 * (1.0 / dt)

            frame_result = build_frame_result(result)
            # 检测坐标的基准尺寸: 预览图可能缩小过, 前端须按此映射框位置
            frame_result.frame_h, frame_result.frame_w = frame.shape[:2]
            # 角标与状态接口同口径, 避免前端按绘制间隔估算虚高
            frame_result.process_fps = round(self.process_fps, 1)

            # jpeg=None 表示本帧没编预览(无观看端); 不广播——viewer_sender
            # 直接 send_bytes(jpeg), None 会炸。check 与 publish 之间新连上的
            # 观看端最多等一帧(下一轮 has_viewers 即为真), 无感
            if jpeg is not None:
                publish_to_viewers(self.camera_id, {
                    "jpeg": jpeg,
                    "result": frame_result.model_dump(mode="json"),
                    "events": [
                        build_ws_event(ev).model_dump(mode="json") for ev in events
                    ],
                })

            # 帧率上限节流 + 强制让出事件循环。
            # process_frame 是 async 但内部全是同步 GPU 推理 (Tier1 ~60-80ms,
            # Tier2 尖峰数百 ms), 没有真正的 await 点; 处理耗时超过 min_interval
            # 时若不 sleep 直接进下一轮, 本循环会独占事件循环 —— viewer 推帧
            # 协程与 REST 全被饿死 (实测观看端只发得出 ~1fps 且 TCP app_limited,
            # 状态接口要 2-4s 才响应)。每帧至少让出 10ms 保证其他协程被调度。
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(min_interval - elapsed, 0.01))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_max_width(frame: np.ndarray, max_w: int) -> np.ndarray:
        """限宽等比缩放; max_w<=0 或未超宽时原样返回。"""
        if max_w <= 0 or frame.shape[1] <= max_w:
            return frame
        h, w = frame.shape[:2]
        new_h = round(h * max_w / w)
        return cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)

    @classmethod
    def _prepare_frame(cls, frame: np.ndarray, cfg) -> np.ndarray:
        """识别前预处理: 畸变矫正 + 可选限宽。

        stream_proc_max_width=0 (默认) 时不缩放, 按视频流原生分辨率识别,
        保证人脸/ReID 裁剪拿到的是无损画质。
        """
        if cfg.image_correction_enabled:
            try:
                from src.utils.image_correction import correct_frame
                frame = correct_frame(frame)
            except Exception:
                logger.warning("拉流帧畸变矫正失败, 使用原始帧")

        return cls._resize_max_width(frame, cfg.stream_proc_max_width)

    @classmethod
    def _encode_preview(cls, frame: np.ndarray, cfg) -> bytes:
        """编码前端预览帧: 独立限宽 + JPEG, 只影响观看带宽, 不影响识别。"""
        frame = cls._resize_max_width(frame, cfg.stream_preview_max_width)
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(cfg.stream_preview_jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()
