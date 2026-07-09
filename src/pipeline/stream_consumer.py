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
"""
from __future__ import annotations

import asyncio
import threading
import time

import cv2
import numpy as np
from loguru import logger

from src.api.schemas import (
    StreamStatusResponse,
    build_frame_result,
    build_ws_event,
)
from src.config import get_config
from src.pipeline.orchestrator import VisionOrchestrator


class StreamConsumer:
    """单摄像头的后台拉流消费器。"""

    def __init__(
        self,
        camera_id: str,
        url: str,
        orchestrator: VisionOrchestrator,
    ) -> None:
        self.camera_id = camera_id
        self.url = url
        self.orchestrator = orchestrator

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动读流线程与处理协程。"""
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"stream-reader-{self.camera_id}",
            daemon=True,
        )
        self._reader_thread.start()
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(
            "StreamConsumer 已启动: camera={}, url={}", self.camera_id, self.url,
        )

    async def stop(self) -> None:
        """停止拉流与处理。"""
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

        self.connected = False
        logger.info("StreamConsumer 已停止: camera={}", self.camera_id)

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
        )

    # ------------------------------------------------------------------
    # 读流线程 (blocking IO, 独立线程)
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        reconnect_delay = get_config().server.stream_reconnect_delay

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self.connected = False
                self.last_error = f"无法打开视频流: {self.url}"
                logger.warning(
                    "拉流打开失败: camera={}, url={}, {}s 后重试",
                    self.camera_id, self.url, reconnect_delay,
                )
                self._stop_event.wait(reconnect_delay)
                continue

            # 尽量压低解码缓冲, 降低画面延迟 (部分后端不支持, 失败无害)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.connected = True
            self.last_error = None
            logger.info("拉流已连接: camera={}, url={}", self.camera_id, self.url)

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
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

            cap.release()
            self.connected = False
            if not self._stop_event.is_set():
                self.last_error = "视频流中断, 正在重连"
                logger.warning(
                    "拉流中断: camera={}, {}s 后重连",
                    self.camera_id, reconnect_delay,
                )
                self._stop_event.wait(reconnect_delay)

    # ------------------------------------------------------------------
    # 处理协程 (event loop)
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        from src.api.registry import publish_to_viewers

        last_seq = 0
        last_done = time.perf_counter()

        while not self._stop_event.is_set():
            cfg = get_config().server
            min_interval = 1.0 / max(cfg.stream_max_fps, 1.0)

            with self._frame_lock:
                seq = self._latest_seq
                frame = self._latest_frame

            if frame is None or seq == last_seq:
                await asyncio.sleep(0.01)
                continue
            last_seq = seq

            t0 = time.perf_counter()
            try:
                # 识别路径: 默认原生分辨率 + 无 JPEG 重压缩, 不引入任何画质损失
                frame = self._prepare_frame(frame, cfg)
                result = await self.orchestrator.process_frame(frame)
                events = self.orchestrator.drain_new_events()
                # 预览路径: 仅供网页观看, 可独立缩放省带宽 (不影响识别)
                jpeg = self._encode_preview(frame, cfg)
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

            publish_to_viewers(self.camera_id, {
                "jpeg": jpeg,
                "result": frame_result.model_dump(mode="json"),
                "events": [
                    build_ws_event(ev).model_dump(mode="json") for ev in events
                ],
            })

            # 帧率上限节流
            elapsed = time.perf_counter() - t0
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

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
