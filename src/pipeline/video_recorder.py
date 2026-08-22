"""拉流录像 — 随 StreamConsumer 生命周期落盘并上传 COS

语义:
- consume/start → 开录; consume/stop / 租约到期 → 收尾上传
- 断线重连、自动重推流换址: 同一 StreamConsumer 内继续写, 算同一拉流会话
  (共享 stream_session_id); 分辨率变化时被迫切新段(VideoWriter 尺寸不可变)
- 单段超过 video_record.max_seconds 自动切段, 避免单个文件过大
- require_person 模式: 只有识别结果里有人时才开段; 连续 no_person_seconds
  秒无人自动收段上传, 下次看到人另起新段(处理协程经 notify_person 喂
  可见性信号, 见 StreamConsumer._process_loop)

实现要点:
- 在读流线程按 video_record.fps 节流写帧(与识别处理解耦)
- 编码走 ffmpeg 子进程出 H.264: 网页 <video> 只认 H.264/VP9/AV1,
  cv2 内置的 mp4v(MPEG-4 Part 2) 浏览器放不了, 只作 ffmpeg 缺失时的
  兜底(仍可下载后本地播放)
- 本地临时 mp4 → COS videos/{device_sn}/{started_at}.mp4 → session_store 元数据
- 元数据在上传成功后落库(失败则写 status=failed); 不阻断拉流/识别

并发模型: 写帧在读流线程, 收尾上传在事件循环。当前段的全部状态收敛在
一个 _Segment 对象里, 由 self._segment 持有并受 self._lock 保护——
收段就是把对象整体从 recorder 上摘下来(之后只有收尾协程持有它, 天然
不再与写帧竞争), 而不是逐字段拷贝/复位。
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from voice_agent_common.utils.clock import now_cst
from voice_agent_common.utils.logger import logger

from src.configs.config import PROJECT_ROOT
from src.configs.override import current_config

VIDEO_COS_PREFIX = "videos"
_TEMP_DIR = PROJECT_ROOT / "data" / "video_record"


# 编码器参数集。两套都出 H.264 + yuv420p + faststart(moov 在文件头,
# 网页拿到临时链即可边下边播):
# - nvenc: 显卡独立编码 ASIC, 不占 CUDA 核心也基本不占 CPU, 大规模路数必选
# - x264: 纯 CPU 兜底; -threads 2 限住单实例线程数, 避免几十路同录时
#   每个 ffmpeg 默认起满核数线程互相踩踏
_CODEC_ARGS: dict[str, list[str]] = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "30"],
    "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-threads", "2"],
}

# NVENC 一次性运行时探测结果(进程级缓存)
_nvenc_available: bool | None = None


def _probe_nvenc() -> bool:
    """真编一帧黑帧探测 NVENC: ffmpeg 构建缺 nvenc / 驱动不匹配都能暴露。

    只在首段开录时跑一次(~百 ms), 结果进程级缓存。
    """
    global _nvenc_available
    if _nvenc_available is None:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=black:size=64x64:rate=1:duration=1",
                    "-c:v", "h264_nvenc", "-f", "null", "-",
                ],
                capture_output=True, timeout=15,
            )
            _nvenc_available = proc.returncode == 0
        except Exception:
            _nvenc_available = False
        logger.info("NVENC 编码探测: {}",
                    "可用" if _nvenc_available else "不可用(将用 libx264)")
    return _nvenc_available


def _select_codec(encoder: str) -> str:
    """按配置选编码器: auto 优先 NVENC; 强制 nvenc 但探测失败时退 x264。"""
    if encoder == "x264":
        return "x264"
    if _probe_nvenc():
        return "nvenc"
    if encoder == "nvenc":
        logger.warning("配置强制 nvenc 但探测不可用, 退回 libx264")
    return "x264"


class _FfmpegWriter:
    """H.264 mp4 编码器(ffmpeg 子进程, 与 cv2.VideoWriter 同接口)。"""

    def __init__(self, path: Path, width: int, height: int, fps: float,
                 codec: str) -> None:
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:g}",
                "-i", "pipe:0",
                *_CODEC_ARGS[codec],
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        self._proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


@dataclass
class _Segment:
    """一个正在写入的录制段(writer + 本地文件 + 统计)。"""
    writer: _FfmpegWriter | cv2.VideoWriter
    path: Path
    started_at: datetime
    width: int
    height: int
    fps: float
    mono_start: float = field(default_factory=time.monotonic)
    last_write_mono: float = 0.0
    frame_count: int = 0

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)
        self.frame_count += 1
        self.last_write_mono = time.monotonic()


class VideoRecorder:
    """挂在单个 StreamConsumer 上的录像器(同一摄像头同时至多一个)。"""

    def __init__(self, device_sn: str, loop: asyncio.AbstractEventLoop) -> None:
        self._device_sn = device_sn
        self._loop = loop
        self._stream_session_id = uuid.uuid4().hex
        self._closed = False
        self._lock = threading.Lock()
        # 当前段; None = 没有在写的段
        self._segment: _Segment | None = None
        # 人物可见性(require_person 门控): 处理协程按每帧识别结果调
        # notify_person 更新。单个 float/bool 赋值受 GIL 保护, 读流线程
        # 直接读即可, 不需要进 self._lock(notify 在事件循环线程跑,
        # 拿锁会被 _open_segment 里的 ffmpeg 启动等慢操作卡住)
        self._person_visible = False
        self._last_person_mono = 0.0

    @property
    def stream_session_id(self) -> str:
        return self._stream_session_id

    def notify_person(self, seen: bool) -> None:
        """事件循环: 每处理帧回报识别结果里是否有人(require_person 门控)。"""
        self._person_visible = seen
        if seen:
            self._last_person_mono = time.monotonic()

    # ------------------------------------------------------------------
    # 读流线程入口
    # ------------------------------------------------------------------

    def maybe_write(self, frame: np.ndarray) -> None:
        """读流线程: 有新帧时按配置帧率写入; 超长/分辨率变化时切段。"""
        if self._closed or frame is None or frame.size == 0:
            return
        # 全部字段热生效, 逐帧现读
        cfg = current_config().video_record
        if not cfg.enabled:
            return

        min_interval = 1.0 / max(float(cfg.fps), 0.1)
        max_seconds = max(float(cfg.max_seconds), 60.0)

        now = time.monotonic()
        finished: _Segment | None = None
        reason = ""
        with self._lock:
            if self._closed:
                return
            seg = self._segment
            if seg is not None and seg.last_write_mono:
                if now - seg.last_write_mono < min_interval:
                    return
                if (cfg.require_person
                        and now - self._last_person_mono
                        >= max(float(cfg.no_person_seconds), 1.0)):
                    # 连续无人超时: 收段上传(段尾自然带上这段无人画面),
                    # 且下面不开新段(可见性门控挡住), 等再看到人另起新视频
                    finished, reason = self._detach_segment(), "no_person"
                elif now - seg.mono_start >= max_seconds:
                    finished, reason = self._detach_segment(), "max_duration"
                else:
                    h, w = frame.shape[:2]
                    if self._target_size(w, h, cfg.max_width) != (seg.width, seg.height):
                        finished, reason = self._detach_segment(), "resolution_change"

            if self._segment is None:
                if cfg.require_person and not (
                    self._person_visible
                    and now - self._last_person_mono
                    < max(float(cfg.person_fresh_seconds), 0.1)
                ):
                    if finished is None:
                        return
                else:
                    self._segment = self._open_segment(frame, cfg)
            if self._segment is not None:
                try:
                    self._segment.write(self._prepare_frame(frame, cfg.max_width))
                except Exception as e:
                    # 编码器挂了(如 ffmpeg 进程异常退出): 丢弃当前段,
                    # 下一帧自动开新段, 不逐帧刷报错
                    broken = self._detach_segment()
                    logger.warning("录像写帧失败, 丢弃当前段: device={} ({})",
                                   self._device_sn, e)
                    self._discard(broken)

        if finished is not None:
            logger.info(
                "🎥 录像切段({}): device={} frames={} → {}",
                reason, self._device_sn, finished.frame_count, finished.path.name,
            )
            asyncio.run_coroutine_threadsafe(
                self._finalize_and_upload(finished), self._loop)

    async def stop(self) -> None:
        """事件循环: 停止录制并等待最后一段上传落库。"""
        with self._lock:
            self._closed = True
            finished = self._detach_segment()
        if finished is not None:
            await self._finalize_and_upload(finished)

    # ------------------------------------------------------------------
    # 段生命周期
    # ------------------------------------------------------------------

    def _open_segment(self, frame: np.ndarray, cfg) -> _Segment | None:
        """(持锁) 按当前帧尺寸开新段; 编码器打开失败返回 None。"""
        h, w = frame.shape[:2]
        width, height = self._target_size(w, h, cfg.max_width)
        fps = max(float(cfg.fps), 0.1)
        started_at = now_cst()
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = _TEMP_DIR / f"{self._device_sn}_{started_at:%Y%m%d_%H%M%S_%f}.mp4"

        writer = self._open_writer(path, width, height, fps)
        if writer is None:
            return None

        logger.info(
            "🎥 开始录像: device={} session={} {}x{}@{:.1f} → {}",
            self._device_sn, self._stream_session_id[:8], width, height, fps, path.name,
        )
        return _Segment(
            writer=writer, path=path, started_at=started_at,
            width=width, height=height, fps=fps,
        )

    def _open_writer(
        self, path: Path, width: int, height: int, fps: float,
    ) -> _FfmpegWriter | cv2.VideoWriter | None:
        """优先 ffmpeg 出 H.264(网页可播); 缺 ffmpeg 时退回 cv2 mp4v。"""
        if shutil.which("ffmpeg"):
            codec = _select_codec(current_config().video_record.encoder)
            try:
                return _FfmpegWriter(path, width, height, fps, codec)
            except OSError as e:
                logger.warning("ffmpeg 启动失败, 退回 cv2 编码(网页不可播): {}", e)

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            logger.warning("录像 VideoWriter 打开失败: device={} path={}",
                           self._device_sn, path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        logger.warning("录像使用 mp4v 兜底编码(浏览器无法预览, 仅可下载): device={}",
                       self._device_sn)
        return writer

    def _detach_segment(self) -> _Segment | None:
        """(持锁) 把当前段整体摘下来交给收尾协程, recorder 回到无段状态。"""
        seg = self._segment
        self._segment = None
        return seg

    def _discard(self, seg: _Segment | None) -> None:
        """废弃一个段: 关编码器、删本地文件, 不上传不落库。"""
        if seg is None:
            return
        try:
            seg.writer.release()
        except Exception:
            pass
        try:
            seg.path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 收尾: 关 writer → 上传 COS → 落库 (事件循环, 独占持有 segment)
    # ------------------------------------------------------------------

    async def _finalize_and_upload(self, seg: _Segment) -> None:
        from session_store import VideoStore
        from src import deps

        duration_ms = int(max(time.monotonic() - seg.mono_start, 0.0) * 1000)
        # release 做收尾 flush/mux, 可能耗时, 不在事件循环里同步做
        await asyncio.to_thread(seg.writer.release)
        if seg.frame_count <= 0 or not seg.path.exists() or seg.path.stat().st_size <= 0:
            try:
                seg.path.unlink(missing_ok=True)
            except OSError:
                pass
            return

        ended_at = now_cst()
        cos_key = (
            f"{VIDEO_COS_PREFIX}/{self._device_sn}/"
            f"{seg.started_at:%Y%m%d_%H%M%S_%f}.mp4"
        )
        video_id: int | None = None
        try:
            video_id = await VideoStore.create(
                device_sn=self._device_sn,
                stream_session_id=self._stream_session_id,
                started_at=seg.started_at,
                width=seg.width,
                height=seg.height,
                fps=seg.fps,
            )
            if deps.cos_client is None:
                raise RuntimeError("CosClient 未初始化")
            data = await asyncio.to_thread(seg.path.read_bytes)
            await deps.cos_client.upload_file(data, cos_key, content_type="video/mp4")
            await VideoStore.mark_ready(
                video_id,
                ended_at=ended_at,
                duration_ms=duration_ms,
                frame_count=seg.frame_count,
                cos_key=cos_key,
                width=seg.width,
                height=seg.height,
                fps=seg.fps,
            )
            logger.info(
                "🎥 录像已上传: device={} {}ms frames={} → {}",
                self._device_sn, duration_ms, seg.frame_count, cos_key,
            )
        except Exception as e:
            logger.warning("录像上传/落库失败({}): {}", cos_key, e)
            if video_id is not None:
                try:
                    await VideoStore.mark_failed(video_id, str(e))
                except Exception:
                    logger.exception("录像失败状态写入失败: id={}", video_id)
        finally:
            try:
                seg.path.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _target_size(w: int, h: int, max_w: int) -> tuple[int, int]:
        """录制目标尺寸: 可选限宽等比缩放, 并取偶(VideoWriter 要求偶数尺寸)。"""
        if max_w <= 0 or w <= max_w:
            return (w - w % 2, h - h % 2)
        new_h = round(h * max_w / w)
        return (max_w - max_w % 2, new_h - new_h % 2)

    @classmethod
    def _prepare_frame(cls, frame: np.ndarray, max_w: int) -> np.ndarray:
        w, h = frame.shape[1], frame.shape[0]
        tw, th = cls._target_size(w, h, max_w)
        if (tw, th) != (w, h):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        return frame
