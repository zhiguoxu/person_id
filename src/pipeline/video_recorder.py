"""拉流录像 — 随 StreamConsumer 生命周期落盘并上传 COS

语义:
- consume/start → 开录; consume/stop / 租约到期 → 收尾上传
- 断线重连、自动重推流换址: 同一 StreamConsumer 内继续写, 算同一拉流会话
  (共享 stream_session_id); 分辨率变化时被迫切新段(VideoWriter 尺寸不可变)
- 单段超过 video_record.max_seconds 自动切段, 避免单个文件过大

实现要点:
- 在读流线程按 video_record.fps 节流写帧(与识别处理解耦)
- 本地临时 mp4 → COS videos/{device_sn}/{started_at}.mp4 → session_store 元数据
- 元数据在上传成功后落库(失败则写 status=failed); 不阻断拉流/识别

并发模型: 写帧在读流线程, 收尾上传在事件循环。当前段的全部状态收敛在
一个 _Segment 对象里, 由 self._segment 持有并受 self._lock 保护——
收段就是把对象整体从 recorder 上摘下来(之后只有收尾协程持有它, 天然
不再与写帧竞争), 而不是逐字段拷贝/复位。
"""
from __future__ import annotations

import asyncio
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

from src.configs.config import PROJECT_ROOT, config

VIDEO_COS_PREFIX = "videos"
_TEMP_DIR = PROJECT_ROOT / "data" / "video_record"


@dataclass
class _Segment:
    """一个正在写入的录制段(writer + 本地文件 + 统计)。"""
    writer: cv2.VideoWriter
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

    @property
    def stream_session_id(self) -> str:
        return self._stream_session_id

    # ------------------------------------------------------------------
    # 读流线程入口
    # ------------------------------------------------------------------

    def maybe_write(self, frame: np.ndarray) -> None:
        """读流线程: 有新帧时按配置帧率写入; 超长/分辨率变化时切段。"""
        if self._closed or frame is None or frame.size == 0:
            return
        cfg = config.video_record
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
                if now - seg.mono_start >= max_seconds:
                    finished, reason = self._detach_segment(), "max_duration"
                else:
                    h, w = frame.shape[:2]
                    if self._target_size(w, h, cfg.max_width) != (seg.width, seg.height):
                        finished, reason = self._detach_segment(), "resolution_change"

            if self._segment is None:
                self._segment = self._open_segment(frame, cfg)
            if self._segment is not None:
                self._segment.write(self._prepare_frame(frame, cfg.max_width))

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
        """(持锁) 按当前帧尺寸开新段; VideoWriter 打开失败返回 None。"""
        h, w = frame.shape[:2]
        width, height = self._target_size(w, h, cfg.max_width)
        fps = max(float(cfg.fps), 0.1)
        started_at = now_cst()
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = _TEMP_DIR / f"{self._device_sn}_{started_at:%Y%m%d_%H%M%S_%f}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            logger.warning("录像 VideoWriter 打开失败: device={} path={}",
                           self._device_sn, path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

        logger.info(
            "🎥 开始录像: device={} session={} {}x{}@{:.1f} → {}",
            self._device_sn, self._stream_session_id[:8], width, height, fps, path.name,
        )
        return _Segment(
            writer=writer, path=path, started_at=started_at,
            width=width, height=height, fps=fps,
        )

    def _detach_segment(self) -> _Segment | None:
        """(持锁) 把当前段整体摘下来交给收尾协程, recorder 回到无段状态。"""
        seg = self._segment
        self._segment = None
        return seg

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
            await deps.cos_client.upload_file(data, cos_key)
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
