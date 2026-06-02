"""
Tier 1 流水线 — 快速逐帧追踪

每帧执行:
1. 轻量 YOLO11n-pose 检测
2. BoT-SORT 追踪器更新
3. 身份缓存查询 (track_id → person_id)
4. 注意力评分更新

输出 TrackedPerson 列表, 供 Tier 2 判定是否需要深度识别。
"""
from __future__ import annotations

import time
from typing import Any, Optional, Protocol

import numpy as np
from loguru import logger

from src.config import Config
from src.gallery.data_models import (
    Detection,
    IdentityStatus,
    PoseBucket,
    TrackedPerson,
)


# ---------------------------------------------------------------------------
# 依赖接口协议 (duck-typing contracts)
# ---------------------------------------------------------------------------

class DetectorProtocol(Protocol):
    """YOLO 检测器接口。"""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class TrackerProtocol(Protocol):
    """多目标追踪器接口。"""

    def update(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> list[TrackedPerson]:
        """返回 TrackedPerson 列表。"""
        ...


class AttentionProtocol(Protocol):
    """注意力评分引擎接口。"""

    def compute_scores(
        self, tracked_persons: list[TrackedPerson], frame_shape: tuple[int, ...]
    ) -> list[float]:
        """为每个 TrackedPerson 计算注意力分 [0, 1]。"""
        ...


class Tier1Processor:
    """
    Tier 1 处理器：快速逐帧检测 + 追踪 + 缓存查询。

    在机器人主循环中以全帧率运行，使用轻量 YOLO11n 和
    BoT-SORT 维护多人追踪，并利用身份缓存避免重复识别。

    Args:
        config: 全局配置。
        detector: 轻量 YOLO 检测器（Tier 1）。
        tracker: 多目标追踪器。
        attention_engine: 注意力评分引擎（可选）。
    """

    def __init__(
        self,
        config: Config,
        detector: Any,
        tracker: Any,
        attention_engine: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._detector = detector
        self._tracker = tracker
        self._attention = attention_engine

        # track_id → 缓存的身份信息
        self._identity_cache: dict[int, _CachedIdentity] = {}

        # 轨迹历史 (track_id → 最近 N 帧中心点)
        self._trail_buffer: dict[int, list[tuple[float, float]]] = {}
        self._max_trail_len = 60  # 保留约 2 秒 @30fps

        logger.info("Tier1Processor initialized (fast detector + tracker)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> list[TrackedPerson]:
        """
        处理单帧图像，返回追踪人物列表。

        Args:
            frame: BGR 图像 (H, W, 3)。

        Returns:
            当前帧的 TrackedPerson 列表（已含缓存身份和注意力分）。
        """
        t0 = time.perf_counter()

        # 1. 检测
        try:
            detections = self._detector.detect(frame)
        except Exception:
            logger.exception("Tier1 detection failed")
            detections = []

        # 2. 追踪 — TrackingEngine.update(frame, detections) -> list[TrackedPerson]
        try:
            tracked_persons = self._tracker.update(frame, detections)
        except Exception:
            logger.exception("Tier1 tracker update failed")
            tracked_persons = []

        # 3. 注入缓存身份 + 更新轨迹
        persons: list[TrackedPerson] = []
        active_ids: set[int] = set()

        for person in tracked_persons:
            track_id = person.track_id
            active_ids.add(track_id)

            # 用缓存身份覆盖 (如果有)
            cached = self._identity_cache.get(track_id)
            if cached:
                person.person_id = cached.person_id
                person.display_name = cached.display_name
                person.identity_status = cached.status
                person.confidence = cached.confidence
                person.face_quality = cached.face_quality
                person.last_tier2_time = cached.last_tier2_time

            persons.append(person)

        # 4. 注意力评分
        if self._attention and persons:
            try:
                scores = self._attention.compute_scores(persons, frame.shape)
                for person, score in zip(persons, scores):
                    person.attention_score = score
            except Exception:
                logger.exception("Attention scoring failed")

        # 5. 清理过期轨迹缓存
        self._cleanup_stale(active_ids)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Tier1: {} detections, {} tracked, {:.1f}ms",
            len(detections),
            len(persons),
            elapsed_ms,
        )

        return persons

    # ------------------------------------------------------------------
    # 身份缓存管理
    # ------------------------------------------------------------------

    def update_identity(
        self,
        track_id: int,
        person_id: Optional[str],
        display_name: Optional[str],
        status: IdentityStatus,
        confidence: float,
        face_quality: Optional[float] = None,
    ) -> None:
        """
        更新指定 track 的缓存身份信息（由 Tier 2 或人工确认后调用）。

        Args:
            track_id: 轨迹 ID。
            person_id: 底库人物 ID。
            display_name: 显示名称。
            status: 身份状态。
            confidence: 置信度。
            face_quality: 人脸质量。
        """
        self._identity_cache[track_id] = _CachedIdentity(
            person_id=person_id,
            display_name=display_name,
            status=status,
            confidence=confidence,
            face_quality=face_quality,
            last_tier2_time=time.time(),
        )

    def get_cached_identity(self, track_id: int) -> Optional[_CachedIdentity]:
        """获取 track 的缓存身份。"""
        return self._identity_cache.get(track_id)

    def clear_cache(self) -> None:
        """清空身份缓存。"""
        self._identity_cache.clear()
        self._trail_buffer.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup_stale(self, active_ids: set[int]) -> None:
        """清除不再活跃的轨迹数据。"""
        stale_ids = set(self._trail_buffer.keys()) - active_ids
        for sid in stale_ids:
            self._trail_buffer.pop(sid, None)
            # 身份缓存保留更长时间（空间约束用）
            # 只清理轨迹


class _CachedIdentity:
    """track_id → 身份信息的内部缓存条目。"""

    __slots__ = (
        "person_id",
        "display_name",
        "status",
        "confidence",
        "face_quality",
        "last_tier2_time",
    )

    def __init__(
        self,
        person_id: Optional[str],
        display_name: Optional[str],
        status: IdentityStatus,
        confidence: float,
        face_quality: Optional[float],
        last_tier2_time: float,
    ) -> None:
        self.person_id = person_id
        self.display_name = display_name
        self.status = status
        self.confidence = confidence
        self.face_quality = face_quality
        self.last_tier2_time = last_tier2_time
