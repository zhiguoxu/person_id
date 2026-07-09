"""
追踪引擎 — 封装 BoT-SORT 多目标追踪器

提供帧级追踪更新、轨迹管理。
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from boxmot.trackers import BotSort  # type: ignore[import-untyped]
except ImportError:
    # Legacy boxmot (<v18) exported BoTSORT from root
    from boxmot import BoTSORT as BotSort  # type: ignore[import-untyped]
from loguru import logger

from src.config import get_config
from src.pipeline.data_models import (
    Detection,
    TrackedPerson,
)


class TrackingEngine:
    """多目标追踪引擎。

    封装 BoT-SORT (boxmot)，提供:
    - 帧级追踪更新 (Detection → TrackedPerson)
    - 中心轨迹缓冲 (trails)
    """

    _MAX_TRAIL_LEN: int = 30

    def __init__(self) -> None:
        """初始化追踪引擎。"""
        self._trails: dict[int, list[tuple[float, float]]] = {}
        self._tracker = self._create_tracker()
        self._frame_shape: tuple[int, int] | None = None
        logger.info("TrackingEngine 已初始化 — backend=BoT-SORT")

    def reset(self) -> None:
        """重建底层追踪器并清空轨迹。

        输入源切换 (前端推流 ↔ 服务端拉流) 或流分辨率变化时调用:
        BoT-SORT 的光流 GMC 要求前后帧同尺寸, 尺寸跳变会断言失败,
        且失败后内部残留旧尺寸帧, 不重建无法自愈。
        """
        self._tracker = self._create_tracker()
        self._trails.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
            self,
            frame: np.ndarray,
            detections: list[Detection],
    ) -> list[TrackedPerson]:
        """运行一帧追踪更新。

        Args:
            frame: 当前视频帧 (BGR, H×W×3)。
            detections: 当前帧的人体检测列表。

        Returns:
            追踪后的 TrackedPerson 列表。
        """
        # 0. 帧尺寸变化 (切换输入源/流分辨率) → 先重建追踪器, 防光流 GMC 断言崩溃
        shape = (frame.shape[0], frame.shape[1])
        if shape != self._frame_shape:
            if self._frame_shape is not None:
                logger.info("帧尺寸变化 {} → {}, 重建追踪器", self._frame_shape, shape)
                self.reset()
            self._frame_shape = shape

        # 1. Convert detections to (N, 6) array for tracker
        det_array = self._detections_to_array(detections)

        # 2. Run tracker
        try:
            results = self._tracker.update(det_array, frame)
        except Exception:
            # update 非事务性: 异常可能把内部状态留在半更新的不一致态, 且这种损坏
            # 往往自持 (逐帧持续失败)。重建换回干净状态, with_reid=False 时代价极低。
            logger.exception("Tracker 更新失败, 已重建追踪器, 本帧返回空结果")
            self.reset()
            return []

        if results is None or len(results) == 0:
            return []

        # 3. Build Detection lookup by bbox for keypoint mapping
        det_lookup = self._build_detection_lookup(detections)

        # 4. Convert results to TrackedPerson
        tracked_persons: list[TrackedPerson] = []
        active_ids: set[int] = set()

        for row in results:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            conf = float(row[5])
            active_ids.add(track_id)

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Find best matching original Detection for keypoints
            det = self._match_detection(bbox, det_lookup)

            # Build Detection for this tracked result
            if det is not None:
                tracked_det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    keypoints=det.keypoints,
                    pose_bucket=det.pose_bucket,
                    has_face=det.has_face,
                )
            else:
                tracked_det = Detection(
                    bbox=bbox,
                    confidence=conf,
                    keypoints=np.zeros((17, 3), dtype=np.float32),
                )

            # Update trail
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            trail = self._trails.setdefault(track_id, [])
            trail.append((cx, cy))
            if len(trail) > self._MAX_TRAIL_LEN:
                trail[:] = trail[-self._MAX_TRAIL_LEN:]

            person = TrackedPerson(
                track_id=track_id,
                detection=tracked_det,
                trail=list(trail),
            )
            tracked_persons.append(person)

        # 5. Clean up stale trails for lost tracks
        self._cleanup_stale(active_ids)

        return tracked_persons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_tracker() -> BotSort:
        """Create the BoT-SORT tracker backend."""
        config = get_config()
        tracker = BotSort(
            reid_model=None,
            track_high_thresh=config.tracking.track_high_thresh,
            track_low_thresh=config.tracking.track_low_thresh,
            new_track_thresh=config.tracking.new_track_thresh,
            track_buffer=config.tracking.track_buffer,
            match_thresh=config.tracking.match_thresh,
            cmc_method=config.tracking.cmc_method,
            with_reid=False,
        )
        logger.info("BoT-SORT tracker 创建成功")
        return tracker

    @staticmethod
    def _detections_to_array(detections: list[Detection]) -> np.ndarray:
        """Convert list of Detection to (N, 6) array: [x1, y1, x2, y2, conf, cls]."""
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        rows = []
        for det in detections:
            rows.append([
                det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                det.confidence,
                0.0,  # cls = 0 (person)
            ])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _build_detection_lookup(
            detections: list[Detection],
    ) -> list[tuple[np.ndarray, Detection]]:
        """Build a list of (center, Detection) for matching."""
        return [(det.center, det) for det in detections]

    @staticmethod
    def _match_detection(
            bbox: np.ndarray,
            lookup: list[tuple[np.ndarray, Detection]],
            max_dist: float = 100.0,
    ) -> Detection | None:
        """Match a tracked bbox to the closest original Detection by center distance."""
        if not lookup:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center = np.array([cx, cy])
        best_det: Detection | None = None
        best_dist = max_dist
        for det_center, det in lookup:
            dist = float(np.linalg.norm(center - det_center))
            if dist < best_dist:
                best_dist = dist
                best_det = det
        return best_det

    def _cleanup_stale(self, active_ids: set[int]) -> None:
        """Remove trail entries for tracks no longer active."""
        stale_trail = [tid for tid in self._trails if tid not in active_ids]
        for tid in stale_trail:
            self._trails.pop(tid, None)
