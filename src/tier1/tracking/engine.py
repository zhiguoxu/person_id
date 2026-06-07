"""
追踪引擎 — 封装 BoT-SORT 多目标追踪器

提供帧级追踪更新、身份缓存注入、轨迹管理。
当 boxmot 未安装时自动降级为简易 IOU 追踪器。
"""
from __future__ import annotations


import numpy as np
from loguru import logger

from src.config import get_config
from src.gallery.data_models import (
    Detection,
    TrackedPerson,
)

# ---------------------------------------------------------------------------
# BoT-SORT 后端 (惰性导入)
# ---------------------------------------------------------------------------

_BOXMOT_AVAILABLE: bool = False
BoTSORT: type  # forward declaration for type checker

try:
    from boxmot import BotSort as BoTSORT  # type: ignore[import-untyped]
    _BOXMOT_AVAILABLE = True
except ImportError:
    try:
        from boxmot import BoTSORT  # type: ignore[import-untyped]
        _BOXMOT_AVAILABLE = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 简易回退追踪器 (IOU-based)
# ---------------------------------------------------------------------------

class _SimpleTracker:
    """
    最小化 IOU 追踪器，当 boxmot 不可用时充当后备。

    逻辑：
    1. 根据 IOU 贪心匹配现有轨迹与新检测
    2. 未匹配的检测创建新轨迹
    3. 连续 ``max_age`` 帧未匹配的轨迹删除
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self._next_id: int = 1
        self._tracks: dict[int, np.ndarray] = {}  # track_id → bbox
        self._ages: dict[int, int] = {}            # track_id → 未匹配帧数
        self._iou_threshold = iou_threshold
        self._max_age = max_age

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    def update(self, dets: np.ndarray, _frame: np.ndarray) -> np.ndarray:
        """
        Run one tracking step.

        Args:
            dets: (N, 6) — [x1, y1, x2, y2, conf, cls]
            _frame: current video frame (unused by simple tracker)

        Returns:
            (M, 7) — [x1, y1, x2, y2, track_id, conf, cls]
        """
        if dets is None or len(dets) == 0:
            # Age all existing tracks
            expired = []
            for tid in list(self._ages):
                self._ages[tid] += 1
                if self._ages[tid] > self._max_age:
                    expired.append(tid)
            for tid in expired:
                del self._tracks[tid]
                del self._ages[tid]
            return np.empty((0, 7), dtype=np.float32)

        det_boxes = dets[:, :4]
        det_confs = dets[:, 4]
        det_cls = dets[:, 5]
        n_dets = len(dets)

        # Build IOU cost matrix
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)

        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        if n_tracks > 0:
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for ti, tid in enumerate(track_ids):
                for di in range(n_dets):
                    iou_matrix[ti, di] = self._iou(self._tracks[tid], det_boxes[di])

            # Greedy matching
            while True:
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[idx[0], idx[1]] < self._iou_threshold:
                    break
                ti, di = int(idx[0]), int(idx[1])
                matched_trk.add(ti)
                matched_det.add(di)
                tid = track_ids[ti]
                self._tracks[tid] = det_boxes[di].copy()
                self._ages[tid] = 0
                iou_matrix[ti, :] = 0.0
                iou_matrix[:, di] = 0.0

        # Create new tracks for unmatched detections
        for di in range(n_dets):
            if di not in matched_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = det_boxes[di].copy()
                self._ages[tid] = 0

        # Age unmatched tracks
        expired = []
        for ti, tid in enumerate(track_ids):
            if ti not in matched_trk:
                self._ages[tid] = self._ages.get(tid, 0) + 1
                if self._ages[tid] > self._max_age:
                    expired.append(tid)
        for tid in expired:
            del self._tracks[tid]
            del self._ages[tid]

        # Build output
        results: list[np.ndarray] = []
        for tid, bbox in self._tracks.items():
            if self._ages[tid] == 0:  # Only return actively matched tracks
                # Find the matching detection confidence / cls
                best_iou = 0.0
                best_di = 0
                for di in range(n_dets):
                    iou_val = self._iou(bbox, det_boxes[di])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_di = di
                row = np.array([
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    float(tid), det_confs[best_di], det_cls[best_di],
                ], dtype=np.float32)
                results.append(row)

        if not results:
            return np.empty((0, 7), dtype=np.float32)
        return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# 追踪引擎
# ---------------------------------------------------------------------------

class TrackingEngine:
    """
    多目标追踪引擎。

    封装 BoT-SORT (boxmot) 或简易回退追踪器，提供:
    - 帧级追踪更新 (Detection → TrackedPerson)
    - 身份缓存管理 (track_id → PersonIdentity)
    - 中心轨迹缓冲 (trails)
    """

    _MAX_TRAIL_LEN: int = 30

    def __init__(self) -> None:
        """初始化追踪引擎。"""

        # Trail buffer: track_id → list[(cx, cy)]
        self._trails: dict[int, list[tuple[float, float]]] = {}


        # Initialise tracker backend
        self._tracker = self._create_tracker()
        logger.info(
            "TrackingEngine initialised — backend={}",
            "BoT-SORT" if _BOXMOT_AVAILABLE else "SimpleTracker(fallback)",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[TrackedPerson]:
        """
        运行一帧追踪更新。

        Args:
            frame: 当前视频帧 (BGR, H×W×3)。
            detections: 当前帧的人体检测列表。

        Returns:
            追踪后的 TrackedPerson 列表（含身份缓存信息）。
        """
        # 1. Convert detections to (N, 6) array for tracker
        det_array = self._detections_to_array(detections)

        # 2. Run tracker
        try:
            results = self._tracker.update(det_array, frame)
        except Exception:
            logger.exception("Tracker update failed, returning empty")
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

        # 5. Clean up stale trails / area history for lost tracks
        self._cleanup_stale(active_ids)

        return tracked_persons

    def remove_track(self, track_id: int) -> None:
        """
        移除追踪 ID 的所有缓存信息。

        Args:
            track_id: 追踪 ID。
        """
        self._trails.pop(track_id, None)
        logger.debug("Track removed: {}", track_id)

    def get_active_track_ids(self) -> list[int]:
        """
        获取当前所有有轨迹记录的追踪 ID。

        Returns:
            活跃追踪 ID 列表。
        """
        return list(self._trails.keys())


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_tracker() -> BoTSORT | _SimpleTracker:
        """Create the tracker backend."""
        if _BOXMOT_AVAILABLE:
            try:
                import torch
                config = get_config()
                device = torch.device(config.detection.yolo_device)
                from pathlib import Path
                tracker = BoTSORT(
                    reid_weights=Path(""),
                    device=device,
                    half=False,
                    track_high_thresh=config.tracking.track_high_thresh,
                    track_low_thresh=config.tracking.track_low_thresh,
                    new_track_thresh=config.tracking.new_track_thresh,
                    track_buffer=config.tracking.track_buffer,
                    match_thresh=config.tracking.match_thresh,
                    cmc_method=config.tracking.cmc_method,
                    with_reid=False,
                )
                logger.info("BoT-SORT tracker created successfully")
                return tracker
            except Exception as e:
                logger.warning("Failed to create BoTSORT: {}. Using fallback.", e)
                return _SimpleTracker(max_age=get_config().tracking.track_buffer)
        else:
            logger.warning("boxmot not installed. Using fallback tracker.")
            return _SimpleTracker(max_age=get_config().tracking.track_buffer)

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
