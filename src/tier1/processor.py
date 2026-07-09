"""
Tier 1 流水线 — 快速逐帧追踪

每帧执行:
1. 轻量 YOLO11n-pose 检测
2. BoT-SORT 追踪器更新
3. 注意力评分更新
4. 人体裁剪 + 人脸检测 + 质量评估

输出 TrackedPerson 列表 (纯帧级数据), 身份由 Orchestrator 从 TrackState 注入。
"""
from __future__ import annotations

import time

import numpy as np
from loguru import logger

from src.gallery.data_models import PoseBucket
from src.tier1.attention import AttentionEngine
from src.tier1.detection import get_fast_detector
from src.tier1.face_detector_light import get_face_detector_light
from src.pipeline.data_models import TrackedPerson
from src.pipeline.quality_utils import compute_quality_hint, compute_blur_score
from src.tier1.tracking import TrackingEngine
# eDifFIQA 统一使用 tier2/features 的实现 (与入库把关、测试接口共用同一单例, 避免同模型重复加载)
from src.tier2.features.ediffiqa import get_ediffiqa


class Tier1Processor:
    """Tier 1 处理器：快速逐帧检测 + 追踪 + 注意力评分 + 人脸检测。

    在机器人主循环中以全帧率运行，使用轻量 YOLO11n 和
    BoT-SORT 维护多人追踪。

    身份管理由 Orchestrator 通过 TrackState 统一负责，
    Tier1 不持有身份缓存。
    """

    def __init__(self) -> None:
        self.detector = get_fast_detector()
        self.tracker = TrackingEngine()
        self.attention = AttentionEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> list[TrackedPerson]:
        """处理单帧图像，返回追踪人物列表。

        Args:
            frame: BGR 图像 (H, W, 3)。

        Returns:
            当前帧的 TrackedPerson 列表（含注意力分、裁剪、人脸检测结果）。
        """
        t0 = time.perf_counter()

        # 1. 检测
        detections = self.detector.detect(frame)

        # 2. 追踪
        persons = self.tracker.update(frame, detections)

        # 3. 注意力评分
        scores = self.attention.compute_scores(persons, frame.shape)
        for person in persons:
            person.attention_score = scores.get(person.track_id, 0.0)

        # 4. 裁剪 + 人脸检测 + 质量评估
        self._extract_face_info(frame, persons)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.trace(
            "Tier1: {} detections, {} tracked, {:.1f}ms",
            len(detections),
            len(persons),
            elapsed_ms,
        )

        return persons

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_face_info(frame: np.ndarray, persons: list[TrackedPerson]) -> None:
        """对每个 person 执行裁剪 + 人脸检测 + 质量评估。

        结果直接写入 TrackedPerson 的字段。
        """
        h_f, w_f = frame.shape[:2]
        for person in persons:
            det = person.detection
            if det is None or det.bbox is None:
                continue

            # crop
            x1 = max(0, int(det.bbox[0]))
            y1 = max(0, int(det.bbox[1]))
            x2 = min(w_f, int(det.bbox[2]))
            y2 = min(h_f, int(det.bbox[3]))
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            person.crop = crop
            person.crop_offset = (x1, y1)
            person.quality_hint = compute_quality_hint(det.bbox, det.keypoints, (h_f, w_f))

            # local keypoints
            local_kps = det.keypoints.copy() if det.keypoints is not None else None
            if local_kps is not None:
                local_kps[:, 0] -= x1
                local_kps[:, 1] -= y1
            person.local_keypoints = local_kps

            # 人脸检测 + 质量评估 (非 BACK 姿态)
            if det.pose_bucket != PoseBucket.BACK:
                t_det = time.perf_counter()
                result = get_face_detector_light().get_aligned_face(crop)
                person.face_detect_ms = (time.perf_counter() - t_det) * 1000
                if result is not None:
                    aligned_face, face_bbox, face_kps = result
                    person.aligned_face = aligned_face
                    person.face_bbox = face_bbox
                    t_assess = time.perf_counter()
                    ediffiqa_score = get_ediffiqa().predict(aligned_face)
                    blur = compute_blur_score(aligned_face)
                    person.face_assess_ms = (time.perf_counter() - t_assess) * 1000
                    person.face_quality = 0.8 * ediffiqa_score + 0.2 * blur

