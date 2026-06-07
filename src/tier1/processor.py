"""
Tier 1 流水线 — 快速逐帧追踪

每帧执行:
1. 轻量 YOLO11n-pose 检测
2. BoT-SORT 追踪器更新
3. 注意力评分更新

输出 TrackedPerson 列表 (纯帧级数据), 身份由 Orchestrator 从 TrackState 注入。
"""
from __future__ import annotations

import time

import numpy as np
from loguru import logger

from src.tier1.attention import AttentionEngine
from src.tier1.detection import get_fast_detector
from src.gallery.data_models import TrackedPerson
from src.tier1.tracking import TrackingEngine


class Tier1Processor:
    """Tier 1 处理器：快速逐帧检测 + 追踪 + 注意力评分。

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
            当前帧的 TrackedPerson 列表（含注意力分，不含身份信息）。
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

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Tier1: {} detections, {} tracked, {:.1f}ms",
            len(detections),
            len(persons),
            elapsed_ms,
        )

        return persons
