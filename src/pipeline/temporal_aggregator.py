"""
时序特征聚合器 — 质量加权的滑动窗口特征融合

对每个 track_id 维护固定大小的滑动窗口，以质量分为权重
对多帧嵌入向量加权平均，输出 L2 归一化的聚合特征。
用于 Tier 2 流水线中平滑单帧噪声。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class _WindowEntry:
    """滑动窗口中的单帧条目。"""
    embedding: np.ndarray
    quality: float


class TemporalAggregator:
    """
    质量加权的时序特征聚合器。

    为每个 track_id 维护一个固定大小的滑动窗口，
    接收新的嵌入向量 + 质量分后，返回质量加权平均的
    L2 归一化聚合特征。

    Args:
        window_size: 每个 track 的滑动窗口大小（帧数）。
    """

    def __init__(self, window_size: int = 5) -> None:
        self._window_size = max(1, window_size)
        self._windows: dict[int, deque[_WindowEntry]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_and_get(
        self, track_id: int, embedding: np.ndarray, quality: float
    ) -> np.ndarray:
        """
        向指定 track 的窗口追加一帧嵌入，并返回聚合特征。

        Args:
            track_id: 追踪器分配的轨迹 ID。
            embedding: 当前帧的特征向量（应已 L2 归一化）。
            quality: 当前帧的质量评分 [0, 1]。

        Returns:
            L2 归一化的质量加权平均嵌入向量。
        """
        if track_id not in self._windows:
            self._windows[track_id] = deque(maxlen=self._window_size)

        window = self._windows[track_id]
        window.append(_WindowEntry(embedding=embedding.copy(), quality=quality))

        return self._aggregate(window)

    def remove(self, track_id: int) -> None:
        """
        移除指定 track 的窗口数据。

        Args:
            track_id: 要移除的轨迹 ID。
        """
        if track_id in self._windows:
            del self._windows[track_id]
            logger.debug("Removed temporal window for track_id={}", track_id)

    def clear(self) -> None:
        """清空所有 track 的窗口数据。"""
        count = len(self._windows)
        self._windows.clear()
        logger.debug("Cleared all temporal windows ({})", count)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(window: deque[_WindowEntry]) -> np.ndarray:
        """
        对窗口内嵌入做质量加权平均并 L2 归一化。

        权重公式: w_i = max(quality_i, 1e-6)
        """
        embeddings = np.stack([e.embedding for e in window])
        weights = np.array(
            [max(e.quality, 1e-6) for e in window], dtype=np.float64
        )

        # 加权平均
        weighted_sum = (embeddings * weights[:, np.newaxis]).sum(axis=0)
        aggregated = weighted_sum / weights.sum()

        # L2 归一化
        norm = np.linalg.norm(aggregated)
        if norm > 1e-8:
            aggregated = aggregated / norm

        return aggregated.astype(np.float32)
