"""
时空约束记忆 — 丢失轨迹的空间恢复机制

当追踪丢失时，记住 track 最后出现的位置和身份。
新检测出现在附近区域且在超时时间内时，自动关联身份。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from src.config import TrackingConfig


# ---------------------------------------------------------------------------
# 空间记忆条目
# ---------------------------------------------------------------------------

@dataclass
class _SpatialEntry:
    """单条空间记忆"""
    track_id: int
    person_id: str
    center: np.ndarray  # (2,) — 最后出现的中心坐标
    timestamp: float     # 记忆创建时间


# ---------------------------------------------------------------------------
# 空间记忆管理器
# ---------------------------------------------------------------------------

class SpatialMemory:
    """
    时空约束记忆器。

    工作流程:
    1. 追踪丢失时调用 :meth:`remember` 存储最后位置和身份。
    2. 新检测出现时调用 :meth:`check` 查找附近的记忆条目。
    3. 定期调用 :meth:`cleanup` 清除过期记忆。
    """

    def __init__(self, config: TrackingConfig) -> None:
        """
        初始化空间记忆。

        Args:
            config: 追踪配置，使用 spatial_timeout_sec 和 spatial_distance_px。
        """
        self._timeout_sec: float = config.spatial_timeout_sec
        self._max_distance_px: float = config.spatial_distance_px
        self._memories: list[_SpatialEntry] = []
        logger.debug(
            "SpatialMemory initialised: timeout={:.1f}s, max_dist={:.0f}px",
            self._timeout_sec, self._max_distance_px,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(
        self,
        track_id: int,
        center: np.ndarray,
        person_id: str,
        now: float,
    ) -> None:
        """
        记住丢失轨迹的最后位置。

        如果同一 track_id 已存在，则更新位置和时间。

        Args:
            track_id: 丢失的追踪 ID。
            center: 最后出现的中心坐标 (2,)。
            person_id: 关联的底库人物 ID。
            now: 当前时间戳 (Unix seconds)。
        """
        # 更新已有记忆
        for entry in self._memories:
            if entry.track_id == track_id:
                entry.center = np.asarray(center, dtype=np.float32).copy()
                entry.person_id = person_id
                entry.timestamp = now
                logger.debug(
                    "SpatialMemory updated: track_id={}, person_id={}",
                    track_id, person_id,
                )
                return

        # 新增记忆
        self._memories.append(_SpatialEntry(
            track_id=track_id,
            person_id=person_id,
            center=np.asarray(center, dtype=np.float32).copy(),
            timestamp=now,
        ))
        logger.debug(
            "SpatialMemory added: track_id={}, person_id={}, pos=({:.0f}, {:.0f})",
            track_id, person_id, float(center[0]), float(center[1]),
        )

    def check(
        self,
        center: np.ndarray,
        now: float,
    ) -> Optional[dict]:
        """
        检查给定位置附近是否有记忆中的轨迹。

        在 ``spatial_distance_px`` 范围内且未超时的最近记忆将被返回。

        Args:
            center: 检测的中心坐标 (2,)。
            now: 当前时间戳 (Unix seconds)。

        Returns:
            匹配结果字典 ``{track_id, person_id, distance}``，
            若无匹配则返回 None。
        """
        center = np.asarray(center, dtype=np.float32)
        best: Optional[_SpatialEntry] = None
        best_dist: float = self._max_distance_px

        for entry in self._memories:
            # 检查超时
            if (now - entry.timestamp) > self._timeout_sec:
                continue

            dist = float(np.linalg.norm(center - entry.center))
            if dist < best_dist:
                best_dist = dist
                best = entry

        if best is None:
            return None

        logger.debug(
            "SpatialMemory match: track_id={}, person_id={}, dist={:.1f}px",
            best.track_id, best.person_id, best_dist,
        )
        return {
            "track_id": best.track_id,
            "person_id": best.person_id,
            "distance": best_dist,
        }

    def cleanup(self, now: float) -> None:
        """
        移除过期的空间记忆。

        Args:
            now: 当前时间戳 (Unix seconds)。
        """
        before = len(self._memories)
        self._memories = [
            entry for entry in self._memories
            if (now - entry.timestamp) <= self._timeout_sec
        ]
        removed = before - len(self._memories)
        if removed > 0:
            logger.debug(
                "SpatialMemory cleanup: removed {} expired entries, {} remaining",
                removed, len(self._memories),
            )

    @property
    def size(self) -> int:
        """当前记忆条目数量。"""
        return len(self._memories)
