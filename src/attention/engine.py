"""
注意力引擎 — 确定机器人应关注哪个人

基于多信号加权打分:
- 面积占比 (越大 = 越近)
- 距帧中心距离 (越近 = 越正对)
- 人脸可见性 (正面 > 侧面 > 背面)
- 靠近趋势奖励
- 动量奖励 (维持当前目标的稳定性)

使用滞后机制防止目标频繁切换。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.config import Config
from src.gallery.data_models import PoseBucket, TrackedPerson


class AttentionEngine:
    """
    注意力评分与目标选择引擎。

    每帧对所有追踪人物打分，选出最值得关注的目标。
    支持外部强制设定目标（如用户点击）。
    """

    # 评分权重
    _W_AREA: float = 0.30
    _W_CENTER: float = 0.30
    _W_FACE: float = 0.40

    # 奖励
    _APPROACHING_BONUS: float = 0.15
    _MOMENTUM_BONUS: float = 0.20

    # 目标切换滞后阈值
    _HYSTERESIS_MARGIN: float = 0.15

    def __init__(self, config: Config) -> None:
        """
        初始化注意力引擎。

        Args:
            config: 系统配置。
        """
        self._config = config
        self._current_target: Optional[int] = None
        self._prev_areas: dict[int, list[float]] = {}
        logger.info("AttentionEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_scores(
        self,
        persons: list[TrackedPerson],
        frame_shape: tuple[int, ...],
    ) -> dict[int, float]:
        """
        为每个追踪人物计算注意力分数。

        Args:
            persons: 当前帧追踪人物列表。
            frame_shape: 帧尺寸 (H, W, C) 或 (H, W)。

        Returns:
            字典 {track_id: total_score}。
        """
        if not persons:
            return {}

        frame_h = frame_shape[0]
        frame_w = frame_shape[1]
        frame_area = float(frame_h * frame_w)
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0
        max_distance = np.sqrt(frame_cx ** 2 + frame_cy ** 2)

        scores: dict[int, float] = {}

        for person in persons:
            det = person.detection
            tid = person.track_id

            # --- Area score ---
            area_ratio = min(det.area / frame_area, 1.0) if frame_area > 0 else 0.0
            area_score = area_ratio

            # --- Center score ---
            center = det.center
            dist_to_center = float(np.sqrt(
                (center[0] - frame_cx) ** 2 + (center[1] - frame_cy) ** 2
            ))
            center_score = 1.0 - (dist_to_center / max_distance) if max_distance > 0 else 0.0
            center_score = max(0.0, min(1.0, center_score))

            # --- Face visibility score ---
            face_score = self._face_visibility_score(det.pose_bucket)

            # --- Weighted base score ---
            total = (
                self._W_AREA * area_score
                + self._W_CENTER * center_score
                + self._W_FACE * face_score
            )

            # --- Approaching bonus ---
            area_hist = self._prev_areas.get(tid, [])
            area_hist.append(det.area)
            if len(area_hist) > 10:
                area_hist[:] = area_hist[-10:]
            self._prev_areas[tid] = area_hist

            if len(area_hist) >= 3:
                recent = area_hist[-3:]
                if recent[-1] > recent[0] * 1.05:  # area growing by >5%
                    total += self._APPROACHING_BONUS

            # --- Momentum bonus ---
            if self._current_target is not None and tid == self._current_target:
                total += self._MOMENTUM_BONUS

            scores[tid] = total

        # Cleanup area history for disappeared tracks
        active_ids = {p.track_id for p in persons}
        stale = [tid for tid in self._prev_areas if tid not in active_ids]
        for tid in stale:
            del self._prev_areas[tid]

        return scores

    def select_target(self, scores: dict[int, float]) -> Optional[int]:
        """
        选择注意力目标。

        应用滞后机制：只有当新目标的分数显著高于当前目标时才切换。

        Args:
            scores: 注意力评分字典 {track_id: score}。

        Returns:
            目标 track_id，若场景中无人则返回 None。
        """
        if not scores:
            self._current_target = None
            return None

        best_tid = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_tid]

        # 滞后判断
        if self._current_target is not None and self._current_target in scores:
            current_score = scores[self._current_target]
            if best_score - current_score < self._HYSTERESIS_MARGIN:
                # 差距不够大，保持当前目标
                return self._current_target

        # 切换目标
        if self._current_target != best_tid:
            logger.debug(
                "Attention target switched: {} → {} (score={:.3f})",
                self._current_target, best_tid, best_score,
            )
        self._current_target = best_tid
        return best_tid

    def set_target(self, track_id: int) -> None:
        """
        强制设定当前注意力目标（如用户点击）。

        Args:
            track_id: 要关注的追踪 ID。
        """
        logger.info("Attention target force-set to track_id={}", track_id)
        self._current_target = track_id

    @property
    def current_target(self) -> Optional[int]:
        """当前注意力目标 track_id。"""
        return self._current_target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _face_visibility_score(pose_bucket: PoseBucket) -> float:
        """Map pose bucket to face visibility score."""
        if pose_bucket == PoseBucket.FRONTAL:
            return 1.0
        elif pose_bucket in (PoseBucket.LEFT, PoseBucket.RIGHT):
            return 0.5
        elif pose_bucket == PoseBucket.BACK:
            return 0.0
        else:
            return 0.25  # UNKNOWN — some benefit of the doubt
