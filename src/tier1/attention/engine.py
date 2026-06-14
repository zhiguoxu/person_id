"""
注意力引擎 — 确定机器人应关注哪个人

基于多信号加权打分:
- 面积占比 (越大 = 越近)
- 距帧中心距离 (越近 = 越正对)
- 人脸可见性 (正面 > 侧面 > 背面)
- 靠近趋势奖励

目标切换的滞后防抖由 Orchestrator 负责。
"""
from __future__ import annotations


import numpy as np
from loguru import logger

from src.gallery.data_models import PoseBucket
from src.pipeline.data_models import TrackedPerson, Detection

# ==================================================================
# 共享评分核心 (无状态, 供 AttentionEngine 和 select_best_detection 复用)
# ==================================================================

# 评分权重
W_AREA: float = 0.30
W_CENTER: float = 0.30
W_FACE: float = 0.40


def _face_visibility_score(pose_bucket: PoseBucket) -> float:
    """姿态方向 → 人脸可见性分数。"""
    if pose_bucket == PoseBucket.FRONTAL:
        return 1.0
    elif pose_bucket in (PoseBucket.LEFT, PoseBucket.RIGHT):
        return 0.5
    elif pose_bucket == PoseBucket.BACK:
        return 0.0
    return 0.25  # UNKNOWN


def score_detection(det, frame_area: float, frame_cx: float,
                    frame_cy: float, max_distance: float) -> float:
    """对单个 Detection 计算注意力基础分 (不含靠近趋势奖励)。

    Args:
        det: Detection 对象 (需有 area, center, pose_bucket 属性)。
        frame_area: 帧总面积 (H * W)。
        frame_cx: 帧中心 X 坐标。
        frame_cy: 帧中心 Y 坐标。
        max_distance: 帧对角线半长 (归一化用)。

    Returns:
        加权注意力分数 [0, 1]。
    """
    # 面积占比
    area_score = min(det.area / frame_area, 1.0) if frame_area > 0 else 0.0

    # 距帧中心距离
    center = det.center
    dist = float(np.sqrt(
        (center[0] - frame_cx) ** 2 + (center[1] - frame_cy) ** 2
    ))
    center_score = max(0.0, 1.0 - dist / max_distance) if max_distance > 0 else 0.0

    # 人脸可见性
    face_score = _face_visibility_score(det.pose_bucket)

    return W_AREA * area_score + W_CENTER * center_score + W_FACE * face_score


def _frame_geometry(frame_shape: tuple[int, ...]) -> tuple[float, float, float, float]:
    """从帧尺寸计算评分所需的几何参数。

    Returns:
        (frame_area, frame_cx, frame_cy, max_distance)
    """
    frame_h, frame_w = frame_shape[0], frame_shape[1]
    frame_area = float(frame_h * frame_w)
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    max_distance = float(np.sqrt(frame_cx ** 2 + frame_cy ** 2))
    return frame_area, frame_cx, frame_cy, max_distance


# ==================================================================
# AttentionEngine (有状态, 用于 pipeline 逐帧追踪)
# ==================================================================

class AttentionEngine:
    """
    注意力评分引擎。

    每帧对所有追踪人物打分，供 Orchestrator 选出最值得关注的目标。
    在共享评分基础上, 额外计算靠近趋势奖励 (需跨帧面积历史)。
    """

    # 奖励
    _APPROACHING_BONUS: float = 0.15

    def __init__(self) -> None:
        """初始化注意力引擎。"""
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

        frame_area, frame_cx, frame_cy, max_distance = _frame_geometry(frame_shape)
        scores: dict[int, float] = {}

        for person in persons:
            det = person.detection
            tid = person.track_id

            # 基础分 (面积 + 中心 + 人脸)
            total = score_detection(det, frame_area, frame_cx, frame_cy, max_distance)

            # 靠近趋势奖励 (需要跨帧面积历史)
            area_hist = self._prev_areas.get(tid, [])
            area_hist.append(det.area)
            if len(area_hist) > 10:
                area_hist[:] = area_hist[-10:]
            self._prev_areas[tid] = area_hist

            if len(area_hist) >= 3:
                recent = area_hist[-3:]
                if recent[-1] > recent[0] * 1.05:  # area growing by >5%
                    total += self._APPROACHING_BONUS

            scores[tid] = total

        # Cleanup area history for disappeared tracks
        active_ids = {p.track_id for p in persons}
        stale = [tid for tid in self._prev_areas if tid not in active_ids]
        for tid in stale:
            del self._prev_areas[tid]

        return scores


# ==================================================================
# 无状态选人 (供 API 端点单张图片使用)
# ==================================================================

def select_best_detection(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
) -> int:
    """从多个检测中选出注意力最高的目标, 返回索引。

    使用与 AttentionEngine 相同的基础评分逻辑 (面积 + 中心距 + 人脸可见性),
    但无状态, 不依赖 TrackedPerson 和追踪历史。

    Args:
        detections: Detection 列表 (至少 1 个元素)。
        frame_shape: 帧尺寸 (H, W, C) 或 (H, W)。

    Returns:
        最佳检测的索引。
    """
    if len(detections) == 1:
        return 0

    frame_area, frame_cx, frame_cy, max_distance = _frame_geometry(frame_shape)

    best_idx = 0
    best_score = -1.0

    for i, det in enumerate(detections):
        total = score_detection(det, frame_area, frame_cx, frame_cy, max_distance)
        if total > best_score:
            best_score = total
            best_idx = i

    return best_idx
