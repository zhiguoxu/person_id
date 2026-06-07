"""
多帧特征聚合

将 QualityCache 中的 K 帧特征聚合为鲁棒的代表性特征:
- 人脸: 按姿态分桶, 桶内质量加权质心
- 人体: 按姿态分桶, 桶内质量加权平均 (同时段不换装)
- 体型: 鲁棒中位数
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.config import get_config
from src.gallery.data_models import BodyProportions, PoseBucket
from src.pipeline.frame_buffer import CachedFrame, QualityCache


class AggregatedFeatures(BaseModel):
    """多帧聚合后的特征"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 人脸: 按姿态分桶, 每桶一个聚合特征
    face_per_pose: dict[PoseBucket, tuple[np.ndarray, float]]  # {pose: (embedding, quality)}

    # 人体: 按姿态分桶聚合 (与人脸对称)
    body_per_pose: dict[PoseBucket, tuple[np.ndarray, float]]  # {pose: (embedding, quality)}

    # 体型: 鲁棒中位数
    proportions: BodyProportions | None = None


class MultiFrameAggregator:
    """多帧特征聚合器 — 从 QualityCache 消费 CachedFrame"""

    @staticmethod
    def aggregate_from_cache(cache: QualityCache) -> AggregatedFeatures:
        """从 QualityCache 聚合特征 — 直接消费 CachedFrame"""

        return AggregatedFeatures(
            face_per_pose=MultiFrameAggregator._aggregate_face(cache.face_pool),
            body_per_pose=MultiFrameAggregator._aggregate_body(cache.body_pool),
            proportions=MultiFrameAggregator.aggregate_proportions(cache.body_pool)
        )

    @staticmethod
    def _aggregate_face(face_pool: list[CachedFrame]
                        ) -> dict[PoseBucket, tuple[np.ndarray, float]]:
        """按姿态分桶, 桶内质量加权聚合"""
        min_quality = get_config().multiframe.agg_min_face_quality
        buckets: dict[PoseBucket, list[tuple[np.ndarray, float]]] = defaultdict(list)

        for cf in face_pool:
            if cf.face_quality >= min_quality:
                buckets[cf.entry.pose_bucket].append((cf.face_embedding, cf.face_quality))

        return MultiFrameAggregator._weighted_aggregate(buckets)

    @staticmethod
    def _aggregate_body(body_pool: list[CachedFrame]
                        ) -> dict[PoseBucket, tuple[np.ndarray, float]]:
        """按姿态分桶, 桶内质量加权聚合人体特征 (与 _aggregate_face 对称)

        注: 同一批 Tier2 帧不会换装, 桶内质心是安全的.
        Gallery 端因换装导致多峰分布, 所以 Gallery 端逐条保留不做质心.
        """
        min_quality = get_config().multiframe.agg_min_body_quality
        buckets: dict[PoseBucket, list[tuple[np.ndarray, float]]] = defaultdict(list)

        for cf in body_pool:
            if cf.body_quality >= min_quality:
                buckets[cf.entry.pose_bucket].append((cf.body_embedding, cf.body_quality))

        # 降级: 如果所有帧都低于阈值, 使用所有帧
        if not buckets:
            for cf in body_pool:
                buckets[cf.entry.pose_bucket].append(
                    (cf.body_embedding, max(cf.body_quality, 0.01)))

        return MultiFrameAggregator._weighted_aggregate(buckets)

    @staticmethod
    def _weighted_aggregate(
            buckets: dict[PoseBucket, list[tuple[np.ndarray, float]]],
    ) -> dict[PoseBucket, tuple[np.ndarray, float]]:
        """质量加权聚合: 每个姿态桶内做加权平均 + L2 归一化。"""
        result: dict[PoseBucket, tuple[np.ndarray, float]] = {}
        for pose, entries in buckets.items():
            embeddings = np.stack([e[0] for e in entries])
            qualities = np.array([e[1] for e in entries])
            weights = np.maximum(qualities, 1e-6)

            agg = np.average(embeddings, axis=0, weights=weights)
            agg = agg / (np.linalg.norm(agg) + 1e-8)

            avg_quality = float(np.average(qualities, weights=weights))
            result[pose] = (agg, avg_quality)

        return result

    @staticmethod
    def aggregate_proportions(body_pool: list[CachedFrame]
                              ) -> BodyProportions | None:
        """鲁棒中位数聚合体型比例 (抗离群值)"""
        valid: list[BodyProportions] = [BodyProportions.from_keypoints(cf.entry.keypoints) for cf in body_pool]
        valid = [p for p in valid if p is not None]
        if not valid:
            return None

        vectors: np.ndarray = np.stack([p.to_vector() for p in valid])  # (N, 4)
        median_vec: np.ndarray = np.median(vectors, axis=0)  # (4,) 中位数

        # 相对高度也取中位数
        heights: list[float] = [p.relative_height_px for p in valid if p.relative_height_px > 0]
        median_height: float = float(np.median(heights)) if heights else 0.0

        return BodyProportions(
            torso_leg_ratio=float(median_vec[0]),
            shoulder_hip_ratio=float(median_vec[1]),
            arm_torso_ratio=float(median_vec[2]),
            head_body_ratio=float(median_vec[3]),
            relative_height_px=median_height,
        )
