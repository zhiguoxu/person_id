"""
Gallery Matcher — 姿态感知最大池化匹配

提供三种模态的底库匹配能力:
- 人脸匹配: 按姿态桶优先级, 对每个桶内特征做 max cosine similarity
- 全身匹配: 对衣橱库做 max cosine similarity, 带近因衰减
- 体型比例匹配: 利用高斯核相似度
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from loguru import logger

from src.config import Config
from src.gallery.data_models import (
    BodyProportions,
    FeatureEntry,
    MatchCandidate,
    PersonProfile,
    PoseBucket,
)

# 姿态桶邻接关系: 每个桶的优先检索顺序 (自身 → 相邻 → 远处)
_ADJACENT_BUCKETS: dict[PoseBucket, list[PoseBucket]] = {
    PoseBucket.FRONTAL: [PoseBucket.LEFT, PoseBucket.RIGHT],
    PoseBucket.LEFT: [PoseBucket.FRONTAL, PoseBucket.BACK],
    PoseBucket.RIGHT: [PoseBucket.FRONTAL, PoseBucket.BACK],
    PoseBucket.BACK: [PoseBucket.LEFT, PoseBucket.RIGHT],
    PoseBucket.UNKNOWN: [
        PoseBucket.FRONTAL,
        PoseBucket.LEFT,
        PoseBucket.RIGHT,
        PoseBucket.BACK,
    ],
}


class GalleryMatcher:
    """底库匹配器 — 支持人脸 / 全身 / 体型三种模态。

    所有匹配方法均返回按相关分数降序排列的 ``MatchCandidate`` 列表。
    """

    def __init__(self, config: Config) -> None:
        self._gallery_cfg = config.gallery
        self._matching_cfg = config.matching
        logger.info("GalleryMatcher initialized")

    # ------------------------------------------------------------------
    # 人脸匹配
    # ------------------------------------------------------------------

    def match_face(
        self,
        face_embedding: np.ndarray,
        pose: PoseBucket,
        gallery: dict[str, PersonProfile],
    ) -> list[MatchCandidate]:
        """按姿态桶优先级进行人脸匹配。

        流程:
            1. 构建优先桶列表 — 同姿态桶优先, 然后相邻桶。
            2. 对每个人, 在优先桶中找到最大余弦相似度。
            3. 对质量分应用时间衰减。
            4. 返回降序排列的候选列表。

        Args:
            face_embedding: 查询人脸的 L2 归一化特征向量。
            pose: 查询人脸的姿态桶。
            gallery: 底库, key=person_id, value=PersonProfile。

        Returns:
            按 ``face_score`` 降序排列的 ``MatchCandidate`` 列表。
        """
        if face_embedding is None or len(gallery) == 0:
            return []

        # 确保查询向量 L2 归一化
        face_embedding = self._l2_normalize(face_embedding)

        # 桶优先级: 自身 → 相邻
        priority_buckets = [pose] + _ADJACENT_BUCKETS.get(pose, [])

        now = time.time()
        half_life = self._gallery_cfg.face_half_life_days
        candidates: list[MatchCandidate] = []

        for person_id, profile in gallery.items():
            best_score = -1.0

            for bucket in priority_buckets:
                entries: list[FeatureEntry] = profile.face_features.get(bucket, [])
                for entry in entries:
                    cos_sim = float(np.dot(face_embedding, entry.embedding))
                    decay = entry.time_decay_weight(now, half_life)
                    weighted = cos_sim * decay
                    if weighted > best_score:
                        best_score = weighted

            if best_score > 0.0:
                candidates.append(
                    MatchCandidate(
                        person_id=person_id,
                        display_name=profile.display_name,
                        face_score=best_score,
                    )
                )

        # 降序
        candidates.sort(key=lambda c: c.face_score or 0.0, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 全身匹配 (衣橱)
    # ------------------------------------------------------------------

    def match_body(
        self,
        body_embedding: np.ndarray,
        gallery: dict[str, PersonProfile],
    ) -> list[MatchCandidate]:
        """对衣橱库进行全身 ReID 匹配。

        对每个人的所有衣橱记录计算余弦相似度, 取 max, 并乘以近因权重。

        Args:
            body_embedding: 查询全身的 L2 归一化特征向量。
            gallery: 底库。

        Returns:
            按 ``body_score`` 降序排列的 ``MatchCandidate`` 列表。
        """
        if body_embedding is None or len(gallery) == 0:
            return []

        body_embedding = self._l2_normalize(body_embedding)
        now = time.time()
        candidates: list[MatchCandidate] = []

        for person_id, profile in gallery.items():
            if not profile.wardrobe:
                continue

            best_score = -1.0
            for outfit in profile.wardrobe:
                cos_sim = float(np.dot(body_embedding, outfit.body_embedding))
                recency = outfit.recency_weight(now)
                weighted = cos_sim * recency
                if weighted > best_score:
                    best_score = weighted

            if best_score > 0.0:
                candidates.append(
                    MatchCandidate(
                        person_id=person_id,
                        display_name=profile.display_name,
                        body_score=best_score,
                    )
                )

        candidates.sort(key=lambda c: c.body_score or 0.0, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 体型比例匹配
    # ------------------------------------------------------------------

    def match_proportions(
        self,
        proportions: BodyProportions,
        gallery: dict[str, PersonProfile],
    ) -> list[MatchCandidate]:
        """使用体型比例高斯核相似度进行匹配。

        Args:
            proportions: 查询目标的体型比例。
            gallery: 底库。

        Returns:
            按 ``proportion_score`` 降序排列的 ``MatchCandidate`` 列表。
        """
        if proportions is None or len(gallery) == 0:
            return []

        candidates: list[MatchCandidate] = []

        for person_id, profile in gallery.items():
            if profile.body_proportions is None:
                continue

            score = BodyProportions.similarity(proportions, profile.body_proportions)
            candidates.append(
                MatchCandidate(
                    person_id=person_id,
                    display_name=profile.display_name,
                    proportion_score=score,
                )
            )

        candidates.sort(key=lambda c: c.proportion_score or 0.0, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2 归一化, 若范数为 0 则原样返回。"""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm
