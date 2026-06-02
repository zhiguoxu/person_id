"""
Gallery Updater — 底库入库与更新

负责将识别管线产出的特征安全地写入人物档案:
- 人脸特征入库 (按质量门槛)
- 衣橱记录入库 / 更新 (换装适应)
- 体型比例累积更新
- 跨模态写入 ("化学反应": 人脸确认身份 → 带动衣橱 + 体型更新)
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
    PersonProfile,
)


class GalleryUpdater:
    """底库更新器 — 管理人脸、衣橱、体型特征的入库流程。

    所有写入操作均受质量门槛约束, 确保底库特征质量只升不降。
    """

    def __init__(self, config: Config) -> None:
        self._gallery_cfg = config.gallery
        logger.info("GalleryUpdater initialized")

    # ------------------------------------------------------------------
    # 人脸入库
    # ------------------------------------------------------------------

    def enroll_face(self, profile: PersonProfile, face_entry: FeatureEntry) -> bool:
        """将人脸特征入库到对应姿态桶。

        入库条件:
            - 质量分 ≥ ``quality_enroll_threshold``

        更新条件 (桶已满时替换):
            - 质量分 ≥ ``quality_update_threshold``

        Args:
            profile: 目标人物档案。
            face_entry: 待入库的人脸特征条目。

        Returns:
            是否成功入库。
        """
        quality = face_entry.quality_score

        if quality < self._gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Face quality {:.3f} below enroll threshold {:.3f} for {}",
                quality,
                self._gallery_cfg.quality_enroll_threshold,
                profile.person_id,
            )
            return False

        # 如果桶已满, 需要更高的质量才能替换
        bucket_entries = profile.face_features.get(face_entry.pose_bucket, [])
        if len(bucket_entries) >= self._gallery_cfg.max_faces_per_bucket:
            if quality < self._gallery_cfg.quality_update_threshold:
                logger.debug(
                    "Face quality {:.3f} below update threshold {:.3f} for {}",
                    quality,
                    self._gallery_cfg.quality_update_threshold,
                    profile.person_id,
                )
                return False

        success = profile.add_face_feature(face_entry)
        if success:
            profile.touch()
            logger.debug(
                "Enrolled face for {} in bucket {} (quality={:.3f})",
                profile.person_id,
                face_entry.pose_bucket.value,
                quality,
            )
        return success

    # ------------------------------------------------------------------
    # 衣橱入库
    # ------------------------------------------------------------------

    def enroll_outfit(
        self,
        profile: PersonProfile,
        body_embedding: np.ndarray,
        quality: float,
    ) -> None:
        """将全身 ReID 特征入库到衣橱。

        低于入库质量门槛的特征将被拒绝。

        Args:
            profile: 目标人物档案。
            body_embedding: L2 归一化的全身特征向量。
            quality: 特征提取时的质量分。
        """
        if quality < self._gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Body quality {:.3f} below threshold for {}",
                quality,
                profile.person_id,
            )
            return

        profile.add_outfit(body_embedding, quality)
        profile.touch()
        logger.debug(
            "Enrolled/updated outfit for {} (quality={:.3f}, wardrobe_size={})",
            profile.person_id,
            quality,
            len(profile.wardrobe),
        )

    # ------------------------------------------------------------------
    # 体型比例更新
    # ------------------------------------------------------------------

    def update_proportions(
        self,
        profile: PersonProfile,
        proportions: BodyProportions,
    ) -> None:
        """累积平均更新体型比例。

        Args:
            profile: 目标人物档案。
            proportions: 从关键点提取的体型比例。
        """
        profile.update_proportions(proportions)
        logger.debug(
            "Updated proportions for {} (samples={})",
            profile.person_id,
            profile.body_proportions_samples,
        )

    # ------------------------------------------------------------------
    # 跨模态写入 ("化学反应")
    # ------------------------------------------------------------------

    def cross_write(
        self,
        profile: PersonProfile,
        face_entry: Optional[FeatureEntry] = None,
        body_embedding: Optional[np.ndarray] = None,
        proportions: Optional[BodyProportions] = None,
        quality: float = 0.0,
    ) -> None:
        """跨模态写入 — "化学反应"。

        当人脸确认了身份后, 顺带更新衣橱和体型:
            人脸确认身份 → 带动衣橱特征入库 + 体型比例更新

        这使得后续即使看不到正脸, 也能通过衣橱或体型匹配到该人。

        Args:
            profile: 身份已确认的人物档案。
            face_entry: 可选的人脸特征 (如果可用)。
            body_embedding: 可选的全身特征向量。
            proportions: 可选的体型比例。
            quality: 当前帧的特征质量分。
        """
        updated_modalities: list[str] = []

        # 1. 人脸入库 (如果提供)
        if face_entry is not None:
            if self.enroll_face(profile, face_entry):
                updated_modalities.append("face")

        # 2. 衣橱入库 (如果提供)
        if body_embedding is not None:
            self.enroll_outfit(profile, body_embedding, quality)
            updated_modalities.append("outfit")

        # 3. 体型比例更新 (如果提供)
        if proportions is not None:
            self.update_proportions(profile, proportions)
            updated_modalities.append("proportions")

        if updated_modalities:
            profile.touch()
            logger.info(
                "Cross-write for {}: updated [{}]",
                profile.person_id,
                ", ".join(updated_modalities),
            )
