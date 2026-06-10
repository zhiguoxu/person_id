"""
机器人视觉人物识别系统 - 底库核心数据模型

定义底库 (Gallery) 相关数据结构，包括:
- PoseBucket: 姿态分桶枚举
- FeatureEntry: 单条特征记录
- FeatureOperation: 特征变动操作
- OutfitRecord: 衣橱记录
- OutfitEnrollResult: 衣橱入库结果
- GalleryUpdateResult: 增量更新结果
- BodyProportions: 体型比例特征
- PersonProfile: 人物档案 (底库核心)
"""
from __future__ import annotations

import time
import uuid

from loguru import logger
from enum import Enum
import numpy as np

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

from src.config import get_config


# ==============================================================================
# 枚举类型
# ==============================================================================

class PoseBucket(str, Enum):
    """基于 YOLO-Pose 关键点的人体朝向分桶"""
    FRONTAL = "frontal"  # 正面: 鼻子+双眼可见
    LEFT = "left"  # 左侧
    RIGHT = "right"  # 右侧
    BACK = "back"  # 背面
    UNKNOWN = "unknown"  # 关键点不足


# ==============================================================================
# 特征数据
# ==============================================================================

class FeatureEntry(BaseModel):
    """
    单条特征记录 (人脸或全身)
    
    存储 L2 归一化的特征向量，附带质量分和时间戳，
    用于底库匹配时的质量加权和时间衰减。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: np.ndarray  # L2 归一化特征向量
    pose_bucket: PoseBucket  # 特征提取时的姿态
    quality_score: float  # 综合质量分 [0, 1]
    timestamp: float  # 提取时间 (Unix timestamp)
    source_image: bytes | None = None  # JPEG 缩略图, 供 VLM 使用
    face_bbox: list[float] | None = None  # 人脸框 [x1,y1,x2,y2] 相对于 source_image

    def time_decay_weight(self, now: float, half_life_days: float) -> float:
        """计算时间衰减权重 (指数衰减)"""
        age_days = (now - self.timestamp) / 86400.0
        if age_days < 0:
            return 1.0
        return 0.5 ** (age_days / half_life_days)


class FeatureOperation(BaseModel):
    """单次特征的变动操作（仅入库成功时创建）。"""
    entry: FeatureEntry  # 新入库的特征
    evicted: FeatureEntry | None = None  # 不为 None 表示替换, 为 None 表示新增
    kind: Literal["face", "body"]


class OutfitRecord(BaseModel):
    """
    衣橱记录
    
    记录一套衣服的全身 ReID 特征，支持近因权重计算。
    同一人的不同衣服分开存储。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    body_embedding: np.ndarray  # 2048 维全身特征
    quality_score: float  # 提取时的质量分
    first_seen: float  # 首次穿着时间
    last_seen: float  # 最后穿着时间
    seen_count: int = 1  # 穿着次数

    def recency_weight(self, now: float) -> float:
        """近因权重: 最近穿过的衣服权重更高"""
        days_since = (now - self.last_seen) / 86400.0
        if days_since < 1:
            return 1.0
        elif days_since < 7:
            return 0.85
        elif days_since < 30:
            return 0.6
        elif days_since < 90:
            return 0.3
        else:
            return 0.1


class OutfitEnrollResult(BaseModel):
    """衣橱入库操作的结果（仅成功时返回）。

    三种情况:
    - updated 不为 None: EMA 更新了已有 outfit → DB UPDATE
    - evicted 不为 None: 替换了最旧 outfit → DB DELETE + INSERT
    - 两者都为 None: 新增 outfit (衣橱未满) → DB INSERT
    """
    outfit: OutfitRecord  # 新增或更新后的 outfit
    updated: OutfitRecord | None = None  # EMA 更新前的旧版本 (用于定位 DB 行)
    evicted: OutfitRecord | None = None  # 被替换淘汰的旧 outfit


class GalleryUpdateResult(BaseModel):
    """Gallery 的增量更新结果。"""
    feature_ops: list[FeatureOperation] = Field(default_factory=list)
    wardrobe_op: OutfitEnrollResult | None = None


class BodyProportions(BaseModel):
    """
    体型比例特征 (基于 COCO 17 关键点)
    
    零额外模型开销, 利用 YOLO-Pose 已输出的关键点计算骨骼几何比例。
    衣服无关的辅助身份信号。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    torso_leg_ratio: float  # 躯干/腿长比例
    shoulder_hip_ratio: float  # 肩宽/髋宽比例
    arm_torso_ratio: float  # 手臂/躯干比例
    head_body_ratio: float  # 头/身体比例
    relative_height_px: float  # 帧内相对高度 (像素)

    def to_vector(self) -> np.ndarray:
        """转换为 numpy 向量 (用于相似度计算)"""
        return np.array([
            self.torso_leg_ratio,
            self.shoulder_hip_ratio,
            self.arm_torso_ratio,
            self.head_body_ratio,
        ], dtype=np.float32)

    @staticmethod
    def from_keypoints(keypoints: np.ndarray) -> BodyProportions | None:
        """
        从 COCO 17 关键点提取体型比例
        
        keypoints shape: (17, 3) — x, y, confidence
        COCO 关键点顺序:
            0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
            5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
            9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
            13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        """
        CONF_THRESH = 0.3

        def _dist(idx_a: int, idx_b: int) -> float | None:
            if keypoints[idx_a, 2] < CONF_THRESH or keypoints[idx_b, 2] < CONF_THRESH:
                return None
            return float(np.linalg.norm(keypoints[idx_a, :2] - keypoints[idx_b, :2]))

        def _midpoint(idx_a: int, idx_b: int) -> np.ndarray | None:
            if keypoints[idx_a, 2] < CONF_THRESH or keypoints[idx_b, 2] < CONF_THRESH:
                return None
            return (keypoints[idx_a, :2] + keypoints[idx_b, :2]) / 2.0

        # 肩中点 → 髋中点 = 躯干长度
        shoulder_mid = _midpoint(5, 6)
        hip_mid = _midpoint(11, 12)
        if shoulder_mid is None or hip_mid is None:
            return None

        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        if torso_len < 1e-6:
            return None

        # 腿长: 髋 → 膝 → 踝
        l_upper_leg = _dist(11, 13)
        l_lower_leg = _dist(13, 15)
        r_upper_leg = _dist(12, 14)
        r_lower_leg = _dist(14, 16)

        leg_lengths: list[float] = []
        if l_upper_leg and l_lower_leg:
            leg_lengths.append(l_upper_leg + l_lower_leg)
        if r_upper_leg and r_lower_leg:
            leg_lengths.append(r_upper_leg + r_lower_leg)
        if not leg_lengths:
            return None
        leg_len = sum(leg_lengths) / len(leg_lengths)

        # 肩宽和髋宽
        shoulder_w = _dist(5, 6)
        hip_w = _dist(11, 12)
        if shoulder_w is None or hip_w is None or hip_w < 1e-6:
            return None

        # 手臂长: 肩 → 肘 → 腕
        l_upper_arm = _dist(5, 7)
        l_lower_arm = _dist(7, 9)
        r_upper_arm = _dist(6, 8)
        r_lower_arm = _dist(8, 10)

        arm_lengths: list[float] = []
        if l_upper_arm and l_lower_arm:
            arm_lengths.append(l_upper_arm + l_lower_arm)
        if r_upper_arm and r_lower_arm:
            arm_lengths.append(r_upper_arm + r_lower_arm)
        arm_len = sum(arm_lengths) / len(arm_lengths) if arm_lengths else torso_len

        # 头: 鼻子到肩中点
        nose_conf = keypoints[0, 2]
        head_len = float(
            np.linalg.norm(keypoints[0, :2] - shoulder_mid)) if nose_conf > CONF_THRESH else torso_len * 0.35
        body_len = torso_len + leg_len

        # 总高度 (像素)
        top_y = min(keypoints[i, 1] for i in range(17) if keypoints[i, 2] > CONF_THRESH)
        bottom_y = max(keypoints[i, 1] for i in range(17) if keypoints[i, 2] > CONF_THRESH)
        relative_height = float(bottom_y - top_y)

        return BodyProportions(
            torso_leg_ratio=torso_len / leg_len if leg_len > 1e-6 else 0.0,
            shoulder_hip_ratio=shoulder_w / hip_w,
            arm_torso_ratio=arm_len / torso_len,
            head_body_ratio=head_len / body_len if body_len > 1e-6 else 0.0,
            relative_height_px=relative_height,
        )

    @staticmethod
    def similarity(a: BodyProportions, b: BodyProportions) -> float:
        """
        计算两个体型比例的相似度 [0, 1]
        使用高斯核: exp(-||a - b||² / (2 * sigma²))
        """
        va = a.to_vector()
        vb = b.to_vector()
        diff = va - vb
        sigma = 0.15  # 允许约 15% 的比例差异
        return float(np.exp(-np.dot(diff, diff) / (2 * sigma * sigma)))


# ==============================================================================
# 人物档案 (底库核心)
# ==============================================================================

class PersonProfile(BaseModel):
    """
    人物档案 — 底库核心数据结构
    
    包含人脸池 (按姿态分桶) + 衣橱记忆库 + 体型比例。
    支持特征入库、衰减淘汰、换装适应。
    """

    person_id: str
    display_name: str

    # 人脸特征池 (按姿态分桶)
    face_features: dict[PoseBucket, list[FeatureEntry]] = Field(default_factory=lambda: {
        PoseBucket.FRONTAL: [],
        PoseBucket.LEFT: [],
        PoseBucket.RIGHT: [],
        PoseBucket.BACK: [],
    })

    # 人体特征池 (按姿态分桶) — 与人脸不同, 换装导致桶内多峰, 不做质心
    body_features: dict[PoseBucket, list[FeatureEntry]] = Field(default_factory=lambda: {
        PoseBucket.FRONTAL: [],
        PoseBucket.LEFT: [],
        PoseBucket.RIGHT: [],
        PoseBucket.BACK: [],
    })

    # 衣橱记忆库
    wardrobe: list[OutfitRecord] = Field(default_factory=list)

    # 体型比例 (累积平均)
    body_proportions: BodyProportions | None = None
    body_proportions_samples: int = 0

    # VLM 文字描述
    vlm_description: str | None = None

    # 元数据
    created_at: float = Field(default_factory=time.time)
    last_updated: float = Field(default_factory=time.time)
    update_count: int = 0

    # 人脸质心缓存 (不序列化, 惰性计算)
    _face_centroids: dict[PoseBucket, np.ndarray] | None = None

    @staticmethod
    def create_new(display_name: str = "") -> PersonProfile:
        """创建新人物档案"""
        pid = f"person_{uuid.uuid4().hex[:8]}"
        return PersonProfile(
            person_id=pid,
            display_name=display_name or pid,
        )

    def get_face_centroids(self) -> dict[PoseBucket, np.ndarray]:
        """获取人脸质心缓存 (惰性计算, enroll 时自动失效)。

        每个姿态桶内做 质量×时间衰减 加权质心, L2 归一化。
        enroll 的时候已经考虑时间衰减，如果时间衰减显著，enroll 大概率会成功，同时触发这个质心变化，
        反之如果时间不长，enroll 不成功，因此缓存也应当成立。

        Returns:
            {PoseBucket: centroid_embedding}，空桶不包含在结果中。
        """
        if self._face_centroids is not None:
            return self._face_centroids

        now = time.time()
        half_life = get_config().gallery.face_match_half_life_days

        centroids: dict[PoseBucket, np.ndarray] = {}
        for bucket, entries in self.face_features.items():
            if not entries:
                continue

            weights = np.array([
                e.quality_score * e.time_decay_weight(now, half_life)
                for e in entries
            ])
            if weights.sum() < 1e-8:
                continue

            embeddings = np.stack([e.embedding for e in entries])
            centroid = np.average(embeddings, axis=0, weights=weights)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[bucket] = centroid

        self._face_centroids = centroids
        return centroids

    @staticmethod
    def _effective_quality(f: FeatureEntry, now: float,
                           half_life_days: float) -> float:
        """计算衰减后的有效质量 — 比例衰减, 半衰期对所有质量等级一致
        
        effective = quality × (1 - 0.5 × age_days / half_life_days)
        含义: 无论原始质量多少, 经过 half_life_days 天后都精确衰减到一半
        下限: 最多衰减到原始质量的 50%
        """
        age_days = (now - f.timestamp) / 86400
        ratio = min(age_days / half_life_days, 1.0)  # 最多衰减 50%
        return f.quality_score * (1.0 - 0.5 * ratio)

    def _enroll_feature(self, features: list[FeatureEntry], entry: FeatureEntry,
                        max_per_bucket: int, half_life_days: float,
                        ) -> tuple[bool, FeatureEntry | None]:
        """通用入库: 未满直接加, 满了按衰减后质量淘汰最差的。

        Returns:
            (success, evicted): success=False 表示质量不够未入库。
            evicted 为被淘汰的旧条目, 桶未满时为 None。
        """
        if len(features) < max_per_bucket:
            features.append(entry)
            return True, None

        now = time.time()
        min_idx = min(range(len(features)),
                      key=lambda i: self._effective_quality(features[i], now, half_life_days))

        if entry.quality_score > self._effective_quality(features[min_idx], now, half_life_days):
            evicted = features[min_idx]
            features[min_idx] = entry
            return True, evicted
        return False, None

    def enroll_face(self, entry: FeatureEntry) -> FeatureOperation | None:
        """入库人脸特征 — 质量门槛 + 时间衰减淘汰。失败返回 None。"""
        if entry.pose_bucket == PoseBucket.UNKNOWN:
            return None
        gallery_cfg = get_config().gallery
        if entry.quality_score < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Face quality {:.3f} below threshold {:.3f} for {}",
                entry.quality_score, gallery_cfg.quality_enroll_threshold, self.person_id,
            )
            return None
        success, evicted = self._enroll_feature(
            self.face_features[entry.pose_bucket], entry,
            gallery_cfg.max_faces_per_bucket,
            gallery_cfg.face_enroll_half_life_days,
        )
        if success:
            self._face_centroids = None  # 失效缓存
            logger.debug(
                "Enrolled face for {} in bucket {} (quality={:.3f})",
                self.person_id, entry.pose_bucket.value, entry.quality_score,
            )
            return FeatureOperation(entry=entry, evicted=evicted, kind="face")
        return None

    def enroll_body_feature(self, entry: FeatureEntry) -> FeatureOperation | None:
        """入库人体特征 — 质量门槛 + 时间衰减淘汰。失败返回 None。"""
        if entry.pose_bucket == PoseBucket.UNKNOWN:
            return None
        gallery_cfg = get_config().gallery
        if entry.quality_score < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Body feature quality {:.3f} below threshold for {}",
                entry.quality_score, self.person_id,
            )
            return None
        success, evicted = self._enroll_feature(
            self.body_features[entry.pose_bucket], entry,
            gallery_cfg.max_body_per_bucket,
            gallery_cfg.body_enroll_half_life_days,
        )
        if success:
            logger.debug(
                "Enrolled body feature for {} in bucket {} (quality={:.3f})",
                self.person_id, entry.pose_bucket.value, entry.quality_score,
            )
            return FeatureOperation(entry=entry, evicted=evicted, kind="body")
        return None

    def enroll_outfit(self, body_embedding: np.ndarray, quality: float) -> OutfitEnrollResult | None:
        """入库/更新衣橱记录 — 质量门槛 + EMA 更新。失败返回 None。"""
        gallery_cfg = get_config().gallery
        if quality < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Body quality {:.3f} below threshold for {}",
                quality, self.person_id,
            )
            return None

        now = time.time()

        # 检查是否与现有衣橱匹配
        for outfit in self.wardrobe:
            sim = float(np.dot(body_embedding, outfit.body_embedding))
            if sim > gallery_cfg.outfit_match_threshold:
                # 记住更新前的快照 (用于 DB 定位)
                old_snapshot = outfit.model_copy(deep=True)
                # EMA 更新特征
                alpha = 0.3
                updated = (1 - alpha) * outfit.body_embedding + alpha * body_embedding
                norm = np.linalg.norm(updated)
                if norm < 0.1:
                    return None
                outfit.body_embedding = updated / norm
                outfit.last_seen = now
                outfit.seen_count += 1
                logger.debug(
                    "Updated outfit for {} (quality={:.3f}, wardrobe_size={})",
                    self.person_id, quality, len(self.wardrobe),
                )
                return OutfitEnrollResult(outfit=outfit, updated=old_snapshot)

        # 新衣服
        new_outfit = OutfitRecord(
            body_embedding=body_embedding.copy(),
            quality_score=quality,
            first_seen=now,
            last_seen=now,
        )

        evicted: OutfitRecord | None = None
        if len(self.wardrobe) < gallery_cfg.max_outfits:
            self.wardrobe.append(new_outfit)
        else:
            oldest_idx = min(range(len(self.wardrobe)), key=lambda i: self.wardrobe[i].last_seen)
            evicted = self.wardrobe[oldest_idx]
            self.wardrobe[oldest_idx] = new_outfit

        logger.debug(
            "Enrolled new outfit for {} (quality={:.3f}, wardrobe_size={})",
            self.person_id, quality, len(self.wardrobe),
        )
        return OutfitEnrollResult(outfit=new_outfit, evicted=evicted)

    def update_proportions(self, proportions: BodyProportions) -> None:
        """累积平均更新体型比例。"""
        if self.body_proportions is None:
            self.body_proportions = proportions
            self.body_proportions_samples = 1
        else:
            n = self.body_proportions_samples
            old = self.body_proportions.to_vector()
            new = proportions.to_vector()
            avg = (old * n + new) / (n + 1)
            self.body_proportions = BodyProportions(
                torso_leg_ratio=float(avg[0]),
                shoulder_hip_ratio=float(avg[1]),
                arm_torso_ratio=float(avg[2]),
                head_body_ratio=float(avg[3]),
                relative_height_px=proportions.relative_height_px,
            )
            self.body_proportions_samples = n + 1
        logger.debug(
            "Updated proportions for {} (samples={})",
            self.person_id, self.body_proportions_samples,
        )


    def total_face_features(self) -> int:
        """所有姿态桶的人脸特征总数"""
        return sum(len(feats) for feats in self.face_features.values())
