"""
机器人视觉人物识别系统 - 核心数据模型

定义系统中所有核心数据结构，包括:
- PoseBucket: 姿态分桶枚举
- FeatureEntry: 单条特征记录
- OutfitRecord: 衣橱记录
- BodyProportions: 体型比例特征
- PersonProfile: 人物档案 (底库核心)
- Detection / TrackedPerson / MatchResult 等流水线数据
"""
from __future__ import annotations

import time
import uuid

from loguru import logger
from enum import Enum
import numpy as np

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict, computed_field, PlainSerializer

from src.config import get_config

# 可序列化的 numpy 数组类型
NdArray = Annotated[np.ndarray, PlainSerializer(lambda x: x.tolist(), return_type=list, when_used='json')]


# ==============================================================================
# 枚举类型
# ==============================================================================

class PoseBucket(str, Enum):
    """基于 YOLO-Pose 关键点的人体朝向分桶"""
    FRONTAL = "frontal"   # 正面: 鼻子+双眼可见
    LEFT = "left"         # 左侧
    RIGHT = "right"       # 右侧
    BACK = "back"         # 背面
    UNKNOWN = "unknown"   # 关键点不足


class IdentityStatus(str, Enum):
    """身份识别状态"""
    CONFIDENT = "confident"       # 确信: 仅一人 ≥ X, 远超第二名
    SUSPECTED = "suspected"       # 疑似: Y ≤ 最高 < X
    CONFLICT = "conflict"         # 冲突: 多人 ≥ X
    STRANGER = "stranger"         # 陌生: 所有人 < Y
    DEFINITE = "definite"         # 笃定: 多次高置信确认, 终态
    IDENTIFYING = "identifying"   # 识别中 (Tier 2 异步处理)
    SPATIAL_INFERRED = "spatial_inferred"  # 时空约束推断



class EventType(str, Enum):
    """系统事件类型"""
    NEW_PERSON = "new_person"
    IDENTITY_CONFIRMED = "identity_confirmed"
    IDENTITY_CONFLICT = "identity_conflict"
    VLM_INVOKED = "vlm_invoked"
    VLM_RESULT = "vlm_result"
    TRACK_LOST = "track_lost"
    TRACK_RECOVERED = "track_recovered"
    OUTFIT_UPDATED = "outfit_updated"
    HUMAN_CONFIRMED = "human_confirmed"
    GALLERY_UPDATED = "gallery_updated"


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

    embedding: np.ndarray           # L2 归一化特征向量
    pose_bucket: PoseBucket         # 特征提取时的姿态
    quality_score: float            # 综合质量分 [0, 1]
    timestamp: float                # 提取时间 (Unix timestamp)
    source_image: bytes | None = None  # JPEG 缩略图, 供 VLM 使用

    def time_decay_weight(self, now: float, half_life_days: float) -> float:
        """计算时间衰减权重 (指数衰减)"""
        age_days = (now - self.timestamp) / 86400.0
        if age_days < 0:
            return 1.0
        return 0.5 ** (age_days / half_life_days)


class OutfitRecord(BaseModel):
    """
    衣橱记录
    
    记录一套衣服的全身 ReID 特征，支持近因权重计算。
    同一人的不同衣服分开存储。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    body_embedding: np.ndarray      # 2048 维全身特征
    quality_score: float            # 提取时的质量分
    first_seen: float               # 首次穿着时间
    last_seen: float                # 最后穿着时间
    seen_count: int = 1             # 穿着次数

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


class BodyProportions(BaseModel):
    """
    体型比例特征 (基于 COCO 17 关键点)
    
    零额外模型开销, 利用 YOLO-Pose 已输出的关键点计算骨骼几何比例。
    衣服无关的辅助身份信号。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    torso_leg_ratio: float          # 躯干/腿长比例
    shoulder_hip_ratio: float       # 肩宽/髋宽比例
    arm_torso_ratio: float          # 手臂/躯干比例
    head_body_ratio: float          # 头/身体比例
    relative_height_px: float       # 帧内相对高度 (像素)

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
        head_len = float(np.linalg.norm(keypoints[0, :2] - shoulder_mid)) if nose_conf > CONF_THRESH else torso_len * 0.35
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
    last_seen: float = Field(default_factory=time.time)
    total_appearances: int = 0

    @staticmethod
    def create_new(display_name: str = "") -> PersonProfile:
        """创建新人物档案"""
        pid = f"person_{uuid.uuid4().hex[:8]}"
        return PersonProfile(
            person_id=pid,
            display_name=display_name or pid,
        )

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
                        max_per_bucket: int, half_life_days: float) -> bool:
        """通用入库: 未满直接加, 满了按衰减后质量淘汰最差的"""
        if len(features) < max_per_bucket:
            features.append(entry)
            return True
        
        now = time.time()
        min_idx = min(range(len(features)),
                      key=lambda i: self._effective_quality(features[i], now, half_life_days))
        
        if entry.quality_score > self._effective_quality(features[min_idx], now, half_life_days):
            features[min_idx] = entry
            return True
        return False

    def enroll_face(self, entry: FeatureEntry) -> bool:
        """入库人脸特征 — 质量门槛 + 时间衰减淘汰。"""
        gallery_cfg = get_config().gallery
        if entry.quality_score < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Face quality {:.3f} below threshold {:.3f} for {}",
                entry.quality_score, gallery_cfg.quality_enroll_threshold, self.person_id,
            )
            return False
        success = self._add_feature(
            entry, self.face_features, gallery_cfg.max_faces_per_bucket,
            gallery_cfg.face_enroll_half_life_days,
        )
        if success:
            logger.debug(
                "Enrolled face for {} in bucket {} (quality={:.3f})",
                self.person_id, entry.pose_bucket.value, entry.quality_score,
            )
        return success

    def enroll_body_feature(self, entry: FeatureEntry) -> bool:
        """入库人体特征 — 质量门槛 + 时间衰减淘汰。"""
        gallery_cfg = get_config().gallery
        if entry.quality_score < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Body feature quality {:.3f} below threshold for {}",
                entry.quality_score, self.person_id,
            )
            return False
        success = self._add_feature(
            entry, self.body_features, gallery_cfg.max_body_per_bucket,
            gallery_cfg.body_enroll_half_life_days,
        )
        if success:
            logger.debug(
                "Enrolled body feature for {} in bucket {} (quality={:.3f})",
                self.person_id, entry.pose_bucket.value, entry.quality_score,
            )
        return success

    def _add_feature(
        self,
        entry: FeatureEntry,
        pool: dict[PoseBucket, list[FeatureEntry]],
        max_per_bucket: int,
        half_life_days: float,
    ) -> bool:
        """通用特征入库: 按姿态分桶 → 衰减淘汰。"""
        bucket = entry.pose_bucket
        if bucket not in pool:
            pool[bucket] = []
        return self._enroll_feature(
            pool[bucket], entry, max_per_bucket, half_life_days,
        )

    def enroll_outfit(self, body_embedding: np.ndarray, quality: float) -> None:
        """入库/更新衣橱记录 — 质量门槛 + EMA 更新。"""
        gallery_cfg = get_config().gallery
        if quality < gallery_cfg.quality_enroll_threshold:
            logger.debug(
                "Body quality {:.3f} below threshold for {}",
                quality, self.person_id,
            )
            return

        now = time.time()

        # 检查是否与现有衣橱匹配
        for outfit in self.wardrobe:
            sim = float(np.dot(body_embedding, outfit.body_embedding))
            if sim > gallery_cfg.outfit_match_threshold:
                # EMA 更新特征
                alpha = 0.3
                updated = (1 - alpha) * outfit.body_embedding + alpha * body_embedding
                norm = np.linalg.norm(updated)
                if norm < 0.1:
                    return
                outfit.body_embedding = updated / norm
                outfit.last_seen = now
                outfit.seen_count += 1
                logger.debug(
                    "Updated outfit for {} (quality={:.3f}, wardrobe_size={})",
                    self.person_id, quality, len(self.wardrobe),
                )
                return

        # 新衣服
        new_outfit = OutfitRecord(
            body_embedding=body_embedding.copy(),
            quality_score=quality,
            first_seen=now,
            last_seen=now,
        )

        if len(self.wardrobe) < gallery_cfg.max_outfits:
            self.wardrobe.append(new_outfit)
        else:
            oldest_idx = min(range(len(self.wardrobe)), key=lambda i: self.wardrobe[i].last_seen)
            self.wardrobe[oldest_idx] = new_outfit

        logger.debug(
            "Enrolled new outfit for {} (quality={:.3f}, wardrobe_size={})",
            self.person_id, quality, len(self.wardrobe),
        )

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

    def touch(self, now: float | None = None) -> None:
        """更新最后出现时间和出现次数"""
        self.last_seen = now or time.time()
        self.total_appearances += 1

    def total_face_features(self) -> int:
        """所有姿态桶的人脸特征总数"""
        return sum(len(feats) for feats in self.face_features.values())


# ==============================================================================
# 流水线中间数据
# ==============================================================================

class Detection(BaseModel):
    """单个人体检测结果"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: NdArray                   # (x1, y1, x2, y2) 像素坐标
    confidence: float               # 检测置信度
    keypoints: NdArray              # (17, 3) — x, y, conf
    pose_bucket: PoseBucket = PoseBucket.UNKNOWN
    has_face: bool = False          # 是否检测到正脸

    @property
    def center(self) -> np.ndarray:
        """检测框中心点"""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    @property
    def area(self) -> float:
        """检测框面积"""
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))

    @property
    def height(self) -> float:
        """检测框高度"""
        return float(self.bbox[3] - self.bbox[1])


class IdentityResult(BaseModel):
    """Tier2/VLM 身份识别结果。

    封装一次身份识别产生的所有信息，由 Tier2 或人工确认写入，
    Tier1 在后续帧中通过缓存复用。
    """
    person_id: str | None = None
    display_name: str | None = None
    status: IdentityStatus = IdentityStatus.IDENTIFYING
    confidence: float = 0.0
    face_quality: float | None = None


class TrackedPerson(BaseModel):
    """追踪中的人物 (Tier 1 输出, 每帧重建)

    纯帧级瞬态数据, 不持有任何跨帧状态。
    身份信息由 Orchestrator 通过 TrackState.identity_result 管理。
    """

    track_id: int                   # 追踪器分配的 ID
    detection: Detection            # 当前帧检测结果
    attention_score: float = 0.0    # 注意力评分
    trail: list[tuple[float, float]] = Field(default_factory=list)  # 中心轨迹


class MatchCandidate(BaseModel):
    """匹配候选人"""
    person_id: str
    display_name: str
    face_score: float | None = None      # 人脸匹配分 [0, 1]
    body_score: float | None = None      # 全身匹配分 [0, 1]
    proportion_score: float | None = None # 体型匹配分 [0, 1]
    fused_score: float = 0.0                # 融合匹配分
    face_match_quality: float = 0.0         # 产生人脸最高分的 query 桶质量
    body_match_quality: float = 0.0         # 产生人体最高分的 query 桶质量


class MatchResult(BaseModel):
    """匹配结果 (用于歧义消除)"""
    candidates: list[MatchCandidate] = Field(default_factory=list)  # 候选人列表 (按 fused_score 降序)
    best_match: MatchCandidate | None = None
    status: IdentityStatus = IdentityStatus.STRANGER
    face_quality: float = 0.0               # 当前人脸质量

    @computed_field
    @property
    def top_score(self) -> float:
        return self.candidates[0].fused_score if self.candidates else 0.0

    @computed_field
    @property
    def margin(self) -> float:
        """第一名和第二名的分差"""
        if len(self.candidates) < 2:
            return self.top_score
        return self.candidates[0].fused_score - self.candidates[1].fused_score


class PipelineDebug(BaseModel):
    """流水线调试信息 (用于前端可视化)"""

    class StageInfo(BaseModel):
        status: str = "pending"     # "pending" | "running" | "done" | "skipped"
        time_ms: float = 0.0
        details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    detection: StageInfo = Field(default_factory=StageInfo)
    pose: StageInfo = Field(default_factory=StageInfo)
    face: StageInfo = Field(default_factory=StageInfo)
    reid: StageInfo = Field(default_factory=StageInfo)
    matching: StageInfo = Field(default_factory=StageInfo)
    identity: StageInfo = Field(default_factory=StageInfo)


class SystemEvent(BaseModel):
    """系统事件 (用于前端事件时间线)"""
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    track_id: int | None = None
    person_id: str | None = None
    display_name: str | None = None
    confidence: float | None = None
    source: str = "system"          # "system" | "reid" | "vlm" | "human" | "spatial"
    message: str = ""

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return self.model_dump()
