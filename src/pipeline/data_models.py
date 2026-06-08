"""
流水线数据模型 — 检测、匹配、追踪、事件

定义流水线中间数据结构，包括:
- NdArray: 可序列化的 numpy 类型别名
- IdentityStatus / EventType: 状态与事件枚举
- FaceResult: 人脸检测结果
- Detection: 人体检测结果
- MatchCandidate / IdentityResult: 匹配候选与身份结果
- TrackedPerson: 追踪人物
- MatchResult: 匹配结果
- PipelineDebug: 调试信息
- SystemEvent: 系统事件
"""
from __future__ import annotations

import time
from enum import Enum

import numpy as np
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict, computed_field, PlainSerializer

from src.gallery.data_models import PoseBucket

# 可序列化的 numpy 数组类型
NdArray = Annotated[np.ndarray, PlainSerializer(lambda x: x.tolist(), return_type=list, when_used='json')]


# ==============================================================================
# 枚举类型
# ==============================================================================

class IdentityStatus(str, Enum):
    """身份识别状态"""
    CONFIDENT = "confident"  # 确信: 仅一人 ≥ X, 远超第二名
    SUSPECTED = "suspected"  # 疑似: Y ≤ 最高 < X
    CONFLICT = "conflict"  # 冲突: 多人 ≥ X
    STRANGER = "stranger"  # 陌生: 所有人 < Y
    DEFINITE = "definite"  # 笃定: 多次高置信确认, 终态
    IDENTIFYING = "identifying"  # 识别中 (Tier 2 异步处理)
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
# 人脸检测结果
# ==============================================================================

class FaceResult(BaseModel):
    """人脸检测+特征提取结果。

    Attributes:
        embedding: 512 维 L2 归一化的 ArcFace 嵌入向量。
        quality: 人脸质量分 [0, 1]。
        landmarks: 5 点人脸关键点, shape (5, 2)。
        bbox: 人脸检测框 (x1, y1, x2, y2) 在原始帧坐标系。
        det_score: 人脸检测置信度。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: np.ndarray
    quality: float
    landmarks: np.ndarray
    bbox: np.ndarray
    det_score: float


# ==============================================================================
# 流水线中间数据
# ==============================================================================

class Detection(BaseModel):
    """单个人体检测结果"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: NdArray  # (x1, y1, x2, y2) 像素坐标
    confidence: float  # 检测置信度
    keypoints: NdArray  # (17, 3) — x, y, conf
    pose_bucket: PoseBucket = PoseBucket.UNKNOWN
    has_face: bool = False  # 是否检测到正脸

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


class MatchCandidate(BaseModel):
    """匹配候选人"""
    person_id: str
    display_name: str
    face_score: float | None = None  # 人脸匹配分 [0, 1]
    body_score: float | None = None  # 全身匹配分 [0, 1]
    proportion_score: float | None = None  # 体型匹配分 [0, 1]
    fused_score: float = 0.0  # 融合匹配分
    face_match_quality: float = 0.0  # 产生人脸最高分的 query 桶质量
    body_match_quality: float = 0.0  # 产生人体最高分的 query 桶质量


class IdentityResult(MatchCandidate):
    """Tier2/VLM 身份识别结果。

    继承 MatchCandidate 的全部匹配分数字段,
    额外增加 status 用于状态机流转。
    """
    person_id: str | None = None  # type: ignore[assignment]  # 初始无身份
    display_name: str | None = None  # type: ignore[assignment]
    status: IdentityStatus = IdentityStatus.IDENTIFYING


class TrackedPerson(BaseModel):
    """追踪中的人物 (Tier 1 输出, 每帧重建)

    纯帧级瞬态数据, 不持有任何跨帧状态。
    身份信息由 Orchestrator 通过 TrackState.identity_result 管理。
    """

    track_id: int  # 追踪器分配的 ID
    detection: Detection  # 当前帧检测结果
    attention_score: float = 0.0  # 注意力评分
    trail: list[tuple[float, float]] = Field(default_factory=list)  # 中心轨迹


class MatchResult(BaseModel):
    """匹配结果 (用于歧义消除)"""
    candidates: list[MatchCandidate] = Field(default_factory=list)  # 候选人列表 (按 fused_score 降序)
    best_match: MatchCandidate | None = None
    status: IdentityStatus = IdentityStatus.STRANGER

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
        status: str = "pending"  # "pending" | "running" | "done" | "skipped"
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
    fused_score: float | None = None
    source: str = "system"  # "system" | "reid" | "vlm" | "human" | "spatial"
    message: str = ""

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return self.model_dump()
