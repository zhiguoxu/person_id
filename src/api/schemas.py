"""
API 数据模型 — Pydantic schemas

定义 REST API 和 WebSocket 通信使用的所有请求/响应模型。
"""
from __future__ import annotations


from src.gallery.data_models import BodyProportions

from typing import Any

from pydantic import BaseModel, Field

from src.pipeline.data_models import IdentityResult, TrackedPerson

# JSON 兼容值类型
JsonValue = Any


# ==============================================================================
# Frame Processing
# ==============================================================================


class TrackedPersonResponse(BaseModel):
    """单个被追踪人物的响应。直接复用服务器的内部数据结构以减少组装开销。"""
    person: TrackedPerson
    identity_result: IdentityResult
    is_current_target: bool = False
    thumbnail_b64: str | None = None


class CurrentTargetResponse(BaseModel):
    """当前注意力目标。"""
    track_id: int | None = None
    person_id: str | None = None
    display_name: str | None = None



# ==============================================================================
# Identity Confirmation
# ==============================================================================

class ConfirmIdentityRequest(BaseModel):
    """人工确认身份请求。"""
    track_id: int = Field(..., description="Track ID to confirm")
    person_id: str | None = Field(None, description="Gallery person ID (None to create new)")
    name: str = Field(..., description="Display name")


class RenamePersonRequest(BaseModel):
    """重命名人物请求。"""
    display_name: str = Field(..., description="New display name", min_length=1, max_length=100)


# ==============================================================================
# Voice-agent integration (语音对话集成: 查询/注册当前对话对象)
# ==============================================================================

class CurrentIdentityResponse(BaseModel):
    """查询"当前镜头前的人是谁"。

    供 voice_agent 在每轮对话前调用, 用于回答"你知道我是谁吗"。
    recognition 字段把内部置信度状态归一化为三档, 让对话端无需理解细节:
    - "known":   已确信识别 (definite/confident), 可直接报出 display_name
    - "suspected": 疑似 (suspected), 不确定, 对话端应试探确认而非断定
    - "unknown": 陌生/识别中/无人/摄像头离线, 必须如实回答不知道, 不可编造
    """
    camera_online: bool = Field(..., description="该 camera_id 是否有活跃视频流")
    has_target: bool = Field(False, description="当前是否有注意力目标(镜头前有人)")
    recognition: str = Field("unknown", description="known | suspected | unknown")
    track_id: int | None = Field(None, description="当前目标的 track_id")
    person_id: str | None = Field(None, description="底库人物 ID (已识别时)")
    display_name: str | None = Field(None, description="人物名称 (已识别时)")
    status: str | None = Field(None, description="原始 IdentityStatus 值, 调试用")
    fused_score: float | None = Field(None, description="融合匹配分, 调试用")


class RegisterCurrentRequest(BaseModel):
    """把"当前镜头前的人"注册到底库并命名 (对应用户说"我是xxx")。

    不接受调用方指定 person_id: 造 key 是本服务的职责 —— 当前目标若已识别则复用其 key、
    否则新建, 由端点内部自行推导。
    """
    name: str = Field(..., description="用户的称呼/姓名", min_length=1, max_length=100)


class RegisterCurrentResponse(BaseModel):
    """注册当前对话对象的结果。

    不回显 camera_id / name: 它们是请求输入, 调用方已有, 回显冗余。
    """
    status: str = Field(
        ...,
        description=(
            "registered | camera_offline | no_target | no_face | "
            "low_face_quality | unknown_person_id"
        ),
    )
    success: bool = Field(...)
    person_id: str | None = Field(None, description="入库/复用的人物 key")
    track_id: int | None = Field(None, description="本次入库的轨迹 ID (调试用)")
    message: str = Field("", description="人类可读的结果说明 (失败时给出原因/引导)")


# ==============================================================================
# Quality Cache
# ==============================================================================

class CachedFrameInfo(BaseModel):
    """质量缓存条目信息 (用于前端展示)。"""
    image_b64: str
    quality: float
    timestamp: float
    pose_bucket: str
    enrolled: bool = False


class QualityCacheResponse(BaseModel):
    """Track 质量缓存响应。"""
    track_id: int
    face_pool: list[CachedFrameInfo] = Field(default_factory=list)
    body_pool: list[CachedFrameInfo] = Field(default_factory=list)


# ==============================================================================
# Config
# ==============================================================================

class TunableParam(BaseModel):
    """单个可调参数。"""
    value: float
    min: float
    max: float
    step: float
    group: str
    label: str


class ConfigResponse(BaseModel):
    """配置响应。"""
    params: dict[str, TunableParam] = Field(default_factory=dict)
    flags: dict[str, JsonValue] = Field(default_factory=dict)  # 开关/阈值状态


class ConfigUpdateRequest(BaseModel):
    """配置更新请求 (key-value 扁平格式)。"""
    updates: dict[str, float] = Field(
        ..., description="Parameter key → new value"
    )


class ConfigUpdateResponse(BaseModel):
    """配置更新响应。"""
    updated_keys: list[str] = Field(default_factory=list)


# ==============================================================================
# Gallery
# ==============================================================================

class PersonSummary(BaseModel):
    """人物摘要 (列表用)。"""
    person_id: str
    display_name: str
    face_count: int = 0
    outfit_count: int = 0
    last_updated: float = 0.0
    update_count: int = 0


class PersonListResponse(BaseModel):
    """人物列表响应。"""
    persons: list[PersonSummary] = Field(default_factory=list)
    total: int = 0


class FeatureEntryInfo(BaseModel):
    """特征条目信息 (详情用)。"""
    pose_bucket: str
    quality_score: float
    timestamp: float
    source_image_b64: str | None = None
    overlay_bbox: list[float] | None = None  # [x1,y1,x2,y2] 叠加框 (人脸框或人体框)


class OutfitInfo(BaseModel):
    """衣橱条目信息。"""
    quality_score: float
    first_seen: float
    last_seen: float
    seen_count: int = 1


class PersonDetailResponse(BaseModel):
    """人物详情响应。"""
    person_id: str
    display_name: str
    face_features: dict[str, list[FeatureEntryInfo]] = Field(
        default_factory=dict
    )
    body_features: dict[str, list[FeatureEntryInfo]] = Field(
        default_factory=dict
    )
    wardrobe: list[OutfitInfo] = Field(default_factory=list)
    body_proportions: BodyProportions | None = None
    vlm_description: str | None = None
    created_at: float = 0.0
    last_updated: float = 0.0
    update_count: int = 0


# ==============================================================================
# WebSocket Messages
# ==============================================================================


class WSFrameResult(BaseModel):
    """WebSocket 帧处理结果。"""
    type: str = "frame_result"
    frame_id: int = 0
    tracked_persons: list[TrackedPersonResponse] = Field(default_factory=list)
    current_target: CurrentTargetResponse | None = None
    processing_ms: float = 0.0
    gallery_size: int = 0
    pending_vlm: list[int] = Field(default_factory=list)
    pipeline_debug: dict[str, JsonValue] | None = None


class WSIdentityConfirm(BaseModel):
    """WebSocket 身份确认消息。"""
    type: str = "confirm_identity"
    track_id: int
    person_id: str | None = None
    name: str


class WSEvent(BaseModel):
    """WebSocket 系统事件推送。"""
    type: str = "event"
    event_type: str
    timestamp: float
    track_id: int | None = None
    person_id: str | None = None
    display_name: str | None = None
    fused_score: float | None = None
    source: str = "system"
    message: str = ""
    candidates: list[dict] = Field(default_factory=list)


class WSError(BaseModel):
    """WebSocket 错误消息。"""
    type: str = "error"
    message: str
    code: str = "unknown"

class BodyQualityTestResponse(BaseModel):
    """测试 body quality 的返回结果"""
    has_person: bool
    quality: float | None = None
    quality_hint: float | None = None
    sharpness: float | None = None
    bbox: list[float] | None = None
    error: str | None = None


class FaceSimilarityFaceInfo(BaseModel):
    """单张图片的人脸检测结果"""
    has_face: bool
    person_bbox: list[float] | None = None
    face_bbox: list[float] | None = None
    face_quality: float | None = None
    aligned_face_b64: str | None = None


class FaceSimilarityTestResponse(BaseModel):
    """人脸相似度测试结果"""
    face1: FaceSimilarityFaceInfo
    face2: FaceSimilarityFaceInfo
    similarity: float | None = None  # backend 默认通道的相似度
    similarity_bgr: float | None = None  # 以 BGR 通道送入模型的相似度
    similarity_rgb: float | None = None  # 以 RGB 通道送入模型的相似度
    corrected_image1_b64: str | None = None  # 畸变矫正后的原图 base64
    corrected_image2_b64: str | None = None
    error: str | None = None


class BodySimilarityBodyInfo(BaseModel):
    """单张图片的人体检测结果"""
    has_body: bool
    person_bbox: list[float] | None = None
    body_crop_b64: str | None = None  # 裁剪后的人体图 base64


class BodySimilarityTestResponse(BaseModel):
    """全身 ReID 相似度测试结果"""
    body1: BodySimilarityBodyInfo
    body2: BodySimilarityBodyInfo
    similarity: float | None = None
    embedding_dim: int | None = None
    corrected_image1_b64: str | None = None  # 畸变矫正后的原图 base64
    corrected_image2_b64: str | None = None
    error: str | None = None


class ReIDCompareResponse(BaseModel):
    """两种 ReID 模型对比结果"""
    body1: BodySimilarityBodyInfo
    body2: BodySimilarityBodyInfo
    solider_similarity: float | None = None
    solider_dim: int | None = None
    osnet_similarity: float | None = None
    osnet_dim: int | None = None
    corrected_image1_b64: str | None = None  # 畸变矫正后的原图 base64
    corrected_image2_b64: str | None = None
    error: str | None = None
