"""
API 数据模型 — Pydantic schemas

定义 REST API 和 WebSocket 通信使用的所有请求/响应模型。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ==============================================================================
# Frame Processing
# ==============================================================================

class ProcessFrameRequest(BaseModel):
    """帧处理请求 (REST, 仅调试用; 正式走 WebSocket)。"""
    image_b64: str = Field(..., description="Base64 encoded JPEG image")
    debug: bool = Field(False, description="Include debug info")


class TrackedPersonResponse(BaseModel):
    """单个被追踪人物的响应。"""
    track_id: int
    bbox: Optional[list[float]] = None
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    identity_status: str = "identifying"
    confidence: float = 0.0
    attention_score: float = 0.0
    is_current_target: bool = False
    face_quality: Optional[float] = None
    trail: list[list[float]] = Field(default_factory=list)
    thumbnail_b64: Optional[str] = None
    keypoints: Optional[list[list[float]]] = None
    pose_bucket: Optional[str] = None


class CurrentTargetResponse(BaseModel):
    """当前注意力目标。"""
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    display_name: Optional[str] = None


class ProcessFrameResponse(BaseModel):
    """帧处理响应。"""
    frame_id: int = 0
    tracked_persons: list[TrackedPersonResponse] = Field(default_factory=list)
    current_target: Optional[CurrentTargetResponse] = None
    pending_tier2: list[int] = Field(default_factory=list)
    gallery_size: int = 0
    processing_ms: float = 0.0
    debug: bool = False


# ==============================================================================
# Identity Confirmation
# ==============================================================================

class ConfirmIdentityRequest(BaseModel):
    """人工确认身份请求。"""
    track_id: int = Field(..., description="Track ID to confirm")
    person_id: str = Field(..., description="Gallery person ID")
    name: str = Field(..., description="Display name")


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
    last_seen: float = 0.0
    total_appearances: int = 0


class PersonListResponse(BaseModel):
    """人物列表响应。"""
    persons: list[PersonSummary] = Field(default_factory=list)
    total: int = 0


class FeatureEntryInfo(BaseModel):
    """特征条目信息 (详情用)。"""
    pose_bucket: str
    quality_score: float
    timestamp: float


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
    wardrobe: list[OutfitInfo] = Field(default_factory=list)
    has_proportions: bool = False
    vlm_description: Optional[str] = None
    created_at: float = 0.0
    last_seen: float = 0.0
    total_appearances: int = 0


# ==============================================================================
# WebSocket Messages
# ==============================================================================

class WSMessage(BaseModel):
    """WebSocket 消息基类。"""
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class WSFrameResult(BaseModel):
    """WebSocket 帧处理结果。"""
    type: str = "frame_result"
    frame_id: int = 0
    tracked_persons: list[TrackedPersonResponse] = Field(default_factory=list)
    current_target: Optional[CurrentTargetResponse] = None
    processing_ms: float = 0.0
    gallery_size: int = 0
    pending_tier2: list[int] = Field(default_factory=list)
    pipeline_debug: Optional[dict[str, Any]] = None


class WSConfigUpdate(BaseModel):
    """WebSocket 配置更新消息。"""
    type: str = "config_update"
    updates: dict[str, float] = Field(default_factory=dict)


class WSIdentityConfirm(BaseModel):
    """WebSocket 身份确认消息。"""
    type: str = "confirm_identity"
    track_id: int
    person_id: str
    name: str


class WSEvent(BaseModel):
    """WebSocket 系统事件推送。"""
    type: str = "event"
    event_type: str
    timestamp: float
    track_id: Optional[int] = None
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "system"
    message: str = ""


class WSError(BaseModel):
    """WebSocket 错误消息。"""
    type: str = "error"
    message: str
    code: str = "unknown"
