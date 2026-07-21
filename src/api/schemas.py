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


class CurrentTargetResponse(BaseModel):
    """当前注意力目标。"""
    track_id: int | None = None
    person_id: str | None = None
    display_name: str | None = None



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

    造新 key 是本服务的职责, 但"疑似确认"场景例外地接受调用方指定 person_id:
    目标处于 SUSPECTED(疑似)时本服务无法自行断定复用还是新建 —— 断定依据是
    "用户报的名字与疑似候选在花名册里的名字是否一致", 而花名册归对话端所有
    (本服务的 display_name 仅调试用, 改名后不保证同步)。对话端核对一致后带上
    候选 person_id, 本端点把本次人脸特征追加进该 person_id 的既有档案;
    不带则维持原行为(DEFINITE/CONFIDENT→复用当前识别到的 person_id,
    其余→新建一个 person_id)。
    """
    name: str = Field(..., description="用户的称呼/姓名", min_length=1, max_length=100)
    person_id: str | None = Field(
        None,
        description="疑似确认场景由对话端指定: 把本次人脸特征追加进这个底库"
                    "人物 ID 的既有档案(对话端已核对用户报的名字与该人物"
                    "花名册名字一致)。不给则由端点自行推导: DEFINITE/"
                    "CONFIDENT 复用当前识别到的 person_id, 否则新建。",
    )
    min_face_quality: float | None = Field(
        None, ge=0.0, le=1.0,
        description="人脸入库质量门槛下限(可选)。与服务端默认 enroll 阈值取较大值后生效, "
                    "即只能提高、不能降低门槛 —— 供主动注册流程用更高标准采集底片; "
                    "不给则用服务端默认阈值。",
    )


class RegisterCurrentResponse(BaseModel):
    """注册当前对话对象的结果。

    不回显 camera_id / name: 它们是请求输入, 调用方已有, 回显冗余。
    """
    status: str = Field(
        ...,
        description=(
            "registered | already_known | camera_offline | no_target | "
            "no_face | low_face_quality | unknown_person_id"
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
    avatar_b64: str | None = None  # 头像: 底库中质量最高的人脸特征缩略图
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
    # 处理帧尺寸 (服务端拉流模式下发): 检测坐标的基准。
    # 预览 JPEG 可能为省带宽而缩小, 前端必须以此字段而非预览图尺寸做坐标映射。
    frame_w: int | None = None
    frame_h: int | None = None


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


class DeviceStreamStartResponse(BaseModel):
    """开启设备推流 (ISS start_stream) 的响应。"""
    flv_url: str


class DeviceStreamStopResponse(BaseModel):
    """停止设备推流 (ISS stop_stream) 的响应。"""
    stopped: bool


class StreamStartRequest(BaseModel):
    """开启服务端拉流消费的请求。"""
    url: str


class StreamStatusResponse(BaseModel):
    """服务端拉流消费状态。"""
    camera_id: str
    running: bool = False
    connected: bool = False  # 是否已成功连上视频流
    url: str | None = None
    # 实际拉到的流原生分辨率 (识别按此分辨率无损处理; 换设备后可在此确认)
    stream_width: int = 0
    stream_height: int = 0
    frames_read: int = 0
    frames_processed: int = 0
    process_fps: float = 0.0
    viewers: int = 0
    last_error: str | None = None


def build_frame_result(result: dict) -> WSFrameResult:
    """把 orchestrator.process_frame 的返回字典转成 WSFrameResult。

    WebSocket 推流路径与服务端拉流路径共用, 保证两种模式下前端收到的结构一致。
    """
    return WSFrameResult(
        frame_id=result.get("frame_id", 0),
        tracked_persons=result.get("tracked_persons", []),
        current_target=result.get("current_target"),
        processing_ms=result.get("processing_ms", 0.0),
        gallery_size=result.get("gallery_size", 0),
        pending_vlm=result.get("pending_vlm", []),
        pipeline_debug=result.get("pipeline_debug"),
    )


def build_ws_event(event) -> WSEvent:
    """把 orchestrator 的 SystemEvent 转成 WSEvent (duck-typed, 避免循环导入)。"""
    return WSEvent(
        event_type=event.event_type.value,
        timestamp=event.timestamp,
        track_id=event.track_id,
        person_id=event.person_id,
        display_name=event.display_name,
        fused_score=event.fused_score,
        source=event.source,
        message=event.message,
        candidates=event.candidates,
    )

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
