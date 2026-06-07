"""
api — Web API 模块

包含 REST 路由、WebSocket 处理器、Pydantic schemas 和 FastAPI 服务。
"""
from src.api.schemas import (
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ConfirmIdentityRequest,
    CurrentTargetResponse,
    PersonDetailResponse,
    PersonListResponse,
    ProcessFrameRequest,
    ProcessFrameResponse,
    TrackedPersonResponse,
    WSConfigUpdate,
    WSError,
    WSEvent,
    WSFrameResult,
    WSIdentityConfirm,
    WSMessage,
)
from src.api.server import create_app
from src.api.websocket import handle_ws_connection

__all__ = [
    # Server
    "create_app",
    "handle_ws_connection",
    # Schemas
    "ProcessFrameRequest",
    "ProcessFrameResponse",
    "TrackedPersonResponse",
    "CurrentTargetResponse",
    "ConfirmIdentityRequest",
    "ConfigResponse",
    "ConfigUpdateRequest",
    "ConfigUpdateResponse",
    "PersonListResponse",
    "PersonDetailResponse",
    "WSMessage",
    "WSFrameResult",
    "WSConfigUpdate",
    "WSIdentityConfirm",
    "WSEvent",
    "WSError",
]
