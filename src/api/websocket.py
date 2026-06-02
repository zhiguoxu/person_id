"""
WebSocket 处理器 — 视频流实时通信

处理浏览器端通过 WebSocket 发送的:
- 二进制消息: JPEG 帧 → 处理 → 返回 JSON 结果
- 文本消息: 配置更新、身份确认等控制命令

支持多客户端连接和事件广播。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional, Set

import cv2
import numpy as np
from loguru import logger

from src.api.schemas import (
    WSConfigUpdate,
    WSError,
    WSEvent,
    WSFrameResult,
    WSIdentityConfirm,
)
from src.config import Config


class VisionWebSocket:
    """
    WebSocket 连接处理器。

    管理多个 WebSocket 客户端连接，处理帧数据，
    返回识别结果，并广播系统事件。

    Args:
        orchestrator: VisionOrchestrator 实例。
        config: 全局配置。
    """

    def __init__(self, orchestrator: Any, config: Config) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._active_connections: Set[Any] = set()
        self._lock = asyncio.Lock()

        logger.info("VisionWebSocket handler initialized")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def connection_count(self) -> int:
        """当前活跃连接数。"""
        return len(self._active_connections)

    async def handle_connection(self, websocket: Any) -> None:
        """
        处理单个 WebSocket 连接的主循环。

        Args:
            websocket: WebSocket 连接对象
                       (fastapi.WebSocket 或 websockets.WebSocketServerProtocol)。
        """
        # 接受连接
        try:
            await websocket.accept()
        except AttributeError:
            # websockets 库无需显式 accept
            pass

        self._active_connections.add(websocket)
        client_id = id(websocket)
        logger.info(
            "WebSocket connected: client={}, total={}",
            client_id,
            self.connection_count,
        )

        try:
            async for message in self._iter_messages(websocket):
                try:
                    if isinstance(message, bytes):
                        # 二进制: JPEG 帧
                        await self._handle_binary(websocket, message)
                    elif isinstance(message, str):
                        # 文本: JSON 控制消息
                        await self._handle_text(websocket, message)
                    else:
                        logger.warning(
                            "Unknown message type from client={}",
                            client_id,
                        )
                except Exception:
                    logger.exception(
                        "Error processing message from client={}",
                        client_id,
                    )
                    await self._send_error(
                        websocket, "processing_error", "Internal error"
                    )
        except Exception as e:
            logger.info(
                "WebSocket disconnected: client={}, reason={}",
                client_id,
                str(e)[:100],
            )
        finally:
            self._active_connections.discard(websocket)
            logger.info(
                "WebSocket cleaned up: client={}, remaining={}",
                client_id,
                self.connection_count,
            )

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_binary(
        self, websocket: Any, data: bytes
    ) -> None:
        """处理二进制消息 (JPEG 帧)。"""
        # 检查帧大小
        max_size = self._config.server.ws_max_frame_size
        if len(data) > max_size:
            await self._send_error(
                websocket,
                "frame_too_large",
                f"Frame size {len(data)} exceeds max {max_size}",
            )
            return

        # 解码 JPEG
        try:
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await self._send_error(
                    websocket, "decode_error", "Failed to decode JPEG"
                )
                return
        except Exception:
            await self._send_error(
                websocket, "decode_error", "Invalid image data"
            )
            return

        # 处理帧
        try:
            result = await self._orchestrator.process_frame(frame)
        except Exception:
            logger.exception("Frame processing failed")
            await self._send_error(
                websocket, "processing_error", "Frame processing failed"
            )
            return

        # 发送结果
        response = WSFrameResult(
            frame_id=result.get("frame_id", 0),
            tracked_persons=result.get("tracked_persons", []),
            current_target=result.get("current_target"),
            processing_ms=result.get("processing_ms", 0.0),
            gallery_size=result.get("gallery_size", 0),
            pending_tier2=result.get("pending_tier2", []),
            pipeline_debug=result.get("pipeline_debug"),
        )
        await self._send_json(websocket, response.model_dump())

        # 广播新事件
        await self._broadcast_new_events()

    async def _handle_text(self, websocket: Any, text: str) -> None:
        """处理文本消息 (JSON 控制命令)。"""
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            await self._send_error(
                websocket, "json_error", "Invalid JSON"
            )
            return

        msg_type = msg.get("type", "")

        if msg_type == "config_update":
            await self._handle_config_update(websocket, msg)
        elif msg_type == "confirm_identity":
            await self._handle_confirm_identity(websocket, msg)
        elif msg_type == "ping":
            await self._send_json(websocket, {"type": "pong"})
        else:
            await self._send_error(
                websocket,
                "unknown_type",
                f"Unknown message type: {msg_type}",
            )

    async def _handle_config_update(
        self, websocket: Any, msg: dict
    ) -> None:
        """处理配置更新请求。"""
        try:
            update = WSConfigUpdate(**msg)
            updated_keys = self._config.update_from_dict(update.updates)
            await self._send_json(
                websocket,
                {
                    "type": "config_updated",
                    "updated_keys": updated_keys,
                },
            )
            logger.info("Config updated via WS: {}", updated_keys)
        except Exception:
            logger.exception("Config update failed")
            await self._send_error(
                websocket, "config_error", "Config update failed"
            )

    async def _handle_confirm_identity(
        self, websocket: Any, msg: dict
    ) -> None:
        """处理身份确认请求。"""
        try:
            confirm = WSIdentityConfirm(**msg)
            await self._orchestrator.confirm_identity(
                track_id=confirm.track_id,
                person_id=confirm.person_id,
                name=confirm.name,
            )
            await self._send_json(
                websocket,
                {
                    "type": "identity_confirmed",
                    "track_id": confirm.track_id,
                    "person_id": confirm.person_id,
                    "name": confirm.name,
                },
            )
        except Exception:
            logger.exception("Identity confirmation failed")
            await self._send_error(
                websocket,
                "confirm_error",
                "Identity confirmation failed",
            )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, message: dict) -> None:
        """
        向所有连接的客户端广播消息。

        Args:
            message: 要广播的 JSON 消息字典。
        """
        if not self._active_connections:
            return

        data = json.dumps(message, ensure_ascii=False)
        disconnected = set()

        for ws in self._active_connections:
            try:
                await asyncio.wait_for(
                    self._send_text(ws, data),
                    timeout=self._config.server.ws_send_timeout,
                )
            except (asyncio.TimeoutError, Exception):
                disconnected.add(ws)

        # 清理断开的连接
        self._active_connections -= disconnected
        if disconnected:
            logger.debug(
                "Cleaned {} disconnected clients", len(disconnected)
            )

    async def _broadcast_new_events(self) -> None:
        """广播最新系统事件给所有客户端 (每个事件只广播一次)。"""
        new_events = self._orchestrator.drain_new_events()
        for event in new_events:
            ws_event = WSEvent(
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                track_id=event.track_id,
                person_id=event.person_id,
                display_name=event.display_name,
                confidence=event.confidence,
                source=event.source,
                message=event.message,
            )
            await self.broadcast(ws_event.model_dump())

    # ------------------------------------------------------------------
    # Low-level send helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_json(websocket: Any, data: dict) -> None:
        """发送 JSON 数据到 WebSocket。"""
        text = json.dumps(data, ensure_ascii=False)
        try:
            # FastAPI WebSocket
            await websocket.send_text(text)
        except AttributeError:
            # websockets library
            await websocket.send(text)

    @staticmethod
    async def _send_text(websocket: Any, text: str) -> None:
        """发送文本到 WebSocket。"""
        try:
            await websocket.send_text(text)
        except AttributeError:
            await websocket.send(text)

    async def _send_error(
        self, websocket: Any, code: str, message: str
    ) -> None:
        """发送错误消息。"""
        error = WSError(code=code, message=message)
        await self._send_json(websocket, error.model_dump())

    @staticmethod
    async def _iter_messages(websocket: Any):
        """
        统一的消息迭代器，兼容 FastAPI 和 websockets 库。

        Yields:
            bytes | str: 收到的消息。
        """
        # FastAPI WebSocket
        if hasattr(websocket, "receive"):
            while True:
                try:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        yield msg["bytes"]
                    elif "text" in msg and msg["text"]:
                        yield msg["text"]
                except Exception:
                    break
        else:
            # websockets library
            async for message in websocket:
                yield message
