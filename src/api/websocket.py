"""
WebSocket 处理器 — 视频流实时通信 (per-camera)

每个 WebSocket 连接绑定一个 camera_id:
- 首次连接时创建该摄像头的 VisionOrchestrator (或复用已有)
- 断开时 shutdown 该摄像头的 orchestrator
- GPU 模型通过 cache 全局共享，不重复加载
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import WebSocket
from loguru import logger

from src.api.schemas import (
    JsonValue,
    WSError,
    WSEvent,
    WSFrameResult,
    WSIdentityConfirm,
)
from src.config import Config, get_config
from src.pipeline.orchestrator import VisionOrchestrator


async def handle_ws_connection(
    websocket: WebSocket,
    camera_id: str,
    registry: dict[str, VisionOrchestrator],
) -> None:
    """处理单个 WebSocket 连接的完整生命周期。

    Args:
        websocket: FastAPI WebSocket 连接。
        camera_id: 摄像头标识。
        registry: 全局摄像头注册表 (server.py 中的 camera_registry)。
    """
    await websocket.accept()

    # 获取或创建该摄像头的 orchestrator
    if camera_id in registry:
        orchestrator = registry[camera_id]
        logger.info(
            "WebSocket 已连接: camera={} (复用已有 orchestrator)",
            camera_id,
        )
    else:
        orchestrator = await VisionOrchestrator.create(camera_id=camera_id)
        registry[camera_id] = orchestrator
        logger.info(
            "WebSocket 已连接: camera={} (新建 orchestrator, gallery={} 人)",
            camera_id, len(orchestrator.gallery),
        )

    config = get_config()
    client_id = id(websocket)

    try:
        async for message in _iter_messages(websocket):
            try:
                if isinstance(message, bytes):
                    await _handle_binary(
                        websocket, message, orchestrator, config,
                    )
                elif isinstance(message, str):
                    await _handle_text(
                        websocket, message, orchestrator, config,
                    )
            except Exception:
                logger.exception(
                    "处理消息出错: camera={}, client={}",
                    camera_id, client_id,
                )
                await _send_error(
                    websocket, "processing_error", "Internal error"
                )
    except Exception as e:
        logger.info(
            "WebSocket 已断开: camera={}, client={}, reason={}",
            camera_id, client_id, str(e)[:100],
        )
    finally:
        # 断开连接时 shutdown 并从注册表移除
        if camera_id in registry:
            await registry[camera_id].shutdown()
            del registry[camera_id]
        logger.info(
            "WebSocket 已清理: camera={}, client={}",
            camera_id, client_id,
        )


# ------------------------------------------------------------------
# Message handling
# ------------------------------------------------------------------

async def _handle_binary(
    websocket: WebSocket,
    data: bytes,
    orchestrator: VisionOrchestrator,
    config: Config,
) -> None:
    """处理二进制消息 (JPEG 帧)。"""
    max_size = config.server.ws_max_frame_size
    if len(data) > max_size:
        await _send_error(
            websocket,
            "frame_too_large",
            f"Frame size {len(data)} exceeds max {max_size}",
        )
        return

    # 镜头畸变矫正 (可通过前端开关控制)
    if config.server.image_correction_enabled:
        try:
            from src.utils.image_correction import correct_image_bytes
            data = correct_image_bytes(data)
        except Exception:
            logger.warning("图像矫正失败，使用原始帧")

    # 解码 JPEG
    try:
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            await _send_error(
                websocket, "decode_error", "Failed to decode JPEG"
            )
            return
    except Exception:
        await _send_error(
            websocket, "decode_error", "Invalid image data"
        )
        return

    # 处理帧
    try:
        result = await orchestrator.process_frame(frame)
    except Exception:
        logger.exception("帧处理失败")
        await _send_error(
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
        pending_vlm=result.get("pending_vlm", []),
        pipeline_debug=result.get("pipeline_debug"),
    )
    await _send_json(websocket, response.model_dump(mode='json'))

    # 推送新事件
    new_events = orchestrator.drain_new_events()
    for event in new_events:
        ws_event = WSEvent(
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
        await _send_json(websocket, ws_event.model_dump(mode='json'))


async def _handle_text(
    websocket: WebSocket,
    text: str,
    orchestrator: VisionOrchestrator,
    config: Config,
) -> None:
    """处理文本消息 (JSON 控制命令)。"""
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        await _send_error(websocket, "json_error", "Invalid JSON")
        return

    msg_type = msg.get("type", "")

    if msg_type == "confirm_identity":
        try:
            confirm = WSIdentityConfirm(**msg)
            await orchestrator.confirm_identity(
                track_id=confirm.track_id,
                person_id=confirm.person_id,
                name=confirm.name,
            )
            await _send_json(
                websocket,
                {
                    "type": "identity_confirmed",
                    "track_id": confirm.track_id,
                    "person_id": confirm.person_id,
                    "name": confirm.name,
                },
            )
        except Exception as e:
            logger.exception("身份确认失败")
            await _send_error(
                websocket, "confirm_error", str(e) or "Identity confirmation failed"
            )

    elif msg_type == "ping":
        await _send_json(websocket, {"type": "pong"})

    else:
        await _send_error(
            websocket, "unknown_type", f"Unknown message type: {msg_type}"
        )


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

async def _send_json(websocket: WebSocket, data: dict[str, JsonValue]) -> None:
    """发送 JSON 数据到 WebSocket。"""
    text = json.dumps(data, ensure_ascii=False)
    await websocket.send_text(text)


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    """发送错误消息。"""
    error = WSError(code=code, message=message)
    await _send_json(websocket, error.model_dump(mode='json'))


async def _iter_messages(websocket: WebSocket) -> AsyncIterator[bytes | str]:
    """FastAPI WebSocket 消息迭代器。"""
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
