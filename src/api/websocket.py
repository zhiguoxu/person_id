"""
WebSocket 处理器 — 视频流实时通信 (per-camera)

每个 WebSocket 连接绑定一个 camera_id, 支持两种数据来源:

1. 前端推流模式 (原有): 浏览器抓帧上传二进制 JPEG → 服务端识别 → 返回结果 JSON
2. 服务端拉流模式 (StreamConsumer 活跃时): 服务端后台拉流识别, 通过本连接向
   前端推送 [二进制 JPEG 帧 + frame_result JSON + event JSON], 前端只负责渲染;
   此模式下客户端上传的帧会被拒绝 (避免双数据源打乱同一 orchestrator 的追踪状态)。

生命周期: orchestrator 按 (WebSocket 连接数 + 是否有拉流消费器) 引用计数回收,
不再随单个连接断开而销毁 —— 后台拉流需要在无人观看时继续运行。
GPU 模型通过 cache 全局共享，不重复加载。
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import WebSocket
from loguru import logger

from src.api import registry
from src.api.schemas import (
    JsonValue,
    WSError,
    WSIdentityConfirm,
    build_frame_result,
    build_ws_event,
)
from src.config import Config, get_config
from src.pipeline.orchestrator import VisionOrchestrator


async def handle_ws_connection(
    websocket: WebSocket,
    camera_id: str,
) -> None:
    """处理单个 WebSocket 连接的完整生命周期。

    Args:
        websocket: FastAPI WebSocket 连接。
        camera_id: 摄像头标识。
    """
    await websocket.accept()

    # 初始化涉及 GPU 模型加载，失败时用 1011 干净关闭，
    # 让前端拿到明确错误码而不是 1006 异常断开
    try:
        orchestrator = await registry.get_or_create_orchestrator(camera_id)
    except Exception:
        logger.exception("VisionOrchestrator 初始化失败: camera={}", camera_id)
        await websocket.close(code=1011, reason="orchestrator init failed")
        return

    registry.ws_client_counts[camera_id] = (
        registry.ws_client_counts.get(camera_id, 0) + 1
    )
    logger.info(
        "WebSocket 已连接: camera={} (连接数={}, gallery={} 人)",
        camera_id,
        registry.ws_client_counts[camera_id],
        len(orchestrator.gallery),
    )

    config = get_config()
    client_id = id(websocket)

    # 注册为观看端: 服务端拉流模式下由 sender 任务把处理结果推给本连接。
    # 无论 consumer 是否已启动都先注册 —— consumer 可能在连接建立后才开启。
    viewer_queue = registry.register_viewer(camera_id)
    sender_task = asyncio.create_task(
        _viewer_sender(websocket, viewer_queue, camera_id, client_id)
    )

    try:
        async for message in _iter_messages(websocket):
            try:
                if isinstance(message, bytes):
                    await _handle_binary(
                        websocket, message, orchestrator, config, camera_id,
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
        sender_task.cancel()
        try:
            await sender_task
        except (asyncio.CancelledError, Exception):
            pass
        registry.unregister_viewer(camera_id, viewer_queue)

        registry.ws_client_counts[camera_id] = max(
            0, registry.ws_client_counts.get(camera_id, 1) - 1
        )
        # 无连接且无拉流消费器时才回收 orchestrator (后台拉流需继续运行)
        await registry.maybe_release_orchestrator(camera_id)
        logger.info(
            "WebSocket 已清理: camera={}, client={} (剩余连接数={})",
            camera_id, client_id, registry.ws_client_counts.get(camera_id, 0),
        )


# ------------------------------------------------------------------
# 服务端拉流 → 观看端推送
# ------------------------------------------------------------------

async def _viewer_sender(
    websocket: WebSocket,
    queue: asyncio.Queue,
    camera_id: str,
    client_id: int,
) -> None:
    """把 StreamConsumer 广播的处理结果推给本连接。

    消息顺序: 二进制 JPEG 帧 → frame_result JSON → event JSON (逐条)。
    发送失败说明连接已断, 直接退出, 由主循环 finally 统一清理。
    """
    try:
        while True:
            item = await queue.get()
            await websocket.send_bytes(item["jpeg"])
            await websocket.send_text(
                json.dumps(item["result"], ensure_ascii=False)
            )
            for event in item.get("events", ()):
                await websocket.send_text(
                    json.dumps(event, ensure_ascii=False)
                )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug(
            "viewer 推送结束: camera={}, client={}, reason={}",
            camera_id, client_id, str(e)[:100],
        )


# ------------------------------------------------------------------
# Message handling
# ------------------------------------------------------------------

async def _handle_binary(
    websocket: WebSocket,
    data: bytes,
    orchestrator: VisionOrchestrator,
    config: Config,
    camera_id: str,
) -> None:
    """处理二进制消息 (JPEG 帧, 前端推流模式)。"""
    # 服务端拉流消费活跃时拒绝客户端上传帧, 避免双数据源共用一个 orchestrator
    if registry.get_stream_consumer(camera_id) is not None:
        await _send_error(
            websocket,
            "consumer_active",
            "Server-side stream consumer is active; client frames are ignored",
        )
        return

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
    response = build_frame_result(result)
    await _send_json(websocket, response.model_dump(mode='json'))

    # 推送新事件
    new_events = orchestrator.drain_new_events()
    for event in new_events:
        ws_event = build_ws_event(event)
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
            result = await orchestrator.confirm_identity(
                track_id=confirm.track_id,
                person_id=confirm.person_id,
                name=confirm.name,
            )
        except Exception as e:
            logger.exception("身份确认失败")
            await _send_error(
                websocket, "confirm_error", str(e) or "Identity confirmation failed"
            )
            return
        if not result.success:
            # 预期内失败(没看清脸等), 前端按 confirm_error 弹窗提示可读 message
            await _send_error(websocket, "confirm_error", result.message)
            return
        await _send_json(
            websocket,
            {
                "type": "identity_confirmed",
                "track_id": confirm.track_id,
                "person_id": result.person_id,
                "name": confirm.name,
            },
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
