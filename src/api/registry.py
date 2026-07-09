"""
全局摄像头注册表

独立模块，避免 __main__ vs src.api.server 双实例问题。
所有模块统一从此处导入 camera_registry。

除 orchestrator 外还维护三类状态:
- consumer_registry: 服务端拉流消费器 (camera_id → StreamConsumer)
- viewer_queues: 观看端广播队列 (camera_id → 每个 WebSocket 连接一个 Queue)
- ws_client_counts: 每个摄像头当前的 WebSocket 连接数 (决定 orchestrator 何时可回收)
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.pipeline.orchestrator import VisionOrchestrator

if TYPE_CHECKING:
    from src.pipeline.stream_consumer import StreamConsumer

# 全局摄像头注册表: camera_id → VisionOrchestrator
camera_registry: dict[str, VisionOrchestrator] = {}

# 服务端拉流消费器: camera_id → StreamConsumer
consumer_registry: dict[str, "StreamConsumer"] = {}

# 观看端广播队列: camera_id → set[asyncio.Queue]
viewer_queues: dict[str, set[asyncio.Queue]] = {}

# 每个摄像头的活跃 WebSocket 连接数
ws_client_counts: dict[str, int] = {}


def get_camera_orchestrator(camera_id: str) -> VisionOrchestrator | None:
    """获取指定摄像头的编排器（供 REST routes 使用）。"""
    return camera_registry.get(camera_id)


async def get_or_create_orchestrator(camera_id: str) -> VisionOrchestrator:
    """获取或创建指定摄像头的编排器并注册。"""
    orch = camera_registry.get(camera_id)
    if orch is None:
        orch = await VisionOrchestrator.create(camera_id=camera_id)
        camera_registry[camera_id] = orch
    return orch


def get_stream_consumer(camera_id: str) -> "StreamConsumer | None":
    """获取指定摄像头的拉流消费器。"""
    return consumer_registry.get(camera_id)


async def maybe_release_orchestrator(camera_id: str) -> None:
    """无 WebSocket 连接且无拉流消费器时回收 orchestrator。"""
    if ws_client_counts.get(camera_id, 0) > 0:
        return
    if camera_id in consumer_registry:
        return
    orch = camera_registry.pop(camera_id, None)
    if orch is not None:
        await orch.shutdown()
        logger.info("orchestrator 已回收: camera={}", camera_id)


# ------------------------------------------------------------------
# Viewer 广播
# ------------------------------------------------------------------

def register_viewer(camera_id: str) -> asyncio.Queue:
    """注册一个观看端, 返回其专属队列 (maxsize=2, 满时丢最旧帧)。"""
    q: asyncio.Queue = asyncio.Queue(maxsize=2)
    viewer_queues.setdefault(camera_id, set()).add(q)
    return q


def unregister_viewer(camera_id: str, q: asyncio.Queue) -> None:
    """注销观看端队列。"""
    queues = viewer_queues.get(camera_id)
    if queues is not None:
        queues.discard(q)
        if not queues:
            del viewer_queues[camera_id]


def publish_to_viewers(camera_id: str, item: dict) -> None:
    """向该摄像头的所有观看端广播 (队列满时丢最旧, 保证最新帧优先)。"""
    for q in list(viewer_queues.get(camera_id, ())):
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


def viewer_count(camera_id: str) -> int:
    """当前观看端数量。"""
    return len(viewer_queues.get(camera_id, ()))
