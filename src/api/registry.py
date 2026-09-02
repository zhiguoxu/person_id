"""
全局摄像头注册表

独立模块，避免 __main__ vs src.main 双实例问题。
所有模块统一从此处导入 camera_registry。

除 orchestrator 外还维护三类状态:
- consumer_registry: 服务端拉流消费器 (camera_id → StreamConsumer)
- viewer_queues: 观看端广播队列 (camera_id → 每个 WebSocket 连接一个 Queue)
- ws_client_counts: 每个摄像头当前的 WebSocket 连接数 (决定 orchestrator 何时可回收)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from voice_agent_common.utils.logger import logger

from src.pipeline.orchestrator import VisionOrchestrator

if TYPE_CHECKING:
    from src.pipeline.stream_consumer import StreamConsumer

# 全局摄像头注册表: camera_id → VisionOrchestrator
camera_registry: dict[str, VisionOrchestrator] = {}

# 服务端拉流消费器: camera_id → StreamConsumer
consumer_registry: dict[str, "StreamConsumer"] = {}

# camera 级生命周期锁: 串行化 orchestrator 创建和 consumer 启停。
# 所有修改 camera_registry / consumer_registry 的调用方都必须持有这把锁,
# 因而一个 camera 在本进程内最多只有一个 orchestrator 和一个 consumer。
_camera_locks: dict[str, asyncio.Lock] = {}

# 观看端广播队列: camera_id → set[asyncio.Queue]
viewer_queues: dict[str, set[asyncio.Queue]] = {}

# 每个摄像头的活跃 WebSocket 连接数
ws_client_counts: dict[str, int] = {}


def camera_lock(camera_id: str) -> asyncio.Lock:
    """返回 camera 生命周期锁。锁对象按 camera 复用, 不跨进程共享。"""
    return _camera_locks.setdefault(camera_id, asyncio.Lock())


@asynccontextmanager
async def camera_operation(camera_id: str) -> AsyncIterator[None]:
    """串行化一个 camera 的生命周期操作。"""
    async with camera_lock(camera_id):
        yield


# orchestrator

def get_camera_orchestrator(camera_id: str) -> VisionOrchestrator | None:
    """获取指定摄像头的编排器（供 REST routes 使用）。"""
    return camera_registry.get(camera_id)


async def get_or_create_orchestrator(camera_id: str) -> VisionOrchestrator:
    """获取或创建指定摄像头的编排器并注册。"""
    async with camera_lock(camera_id):
        return await get_or_create_orchestrator_unlocked(camera_id)


async def get_or_create_orchestrator_unlocked(camera_id: str) -> VisionOrchestrator:
    """获取或创建编排器。调用方须已持有 camera_lock。"""
    orch = camera_registry.get(camera_id)
    if orch is None:
        orch = await VisionOrchestrator.create(camera_id=camera_id)
        camera_registry[camera_id] = orch
    return orch


async def maybe_release_orchestrator_unlocked(camera_id: str) -> None:
    """无 WebSocket 连接且无拉流消费器时回收 orchestrator。调用方须已持有 camera_lock。"""
    if ws_client_counts.get(camera_id, 0) > 0:
        return
    if camera_id in consumer_registry:
        return
    orch = camera_registry.pop(camera_id, None)
    if orch is not None:
        await orch.shutdown()
        logger.info("orchestrator 已回收: camera={}", camera_id)


async def maybe_release_orchestrator(camera_id: str) -> None:
    """无 WebSocket 连接且无拉流消费器时回收 orchestrator。"""
    async with camera_lock(camera_id):
        await maybe_release_orchestrator_unlocked(camera_id)


# consumer

def get_stream_consumer(camera_id: str) -> "StreamConsumer | None":
    """获取指定摄像头的拉流消费器。"""
    return consumer_registry.get(camera_id)


def register_stream_consumer(camera_id: str, consumer: "StreamConsumer") -> None:
    """登记 consumer; 调用方须已持有 camera_lock。"""
    consumer_registry[camera_id] = consumer


def unregister_stream_consumer(camera_id: str, consumer: "StreamConsumer") -> None:
    """移除指定 consumer, 不误删同 camera 后续注册的新实例。"""
    if consumer_registry.get(camera_id) is consumer:
        consumer_registry.pop(camera_id, None)


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
