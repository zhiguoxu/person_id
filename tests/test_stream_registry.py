"""Regression tests for per-camera stream lifecycle serialization."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

# Importing the full orchestrator eagerly loads GPU/config dependencies. Keep the
# lifecycle tests focused by providing only the class needed by registry.py.
_had_pipeline = "src.pipeline" in sys.modules
_had_orchestrator = "src.pipeline.orchestrator" in sys.modules
if not _had_pipeline:
    _pipeline = types.ModuleType("src.pipeline")
    _pipeline.__path__ = [str(Path(__file__).parents[1] / "src" / "pipeline")]
    sys.modules["src.pipeline"] = _pipeline
if not _had_orchestrator:
    _orchestrator = types.ModuleType("src.pipeline.orchestrator")
    _orchestrator.VisionOrchestrator = object
    sys.modules["src.pipeline.orchestrator"] = _orchestrator

from src.api import registry  # noqa: E402


pytestmark = pytest.mark.asyncio


def teardown_module():
    if not _had_orchestrator:
        sys.modules.pop("src.pipeline.orchestrator", None)
    if not _had_pipeline:
        sys.modules.pop("src.pipeline.stream_state", None)
        sys.modules.pop("src.pipeline", None)


class FakeOrchestrator:
    create_calls = 0

    @classmethod
    async def create(cls, camera_id: str):
        cls.create_calls += 1
        # Force concurrent callers to overlap at the await point.
        await asyncio.sleep(0)
        return cls(camera_id)

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.shutdown_calls = 0

    async def shutdown(self):
        self.shutdown_calls += 1


class FakeConsumer:
    def __init__(self, url: str, running: bool = True):
        self.url = url
        self.running = running
        self.stop_calls = 0

    async def stop(self):
        self.stop_calls += 1
        self.running = False
        registry.unregister_stream_consumer("camera-1", self)


@pytest.fixture(autouse=True)
def clean_registry():
    registry.camera_registry.clear()
    registry.consumer_registry.clear()
    registry._camera_locks.clear()
    registry.ws_client_counts.clear()
    yield
    registry.camera_registry.clear()
    registry.consumer_registry.clear()
    registry._camera_locks.clear()
    registry.ws_client_counts.clear()


async def test_get_or_create_is_singleton_per_camera(monkeypatch):
    FakeOrchestrator.create_calls = 0
    monkeypatch.setattr(registry, "VisionOrchestrator", FakeOrchestrator)

    result = await asyncio.gather(*(
        registry.get_or_create_orchestrator("camera-1") for _ in range(20)
    ))

    assert FakeOrchestrator.create_calls == 1
    assert len({id(item) for item in result}) == 1


async def test_register_replaces_single_consumer_slot():
    old = FakeConsumer("old")
    new = FakeConsumer("new")

    async with registry.camera_operation("camera-1"):
        registry.register_stream_consumer("camera-1", old)
        registry.register_stream_consumer("camera-1", new)

    assert registry.get_stream_consumer("camera-1") is new


async def test_stop_consumption_stops_registered_consumer(monkeypatch):
    from src.pipeline import stream_state

    class FakeRedis:
        def __init__(self):
            self.deleted = []

        async def hdel(self, key, camera_id):
            self.deleted.append((key, camera_id))

    redis = FakeRedis()
    monkeypatch.setattr(stream_state, "_get_client", lambda: redis)

    orch = FakeOrchestrator("camera-1")
    consumer = FakeConsumer("same-url")
    registry.camera_registry["camera-1"] = orch
    async with registry.camera_operation("camera-1"):
        registry.register_stream_consumer("camera-1", consumer)

    returned = await stream_state.stop_consumption("camera-1")

    assert returned is consumer
    assert consumer.stop_calls == 1
    assert registry.get_stream_consumer("camera-1") is None
    assert redis.deleted == [(stream_state.STATE_KEY, "camera-1")]
    assert orch.shutdown_calls == 1
