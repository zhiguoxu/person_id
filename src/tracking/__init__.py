"""追踪模块 — 多目标追踪与时空约束记忆"""
from src.tracking.engine import TrackingEngine, PersonIdentity
from src.tracking.spatial_memory import SpatialMemory


def create_tracker(config):
    """创建追踪引擎实例。"""
    return TrackingEngine(config)


__all__ = ["TrackingEngine", "PersonIdentity", "SpatialMemory", "create_tracker"]
