"""
全局摄像头注册表

独立模块，避免 __main__ vs src.api.server 双实例问题。
所有模块统一从此处导入 camera_registry。
"""
from __future__ import annotations

from src.pipeline.orchestrator import VisionOrchestrator

# 全局摄像头注册表: camera_id → VisionOrchestrator
camera_registry: dict[str, VisionOrchestrator] = {}


def get_camera_orchestrator(camera_id: str) -> VisionOrchestrator | None:
    """获取指定摄像头的编排器（供 REST routes 使用）。"""
    return camera_registry.get(camera_id)
