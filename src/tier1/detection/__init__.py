"""
检测模块

提供人体检测和姿态分类:
- YoloPoseDetector: YOLO11 Pose 人体检测器 (双模式)
- classify_pose: 姿态朝向分类
- has_visible_face: 正面人脸可见性判断
- get_fast_detector / get_heavy_detector: 单例获取器
"""

from __future__ import annotations

from functools import cache

from src.tier1.detection.pose_classifier import classify_pose, has_visible_face
from src.tier1.detection.yolo_pose import YoloPoseDetector

from src.config import get_config, MODELS_DIR


@cache
def get_fast_detector() -> YoloPoseDetector:
    """获取 Tier1 轻量检测器（单例）。"""
    det_cfg = get_config().detection
    model_path = str(MODELS_DIR / det_cfg.yolo_fast_model)
    return YoloPoseDetector(model_path=model_path)


@cache
def get_heavy_detector() -> YoloPoseDetector:
    """获取 Tier2 精确检测器（单例）。"""
    det_cfg = get_config().detection
    model_path = str(MODELS_DIR / det_cfg.yolo_heavy_model)
    return YoloPoseDetector(model_path=model_path)


__all__ = [
    "YoloPoseDetector",
    "classify_pose",
    "has_visible_face",
    "get_fast_detector",
    "get_heavy_detector",
]
