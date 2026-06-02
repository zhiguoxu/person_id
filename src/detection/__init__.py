"""
检测模块

提供人体检测和姿态分类:
- YoloPoseDetector: YOLO11 Pose 人体检测器 (双模式)
- classify_pose: 姿态朝向分类
- has_visible_face: 正面人脸可见性判断
- create_detector: 工厂函数
"""
from src.detection.pose_classifier import classify_pose, has_visible_face
from src.detection.yolo_pose import YoloPoseDetector


def create_detector(config, tier="fast"):
    """创建检测器实例。

    Args:
        config: 全局配置。
        tier: 'fast' 使用轻量模型, 'heavy' 使用精确模型。
    """
    model_path = (
        config.detection.yolo_fast_model
        if tier == "fast"
        else config.detection.yolo_heavy_model
    )
    return YoloPoseDetector(
        model_path=model_path,
        device=config.detection.yolo_device,
        conf_thresh=config.detection.yolo_confidence,
        min_person_height=config.detection.min_person_height_px,
        iou_threshold=config.detection.yolo_iou_threshold,
        max_det=config.detection.yolo_max_det,
    )


__all__ = [
    "YoloPoseDetector",
    "classify_pose",
    "has_visible_face",
    "create_detector",
]
