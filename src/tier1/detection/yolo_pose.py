"""
YOLO11 Pose 检测器封装

支持双模式检测:
- Tier 1: yolo11n-pose (轻量快速, 用于实时追踪)
- Tier 2: yolo11x-pose (重型精确, 用于身份确认)

从原始帧中提取人体检测框、17 个 COCO 关键点和置信度,
转换为系统统一的 Detection 数据结构。
"""
from __future__ import annotations


import numpy as np
from loguru import logger

from src.gallery.data_models import Detection
from src.tier1.detection.pose_classifier import classify_pose, has_visible_face
from src.config import get_config


class YoloPoseDetector:
    """YOLO11 Pose 人体检测器。

    封装 ultralytics YOLO 模型，提供全帧检测和 ROI 裁剪检测两种模式。
    自动过滤过小的检测结果并分类姿态朝向。

    Attributes:
        model: 已加载的 YOLO 模型实例。
        device: 推理设备 (e.g. 'cuda:0')。
        conf_thresh: 检测置信度阈值。
        min_person_height: 最小人体像素高度, 过滤噪声。
    """

    def __init__(
        self,
        model_path: str,
    ) -> None:
        """初始化 YOLO Pose 检测器。

        Args:
            model_path: YOLO 模型权重文件路径 (e.g. 'yolo11n-pose.pt')。
        """


        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # Warm up the model on the target device
            cfg = get_config().detection
            self.model(np.zeros((128, 128, 3), dtype=np.uint8), device=cfg.yolo_device, verbose=False)
            logger.info(
                "YoloPoseDetector loaded: model={}, device={}, conf={}",
                model_path, cfg.yolo_device, cfg.yolo_confidence,
            )
        except Exception as e:
            logger.error("Failed to load YOLO model '{}': {}", model_path, e)
            raise

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """在完整帧上执行人体姿态检测。

        Args:
            frame: BGR 格式图像, shape (H, W, 3)。

        Returns:
            检测结果列表, 按置信度降序排列。
            过滤掉高度小于 min_person_height 的检测。
        """
        cfg = get_config().detection
        try:
            results = self.model(
                frame,
                device=cfg.yolo_device,
                conf=cfg.yolo_confidence,
                iou=cfg.yolo_iou_threshold,
                max_det=cfg.yolo_max_det,
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO inference failed: {}", e)
            return []

        return self._parse_results(results)


    @staticmethod
    def _parse_results(
        results: list,
    ) -> list[Detection]:
        """解析 YOLO 推理结果为 Detection 列表。

        Args:
            results: ultralytics 模型推理结果。

        Returns:
            排序后的 Detection 列表。
        """
        detections: list[Detection] = []

        if not results or len(results) == 0:
            return detections

        result = results[0]

        # Ensure we have boxes and keypoints
        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes
        keypoints_data = result.keypoints

        n_detections = len(boxes)

        for i in range(n_detections):
            try:
                # Extract bounding box (xyxy format)
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)

                # Filter by minimum person height
                person_height = bbox[3] - bbox[1]
                if person_height < get_config().detection.min_person_height_px:
                    continue

                # Detection confidence
                conf = float(boxes.conf[i].cpu().numpy())

                # Extract 17 COCO keypoints: (17, 3) -> x, y, confidence
                # ultralytics stores keypoints as (17, 3) with xy and conf
                kpts_xy = keypoints_data.xy[i].cpu().numpy()  # (17, 2)
                kpts_conf = keypoints_data.conf[i].cpu().numpy()  # (17,)

                keypoints = np.zeros((17, 3), dtype=np.float32)
                keypoints[:, :2] = kpts_xy
                keypoints[:, 2] = kpts_conf

                # Classify pose direction
                pose = classify_pose(keypoints)
                face_visible = has_visible_face(keypoints)

                detection = Detection(
                    bbox=bbox,
                    confidence=conf,
                    keypoints=keypoints,
                    pose_bucket=pose,
                    has_face=face_visible,
                )
                detections.append(detection)

            except Exception as e:
                logger.warning("Failed to parse detection {}: {}", i, e)
                continue

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections
