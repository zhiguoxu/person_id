"""
YOLO11 Pose 检测器封装

支持双模式检测:
- Tier 1: yolo11n-pose (轻量快速, 用于实时追踪)
- Tier 2: yolo11x-pose (重型精确, 用于身份确认)

从原始帧中提取人体检测框、17 个 COCO 关键点和置信度,
转换为系统统一的 Detection 数据结构。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.gallery.data_models import Detection, PoseBucket
from src.detection.pose_classifier import classify_pose, has_visible_face


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
        device: str = "cuda:0",
        conf_thresh: float = 0.5,
        min_person_height: int = 80,
        iou_threshold: float = 0.7,
        max_det: int = 20,
    ) -> None:
        """初始化 YOLO Pose 检测器。

        Args:
            model_path: YOLO 模型权重文件路径 (e.g. 'yolo11n-pose.pt')。
            device: 推理设备。
            conf_thresh: 检测置信度阈值。
            min_person_height: 最小人体像素高度过滤。
            iou_threshold: NMS IoU 阈值。
            max_det: 最大检测数量。
        """
        self.device = device
        self.conf_thresh = conf_thresh
        self.min_person_height = min_person_height
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            # Warm up the model on the target device
            logger.info(
                "YoloPoseDetector loaded: model={}, device={}, conf={}",
                model_path, device, conf_thresh,
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
        try:
            results = self.model(
                frame,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO inference failed: {}", e)
            return []

        return self._parse_results(results)

    def detect_single(
        self,
        frame: np.ndarray,
        roi_bbox: np.ndarray,
    ) -> Optional[Detection]:
        """在指定 ROI 区域上执行检测, 返回最高置信度的结果。

        将 ROI 从原始帧中裁剪出来运行检测, 然后将检测坐标映射回
        原始帧坐标系。适用于 Tier 2 对特定追踪目标的精确重检测。

        Args:
            frame: BGR 格式完整帧, shape (H, W, 3)。
            roi_bbox: ROI 区域 (x1, y1, x2, y2) 像素坐标。

        Returns:
            最高置信度的检测结果, 如果无检测则返回 None。
        """
        h, w = frame.shape[:2]

        # Clamp ROI to frame boundaries with a small margin
        margin = 20
        x1 = max(0, int(roi_bbox[0]) - margin)
        y1 = max(0, int(roi_bbox[1]) - margin)
        x2 = min(w, int(roi_bbox[2]) + margin)
        y2 = min(h, int(roi_bbox[3]) + margin)

        if x2 - x1 < 10 or y2 - y1 < 10:
            logger.debug("ROI too small: ({}, {}, {}, {})", x1, y1, x2, y2)
            return None

        crop = frame[y1:y2, x1:x2]

        try:
            results = self.model(
                crop,
                device=self.device,
                conf=self.conf_thresh,
                iou=self.iou_threshold,
                max_det=5,
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO ROI inference failed: {}", e)
            return None

        detections = self._parse_results(results, offset_x=x1, offset_y=y1)

        if not detections:
            return None

        # Return the detection with highest confidence
        return max(detections, key=lambda d: d.confidence)

    def _parse_results(
        self,
        results: list,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[Detection]:
        """解析 YOLO 推理结果为 Detection 列表。

        Args:
            results: ultralytics 模型推理结果。
            offset_x: ROI 裁剪时的 x 偏移量 (用于坐标映射)。
            offset_y: ROI 裁剪时的 y 偏移量。

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

                # Apply coordinate offset for ROI detection
                bbox[0] += offset_x
                bbox[1] += offset_y
                bbox[2] += offset_x
                bbox[3] += offset_y

                # Filter by minimum person height
                person_height = bbox[3] - bbox[1]
                if person_height < self.min_person_height:
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

                # Apply coordinate offset to keypoints
                if offset_x != 0 or offset_y != 0:
                    # Only offset visible keypoints (conf > 0)
                    visible_mask = keypoints[:, 2] > 0
                    keypoints[visible_mask, 0] += offset_x
                    keypoints[visible_mask, 1] += offset_y

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
