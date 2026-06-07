"""
复合人脸质量评估器

无需额外模型, 基于图像处理和几何分析评估人脸质量。
五个评估维度:
1. 模糊度 (blur_score): Laplacian 方差
2. 尺寸 (size_score): 人脸像素尺寸
3. 关键点置信度 (landmark_conf): 平均关键点置信度
4. 姿态 (pose_score): 基于关键点/人脸关键点几何估计
5. 光照 (lighting_score): 灰度直方图标准差

加权求和输出综合质量分 [0, 1]。
"""
from __future__ import annotations


import cv2
import numpy as np
from loguru import logger

from src.config import get_config


class FaceQualityAssessor:
    """复合人脸质量评估器。

    对人脸裁剪图评估多维度质量, 用于决定特征是否满足入库门槛,
    以及在匹配时进行质量加权。

    Attributes:
        config: 人脸配置 (包含各维度权重)。
    """

    # Laplacian blur score normalization: scores above this threshold
    # are considered sharp (score = 1.0)
    _BLUR_SATURATE = 500.0

    # Lighting score: ideal std range for face grayscale histogram
    _LIGHTING_IDEAL_STD = 60.0

    def __init__(self) -> None:
        """初始化质量评估器。"""
        config = get_config().face
        logger.debug(
            "QualityAssessor initialized: weights=[blur={}, size={}, "
            "landmark={}, pose={}, lighting={}]",
            config.quality_blur_weight,
            config.quality_size_weight,
            config.quality_landmark_weight,
            config.quality_pose_weight,
            config.quality_lighting_weight,
        )

    def assess(
        self,
        face_crop: np.ndarray,
        landmarks: np.ndarray,
        face_bbox: np.ndarray,
        keypoints: np.ndarray | None = None,
    ) -> float:
        """综合评估人脸质量。

        Args:
            face_crop: 人脸裁剪图, BGR 格式。
            landmarks: 5 点人脸关键点, shape (5, 2)。
            face_bbox: 人脸检测框 (x1, y1, x2, y2)。
            keypoints: 可选的 COCO 17 关键点 (17, 3),
                       用于更精确的姿态评估。

        Returns:
            综合质量分 [0, 1]。
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        try:
            blur = self._blur_score(face_crop)
            size = self._size_score(face_bbox)
            landmark = self._landmark_conf_score(landmarks)
            pose = self._pose_score(landmarks, keypoints)
            lighting = self._lighting_score(face_crop)

            # Weighted sum
            cfg = get_config().face
            quality = (
                cfg.quality_blur_weight * blur
                + cfg.quality_size_weight * size
                + cfg.quality_landmark_weight * landmark
                + cfg.quality_pose_weight * pose
                + cfg.quality_lighting_weight * lighting
            )

            return float(np.clip(quality, 0.0, 1.0))

        except Exception as e:
            logger.debug("Quality assessment failed: {}", e)
            return 0.0

    def _blur_score(self, face_crop: np.ndarray) -> float:
        """模糊度评分 — 基于 Laplacian 方差。

        Laplacian 方差越高, 图像越清晰。

        Args:
            face_crop: 人脸裁剪图, BGR 格式。

        Returns:
            清晰度分数 [0, 1]。
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize: saturate at _BLUR_SATURATE
        score = min(1.0, laplacian_var / self._BLUR_SATURATE)
        return float(score)

    def _size_score(self, face_bbox: np.ndarray) -> float:
        """尺寸评分 — 人脸像素尺寸相对于最小要求。

        Args:
            face_bbox: 人脸检测框 (x1, y1, x2, y2)。

        Returns:
            尺寸分数 [0, 1]。
        """
        face_w = float(face_bbox[2] - face_bbox[0])
        face_h = float(face_bbox[3] - face_bbox[1])
        face_size = min(face_w, face_h)

        min_size = float(get_config().face.min_face_size)
        if min_size <= 0:
            return 1.0

        # Score: min_size → 0.5, 2*min_size → 1.0
        score = face_size / (2.0 * min_size)
        return float(np.clip(score, 0.0, 1.0))

    def _landmark_conf_score(self, landmarks: np.ndarray) -> float:
        """关键点置信度评分。

        对于 InsightFace 5 点关键点, 没有独立的置信度值,
        因此使用关键点坐标的合理性来估计。

        如果 landmarks 含有第 3 列 (置信度), 则直接取平均。

        Args:
            landmarks: 人脸关键点, shape (5, 2) 或 (5, 3)。

        Returns:
            关键点置信度分数 [0, 1]。
        """
        if landmarks is None or landmarks.size == 0:
            return 0.0

        if landmarks.shape[-1] >= 3:
            # Has confidence column
            conf = landmarks[:, 2]
            return float(np.clip(np.mean(conf), 0.0, 1.0))

        # For InsightFace (no per-landmark confidence), evaluate landmark
        # consistency: inter-eye distance should be reasonable
        if landmarks.shape[0] >= 2:
            eye_dist = np.linalg.norm(landmarks[0] - landmarks[1])
            # Reasonable inter-eye distance: 15-200 pixels
            if 15 < eye_dist < 200:
                return 0.8
            elif 5 < eye_dist < 300:
                return 0.5
            else:
                return 0.2

        return 0.5

    def _pose_score(
        self,
        landmarks: np.ndarray,
        keypoints: np.ndarray | None = None,
    ) -> float:
        """姿态评分 — 正面人脸得分高, 侧面得分低。

        使用人脸 5 点关键点的对称性估计姿态角。
        InsightFace 5 点: [left_eye, right_eye, nose, left_mouth, right_mouth]

        如果提供 COCO 关键点, 可进一步利用耳朵可见性判断。

        Args:
            landmarks: 5 点人脸关键点, shape (5, 2)。
            keypoints: 可选的 COCO 17 关键点 (17, 3)。

        Returns:
            姿态分数 [0, 1], 正面 → 高分。
        """
        score = 0.5  # Default neutral score

        if landmarks is not None and landmarks.shape[0] >= 5:
            # InsightFace 5-point: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye = landmarks[0, :2]
            right_eye = landmarks[1, :2]
            nose = landmarks[2, :2]

            # Eye midpoint
            eye_mid = (left_eye + right_eye) / 2.0
            eye_dist = np.linalg.norm(left_eye - right_eye)

            if eye_dist > 1e-3:
                # Nose deviation from eye midpoint center → yaw indicator
                nose_offset = abs(nose[0] - eye_mid[0])
                nose_ratio = nose_offset / eye_dist

                # Perfect frontal: nose_ratio ≈ 0
                # Profile: nose_ratio > 0.5
                yaw_score = max(0.0, 1.0 - nose_ratio * 2.0)

                # Vertical: nose should be below eyes
                vertical_dist = nose[1] - eye_mid[1]
                if vertical_dist > 0:
                    pitch_score = min(1.0, vertical_dist / (eye_dist * 0.8))
                else:
                    pitch_score = 0.3  # Looking up

                score = 0.7 * yaw_score + 0.3 * pitch_score

        # Supplement with COCO keypoints if available
        if keypoints is not None and keypoints.shape[0] >= 5:
            # Check ear visibility as side-view indicator
            left_ear_vis = float(keypoints[3, 2]) > 0.3
            right_ear_vis = float(keypoints[4, 2]) > 0.3
            nose_vis = float(keypoints[0, 2]) > 0.3

            if nose_vis and not left_ear_vis and not right_ear_vis:
                # Pure frontal — boost score
                score = max(score, 0.9)
            elif not nose_vis:
                # Back view — penalize
                score = min(score, 0.2)

        return float(np.clip(score, 0.0, 1.0))

    def _lighting_score(self, face_crop: np.ndarray) -> float:
        """光照评分 — 基于灰度直方图标准差。

        标准差过低 → 光照不均或过暗/过亮。
        标准差适中 → 光照条件良好。

        Args:
            face_crop: 人脸裁剪图, BGR 格式。

        Returns:
            光照分数 [0, 1]。
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        mean_val = float(np.mean(gray))

        # Penalize extreme brightness
        brightness_score = 1.0
        if mean_val < 40:
            brightness_score = mean_val / 40.0
        elif mean_val > 220:
            brightness_score = (255.0 - mean_val) / 35.0

        # Std-based score: ideal around 50-70
        std_score = min(1.0, std / self._LIGHTING_IDEAL_STD)

        # Combine
        score = 0.6 * std_score + 0.4 * brightness_score

        return float(np.clip(score, 0.0, 1.0))
