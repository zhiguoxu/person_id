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

# Laplacian blur score normalization: scores above this threshold
# are considered sharp (score = 1.0)
_BLUR_SATURATE = 500.0

# Lighting score: ideal std range for face grayscale histogram
_LIGHTING_IDEAL_STD = 60.0


class FaceQualityAssessor:
    """复合人脸质量评估器。

    对人脸裁剪图评估多维度质量, 用于决定特征是否满足入库门槛,
    以及在匹配时进行质量加权。

    所有方法均为 staticmethod, 无实例状态。
    """

    @staticmethod
    def assess(
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
            blur = FaceQualityAssessor._blur_score(face_crop)
            size = FaceQualityAssessor._size_score(face_bbox)
            landmark = FaceQualityAssessor._landmark_conf_score(landmarks)
            pose = FaceQualityAssessor._pose_score(landmarks, keypoints)
            lighting = FaceQualityAssessor._lighting_score(face_crop)

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

    @staticmethod
    def _blur_score(face_crop: np.ndarray) -> float:
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
        score = min(1.0, laplacian_var / _BLUR_SATURATE)
        return float(score)

    @staticmethod
    def _size_score(face_bbox: np.ndarray) -> float:
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

    @staticmethod
    def _landmark_conf_score(landmarks: np.ndarray) -> float:
        """关键点可信度评分。

        对于 InsightFace 5 点关键点, 没有独立的置信度值,
        因此使用关键点坐标的几何合理性和对称性来估计。

        所有判断均使用比值, 不依赖绝对像素, 适应远近不同的人脸尺寸。

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

        # For InsightFace (no per-landmark confidence)
        if landmarks.shape[0] >= 5:
            left_eye = landmarks[0, :2]
            right_eye = landmarks[1, :2]
            nose = landmarks[2, :2]
            left_mouth = landmarks[3, :2]
            right_mouth = landmarks[4, :2]

            # 用五点 bounding box 的对角线作为尺度参考 (远近无关)
            all_pts = landmarks[:5, :2]
            face_span = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
            if face_span < 1e-3:
                return 0.1  # 五点几乎重叠 → 退化

            eye_dist = np.linalg.norm(left_eye - right_eye)
            eye_ratio = eye_dist / face_span
            # 正常人脸: eye_ratio ≈ 0.35–0.55
            # 双眼重合 (严重侧面): eye_ratio → 0
            if eye_ratio < 0.05:
                return 0.1  # 双眼几乎重合
            elif 0.2 < eye_ratio < 0.7:
                base = 0.8
            elif 0.05 < eye_ratio < 0.8:
                base = 0.5
            else:
                return 0.2  # 异常布局

            # 对称性检查: 鼻子到左右眼的距离比值
            # 正面人脸 nose 在双眼中间, 侧面/遮挡时严重偏移
            d_left = np.linalg.norm(nose - left_eye)
            d_right = np.linalg.norm(nose - right_eye)
            ratio = min(d_left, d_right) / max(d_left, d_right, 1e-6)
            symmetry_score = np.clip(ratio, 0.0, 1.0)

            # 嘴角对称性 (辅助判断)
            dm_left = np.linalg.norm(nose - left_mouth)
            dm_right = np.linalg.norm(nose - right_mouth)
            m_ratio = min(dm_left, dm_right) / max(dm_left, dm_right, 1e-6)
            mouth_symmetry = np.clip(m_ratio, 0.0, 1.0)

            # 综合: 对称性差时大幅惩罚
            sym = 0.7 * symmetry_score + 0.3 * mouth_symmetry
            return float(base * (0.3 + 0.7 * sym))

        return 0.5

    @staticmethod
    def _pose_score(
            landmarks: np.ndarray,
            keypoints: np.ndarray | None = None,
    ) -> float:
        """姿态评分 — 正面人脸得分高, 侧面得分低。

        使用人脸 5 点关键点的对称性估计姿态角。
        InsightFace 5 点: [left_eye, right_eye, nose, left_mouth, right_mouth]

        如果提供 COCO 关键点, 利用眼耳可见性做更精确判断。

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
            # COCO: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
            nose_vis = float(keypoints[0, 2]) > 0.3
            left_eye_vis = float(keypoints[1, 2]) > 0.3
            right_eye_vis = float(keypoints[2, 2]) > 0.3
            left_ear_vis = float(keypoints[3, 2]) > 0.3
            right_ear_vis = float(keypoints[4, 2]) > 0.3

            if not nose_vis:
                # 背面 → 严重惩罚
                score = min(score, 0.2)
            elif nose_vis and not left_ear_vis and not right_ear_vis:
                # 双耳不可见 = 纯正面 → boost
                score = max(score, 0.9)
            elif left_ear_vis != right_ear_vis:
                # 只有一只耳朵可见 = 侧面 → 限制上限
                score = min(score, 0.5)

            # 只有一只眼可见 → 半脸, 重罚
            if left_eye_vis != right_eye_vis:
                score = min(score, 0.3)
            elif not left_eye_vis and not right_eye_vis:
                # 双眼都不可见
                score = min(score, 0.15)

        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _lighting_score(face_crop: np.ndarray) -> float:
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
        std_score = min(1.0, std / _LIGHTING_IDEAL_STD)

        # Combine
        score = 0.6 * std_score + 0.4 * brightness_score

        return float(np.clip(score, 0.0, 1.0))
