"""
姿态朝向分类器

基于 COCO 17 关键点的可见性和空间关系, 判断人体朝向。
用于人脸特征的姿态分桶存储, 确保同一姿态下的特征可以有效比较。

关键点索引 (COCO 17):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""
from __future__ import annotations

import numpy as np

from src.gallery.data_models import PoseBucket


# COCO keypoint indices
_NOSE = 0
_LEFT_EYE = 1
_RIGHT_EYE = 2
_LEFT_EAR = 3
_RIGHT_EAR = 4
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6


def classify_pose(
    keypoints: np.ndarray,
    conf_thresh: float = 0.3,
) -> PoseBucket:
    """根据关键点可见性分类人体朝向。

    分类逻辑:
    1. FRONTAL: 鼻子 + 至少一只眼睛可见 (正面)
    2. LEFT:    右耳可见、左耳不可见 (人的左侧朝向相机，右耳朝向相机)
    3. RIGHT:   左耳可见、右耳不可见 (人的右侧朝向相机，左耳朝向相机)
    4. BACK:    双耳可见但鼻子和眼睛均不可见 (背面)
    5. UNKNOWN: 关键点不足以判断

    Args:
        keypoints: 关键点数组, shape (17, 3), 每行 [x, y, confidence]。
        conf_thresh: 关键点可见性的置信度阈值。

    Returns:
        PoseBucket 枚举值。
    """
    if keypoints is None or keypoints.shape[0] < 5:
        return PoseBucket.UNKNOWN

    def _visible(idx: int) -> bool:
        return float(keypoints[idx, 2]) >= conf_thresh

    nose_vis = _visible(_NOSE)
    left_eye_vis = _visible(_LEFT_EYE)
    right_eye_vis = _visible(_RIGHT_EYE)
    left_ear_vis = _visible(_LEFT_EAR)
    right_ear_vis = _visible(_RIGHT_EAR)

    any_eye = left_eye_vis or right_eye_vis

    # --- Frontal: nose visible + at least one eye ---
    if nose_vis and any_eye:
        return PoseBucket.FRONTAL

    # --- Back: both ears visible, no nose, no eyes ---
    if left_ear_vis and right_ear_vis and not nose_vis and not any_eye:
        return PoseBucket.BACK

    # --- Side views ---
    # When a person faces left (their left side toward camera):
    #   right ear visible (facing camera), left ear hidden
    # When a person faces right (their right side toward camera):
    #   left ear visible (facing camera), right ear hidden

    # Use nose + ear combination for partial side views
    if nose_vis and not any_eye:
        # Nose visible but no eyes → near-profile
        if right_ear_vis and not left_ear_vis:
            return PoseBucket.LEFT
        if left_ear_vis and not right_ear_vis:
            return PoseBucket.RIGHT
        # Both ears or neither, with just a nose → ambiguous frontal/back
        if right_ear_vis and left_ear_vis:
            return PoseBucket.FRONTAL  # likely frontal with low-conf eyes

    # Pure ear-based classification (no nose)
    if right_ear_vis and not left_ear_vis:
        return PoseBucket.LEFT
    if left_ear_vis and not right_ear_vis:
        return PoseBucket.RIGHT

    # --- Shoulder-based fallback for side detection ---
    left_shoulder_vis = _visible(_LEFT_SHOULDER)
    right_shoulder_vis = _visible(_RIGHT_SHOULDER)

    if left_shoulder_vis and right_shoulder_vis:
        # Both shoulders visible — check relative position for body twist
        shoulder_diff = abs(
            float(keypoints[_LEFT_SHOULDER, 0]) - float(keypoints[_RIGHT_SHOULDER, 0])
        )
        shoulder_width_norm = shoulder_diff / max(
            abs(float(keypoints[_LEFT_SHOULDER, 1]) - float(keypoints[_RIGHT_SHOULDER, 1])) + 1e-6,
            1.0,
        )
        # Very narrow shoulders → side view, but we can't determine direction
        # without other cues
        if shoulder_width_norm < 0.3:
            # Try to use any remaining facial cues
            if any_eye:
                if left_eye_vis and not right_eye_vis:
                    return PoseBucket.RIGHT
                if right_eye_vis and not left_eye_vis:
                    return PoseBucket.LEFT

    return PoseBucket.UNKNOWN


def has_visible_face(
    keypoints: np.ndarray,
    conf_thresh: float = 0.5,
) -> bool:
    """判断是否有可辨识的正面人脸。

    要求鼻子和至少一只眼睛以较高置信度可见。
    用于决定是否运行人脸识别。

    Args:
        keypoints: 关键点数组, shape (17, 3)。
        conf_thresh: 可见性判断的置信度阈值 (比姿态分类更严格)。

    Returns:
        True 如果检测到正面人脸。
    """
    if keypoints is None or keypoints.shape[0] < 3:
        return False

    nose_vis = float(keypoints[_NOSE, 2]) >= conf_thresh
    left_eye_vis = float(keypoints[_LEFT_EYE, 2]) >= conf_thresh
    right_eye_vis = float(keypoints[_RIGHT_EYE, 2]) >= conf_thresh

    return nose_vis and (left_eye_vis or right_eye_vis)
