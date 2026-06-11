"""
质量评估工具

提供 Tier1 轻量质量预估和 Tier2 图像清晰度计算:
- compute_quality_hint: ~0.1ms 纯数值, 用于 RecentBuffer 窗口内竞争
- compute_sharpness: Laplacian 方差, 用于 Tier2 精确体质量评估
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_quality_hint(
    bbox: np.ndarray,
    keypoints: np.ndarray,
    frame_shape: tuple[int, int],
    conf_thresh: float = 0.3,
) -> float:
    """Tier1 轻量质量预估 — ~0.1ms 纯数值计算
    
    综合 5 维信号:
    1. 分辨率 (bbox 面积占比)
    2. 完整度 (头/侧/脚 分段惩罚)
    3. 身体可见性 (加权关键点)
    4. 姿态正面性 (面部关键点连续置信度)
    5. 宽高比 (惩罚极度扭曲的框)
    
    Args:
        bbox: [x1, y1, x2, y2] 像素坐标
        keypoints: (17, 3) COCO keypoints
        frame_shape: (H, W) 帧尺寸
        conf_thresh: 关键点置信度阈值 (unused, kept for API compat)
    
    Returns:
        quality_hint: float in [0, 1]
    """
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    frame_h, frame_w = frame_shape[:2]
    
    # 1. 分辨率 (0-1): bbox 面积占比, 占画面5%即满分
    area_ratio = (w * h) / max(frame_h * frame_w, 1)
    resolution = min(1.0, area_ratio / 0.05)
    
    # 2. 完整度 (0-1): 头部截断致命(重扣), 底部常见(轻扣)
    top_margin = y1 / max(frame_h, 1)
    bottom_margin = (frame_h - y2) / max(frame_h, 1)
    lr_margin = min(x1 / max(frame_w, 1), (frame_w - x2) / max(frame_w, 1))
    
    head_score = 1.0 if top_margin > 0.02 else (0.1 + 0.9 * top_margin / 0.02)
    side_score = 1.0 if lr_margin > 0.02 else (0.5 + 0.5 * lr_margin / 0.02)
    foot_score = 1.0 if bottom_margin > 0.02 else (0.8 + 0.2 * bottom_margin / 0.02)
    completeness = head_score * side_score * foot_score
    
    # 3. 身体可见性 (0-1): 躯干+四肢加权关键点
    if keypoints is not None and len(keypoints) >= 17:
        body_confs = keypoints[5:, 2]
        # 躯干(肩髋)权重高, 四肢边缘(手腕脚踝)权重低
        weights = np.array([1.5, 1.5, 1.0, 1.0, 0.5, 0.5,
                            1.5, 1.5, 1.0, 1.0, 0.5, 0.5])
        body_visibility = float(np.sum(body_confs * weights) / np.sum(weights))
    else:
        body_visibility = 0.0
    
    # 4. 姿态正面性 (0-1): 面部关键点连续置信度
    if keypoints is not None and len(keypoints) >= 5:
        nose = keypoints[0, 2]
        l_eye, r_eye = keypoints[1, 2], keypoints[2, 2]
        l_ear, r_ear = keypoints[3, 2], keypoints[4, 2]
        frontality = (0.4 * nose +
                      0.2 * l_eye + 0.2 * r_eye +
                      0.1 * l_ear + 0.1 * r_ear)
    else:
        frontality = 0.0
    
    # 5. 宽高比 (0-1): 惩罚极度扭曲的框
    aspect = w / max(h, 1)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.4) / 0.4)
    
    # 加权组合
    quality = (0.25 * resolution +
               0.15 * completeness +
               0.20 * body_visibility +
               0.25 * frontality +
               0.15 * aspect_score)
    return float(np.clip(quality, 0.0, 1.0))


def compute_sharpness(crop: np.ndarray) -> float:
    """计算图像清晰度 — Laplacian 方差
    
    Args:
        crop: BGR 人体裁剪
    
    Returns:
        sharpness: float in [0, 1], 归一化后的清晰度分数
    """
    if crop is None or crop.size == 0:
        return 0.0
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    # Laplacian 方差 — 值越高越清晰
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 归一化: 典型范围 [0, 300], 映射到 [0, 1]
    sharpness = min(lap_var / 300.0, 1.0)
    return float(sharpness)


def compute_blur_score(face_crop: np.ndarray) -> float:
    """人脸模糊度评分 — 基于 Laplacian 方差

    Laplacian 方差越高, 图像越清晰。
    用于补充 eDifFIQA 对模糊的短板。

    Args:
        face_crop: 人脸裁剪图, BGR 格式。

    Returns:
        清晰度分数 [0, 1]。
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: saturate at 500
    score = min(1.0, laplacian_var / 500.0)
    return float(score)
