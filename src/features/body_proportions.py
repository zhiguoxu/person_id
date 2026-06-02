"""
体型比例特征提取与比较

封装 BodyProportions 数据模型的便捷函数:
- extract_proportions: 从 COCO 17 关键点提取体型比例
- proportions_similarity: 计算两个体型比例的相似度

体型比例是衣服无关的辅助身份信号, 基于骨骼几何计算,
不需要额外模型。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from src.gallery.data_models import BodyProportions


def extract_proportions(keypoints: np.ndarray) -> Optional[BodyProportions]:
    """从 COCO 17 关键点提取体型比例特征。

    封装 BodyProportions.from_keypoints(), 添加输入验证和错误处理。

    Args:
        keypoints: COCO 17 关键点数组, shape (17, 3), 每行 [x, y, confidence]。

    Returns:
        BodyProportions 实例, 如果关键点不足则返回 None。
    """
    if keypoints is None:
        logger.debug("Cannot extract proportions: keypoints is None")
        return None

    if keypoints.shape[0] < 17 or keypoints.shape[1] < 3:
        logger.debug(
            "Cannot extract proportions: invalid keypoints shape {}",
            keypoints.shape,
        )
        return None

    try:
        return BodyProportions.from_keypoints(keypoints)
    except Exception as e:
        logger.debug("Body proportions extraction failed: {}", e)
        return None


def proportions_similarity(a: BodyProportions, b: BodyProportions) -> float:
    """计算两个体型比例的相似度。

    封装 BodyProportions.similarity(), 使用高斯核:
    exp(-||a - b||² / (2σ²)), σ=0.15

    Args:
        a: 第一个体型比例。
        b: 第二个体型比例。

    Returns:
        相似度分数 [0, 1], 1 表示完全相同。
    """
    try:
        return BodyProportions.similarity(a, b)
    except Exception as e:
        logger.debug("Body proportions similarity failed: {}", e)
        return 0.0
