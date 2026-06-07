"""
图像处理通用工具类。
"""
from __future__ import annotations

import numpy as np


def crop_image(
    image: np.ndarray,
    bbox: np.ndarray | tuple[int, int, int, int] | list[int],
    min_width: int = 10,
    min_height: int = 10,
) -> np.ndarray | None:
    """带边界保护的安全裁剪函数。

    如果裁剪出来的区域小于最小尺寸, 则返回 None。

    Args:
        image: 原始图像, shape (H, W, C)。
        bbox: 边界框 (x1, y1, x2, y2)。
        min_width: 允许的最小宽度。
        min_height: 允许的最小高度。

    Returns:
        裁剪出的图像副本 (深拷贝), 无效或过小时返回 None。
    """
    h, w = image.shape[:2]

    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    if x2 - x1 < min_width or y2 - y1 < min_height:
        return None

    return image[y1:y2, x1:x2].copy()
