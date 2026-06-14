"""
Tier1 人脸检测器

使用 SCRFD_10G (detection-only, 不加载 recognition 模型) 获取人脸 bbox + 5 点关键点,
用于 norm_crop 对齐 → eDifFIQA 质量评估。

与 Tier2 的 FaceExtractor 区分:
- 本模块: 只做检测 + 对齐, 不做 embedding 提取
- FaceExtractor: 只做人脸 embedding 提取 (ArcFace/AdaFace 可切换, 不再包含 SCRFD)
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
from insightface.utils import face_align
from loguru import logger

from src.config import get_config


class FaceDetectorLight:
    """Tier1 人脸检测 — 只做 detection, 不做 recognition.

    使用 SCRFD_10G (det_10g.onnx, 来自 buffalo_l),
    输出 bbox + 5-point landmarks, 用于 norm_crop 对齐后交给 eDifFIQA 评估质量。
    """

    def __init__(self) -> None:
        import insightface

        # 使用 buffalo_l 自带的 det_10g.onnx (SCRFD_10G), 无需额外下载
        insightface_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
        model_path = insightface_dir / "det_10g.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"SCRFD model not found: {model_path}. "
                "Run: bash download_models.sh"
            )

        config = get_config().face
        hw_config = get_config().hardware
        ctx_id = hw_config.insightface_ctx_id
        providers = (
            [("CUDAExecutionProvider", {"device_id": ctx_id}), "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        self._detector = insightface.model_zoo.get_model(
            str(model_path), providers=providers,
        )
        self._detector.prepare(ctx_id=ctx_id, input_size=config.det_size)
        logger.info("FaceDetectorLight loaded: model=det_10g (SCRFD_10G), ctx_id={}, det_size={}", ctx_id, config.det_size)

    def detect(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """检测最大人脸。

        Args:
            crop: BGR 人体裁剪。

        Returns:
            (bbox, kps_5point) 或 None。
            bbox: (5,) — x1, y1, x2, y2, score
            kps: (5, 2) — 5 点人脸关键点
        """
        bboxes, kpss = self._detector.detect(crop)
        if len(bboxes) == 0:
            return None

        # 选最大脸
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = int(np.argmax(areas))
        return bboxes[best_idx], kpss[best_idx]

    def get_aligned_face(
            self, crop: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """检测 + 对齐。

        Args:
            crop: BGR 人体裁剪。

        Returns:
            (aligned_112, bbox, kps) 或 None。
            aligned_112: 112×112 BGR 对齐人脸
            bbox: (5,) — x1, y1, x2, y2, score
            kps: (5, 2) — 5 点关键点
        """
        result = self.detect(crop)
        if result is None:
            return None
        bbox, kps = result
        aligned = face_align.norm_crop(crop, landmark=kps, image_size=112)
        return aligned, bbox, kps


@cache
def get_face_detector_light() -> FaceDetectorLight:
    """获取 Tier1 轻量人脸检测器 (单例缓存)。"""
    return FaceDetectorLight()
