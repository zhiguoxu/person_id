"""
人脸特征提取器

直接加载 ArcFace ONNX 模型 (w600k_r50.onnx from buffalo_l),
从预对齐的 112×112 人脸提取 512 维 L2 归一化嵌入向量。

Tier1 SCRFD 已完成人脸检测 + 对齐 (aligned_face 112×112),
本模块只负责 ArcFace embedding 提取, 不再重复检测。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from src.config import get_config


class FaceExtractor:
    """ArcFace 人脸嵌入提取器。

    直接加载 ArcFace ONNX 模型, 从预对齐的 112×112 人脸
    提取 512 维 L2 归一化特征向量。

    绕过 FaceAnalysis (它强制要求 detection 模块),
    直接通过 insightface.model_zoo 加载 recognition 模型。
    """

    def __init__(self) -> None:
        """初始化 ArcFace 嵌入提取器。"""
        config = get_config().face

        try:
            import insightface

            # 直接加载 ArcFace 模型, 绕过 FaceAnalysis
            model_dir = Path.home() / ".insightface" / "models" / config.insightface_model
            rec_path = model_dir / "w600k_r50.onnx"
            if not rec_path.exists():
                raise FileNotFoundError(
                    f"ArcFace model not found: {rec_path}. "
                    "Run: bash download_models.sh"
                )

            ctx_id = config.insightface_ctx_id
            providers = self._get_providers(ctx_id)

            self._rec_model = insightface.model_zoo.get_model(
                str(rec_path), providers=providers,
            )
            self._rec_model.prepare(ctx_id=ctx_id)

            logger.info(
                "FaceExtractor loaded: model=w600k_r50 (ArcFace only), ctx_id={}",
                ctx_id,
            )
        except ImportError:
            logger.error(
                "insightface is not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
            raise
        except Exception as e:
            logger.error("Failed to initialize FaceExtractor: {}", e)
            raise

    @staticmethod
    def _get_providers(ctx_id: int) -> list[str]:
        """根据设备 ID 获取 ONNX Runtime providers。

        Args:
            ctx_id: CUDA 设备 ID, 负数表示使用 CPU。

        Returns:
            ONNX Runtime execution providers 列表。
        """
        if ctx_id >= 0:
            return [
                ("CUDAExecutionProvider", {"device_id": ctx_id}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray | None:
        """从预对齐的 112×112 人脸提取 ArcFace 嵌入。

        Args:
            aligned_face: 112×112 BGR 对齐人脸 (来自 Tier1 norm_crop)。

        Returns:
            512 维 L2 归一化嵌入向量, 或 None (提取失败时)。
        """
        if self._rec_model is None:
            logger.warning("FaceExtractor recognition model not initialized")
            return None

        try:
            # ArcFace 直接从对齐人脸提取特征 — 不需要重新检测
            embedding = self._rec_model.get_feat(aligned_face)

            if embedding is None:
                logger.debug("No embedding extracted for aligned face")
                return None

            embedding = embedding.flatten().astype(np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm < 1e-6:
                logger.debug("Face embedding has near-zero norm")
                return None
            embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.debug("Failed to extract face embedding: {}", e)
            return None
