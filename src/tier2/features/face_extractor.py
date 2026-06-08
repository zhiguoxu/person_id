"""
人脸特征提取器

封装 InsightFace (SCRFD 检测 + ArcFace 嵌入), 从人体检测区域中提取:
- 512 维 L2 归一化人脸嵌入向量
- 人脸检测框和 5 点关键点
- 人脸质量评分

用于人脸库的特征入库和匹配。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.config import get_config
from src.pipeline.data_models import FaceResult

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face


class FaceExtractor:
    """InsightFace 人脸特征提取器。

    使用 InsightFace 的 FaceAnalysis 进行人脸检测 (SCRFD) 和
    嵌入提取 (ArcFace), 输出 512 维 L2 归一化特征向量。

    Attributes:
        config: 人脸配置。
        app: InsightFace FaceAnalysis 实例。
    """

    def __init__(self) -> None:
        """初始化人脸特征提取器。"""
        config = get_config().face

        try:
            from insightface.app import FaceAnalysis
            self._app: FaceAnalysis = FaceAnalysis(
                name=config.insightface_model,
                allowed_modules=["detection", "recognition"],
                providers=self._get_providers(config.insightface_ctx_id),
            )
            self._app.prepare(
                ctx_id=config.insightface_ctx_id,
                det_size=config.det_size,
            )
            logger.info(
                "FaceExtractor loaded: model={}, ctx_id={}, det_size={}",
                config.insightface_model,
                config.insightface_ctx_id,
                config.det_size,
            )
        except ImportError:
            logger.error(
                "insightface is not installed. "
                "Install with: pip install insightface onnxruntime-gpu"
            )
            raise
        except Exception as e:
            logger.error("Failed to initialize InsightFace: {}", e)
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

    def extract(
            self,
            frame: np.ndarray,
            person_bbox: np.ndarray,
    ) -> FaceResult | None:
        """从人体检测区域中提取人脸特征。

        步骤:
        1. 裁剪人体上半身区域 (检测框上方 60% 部分)
        2. 运行 SCRFD 人脸检测
        3. 选择面积最大的人脸 (避免误检背景人脸)
        4. 提取 ArcFace 512 维嵌入并 L2 归一化
        5. 计算简单的人脸质量分

        Args:
            frame: BGR 格式完整帧, shape (H, W, 3)。
            person_bbox: 人体检测框 (x1, y1, x2, y2)。

        Returns:
            FaceResult 如果成功检测到人脸, 否则 None。
        """
        if self._app is None:
            logger.warning("FaceExtractor not initialized")
            return None

        h, w = frame.shape[:2]

        # Crop upper body region (top 60% of person bbox)
        px1: int = max(0, int(person_bbox[0]))
        py1: int = max(0, int(person_bbox[1]))
        px2: int = min(w, int(person_bbox[2]))
        py2: int = min(h, int(person_bbox[3]))
        person_height: int = py2 - py1

        # Focus on upper body for face detection
        crop_y2: int = py1 + int(person_height * 0.6)
        crop: np.ndarray = frame[py1:crop_y2, px1:px2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        try:
            faces: list[Face] = self._app.get(crop)
        except Exception as e:
            logger.debug("InsightFace detection failed: {}", e)
            return None

        if not faces:
            return None

        # Select the largest face (most likely the target person)
        best_face: Face = max(faces, key=lambda f: _face_area(f))

        try:
            # Extract embedding — InsightFace returns L2-normalized 512-dim
            embedding: np.ndarray | None = best_face.embedding
            if embedding is None:
                logger.debug("No embedding extracted for detected face")
                return None

            # Ensure L2 normalization
            embedding = embedding.astype(np.float32)
            norm: float = float(np.linalg.norm(embedding))
            if norm < 1e-6:
                logger.debug("Face embedding has near-zero norm")
                return None
            embedding = embedding / norm

            # Get face bbox in original frame coordinates
            face_bbox: np.ndarray = best_face.bbox.astype(np.float32)
            face_bbox[0] += px1
            face_bbox[1] += py1
            face_bbox[2] += px1
            face_bbox[3] += py1

            # Get landmarks (5-point) in original frame coordinates
            landmarks: np.ndarray = best_face.kps.astype(np.float32)  # (5, 2)
            landmarks[:, 0] += px1
            landmarks[:, 1] += py1

            # Detection score
            det_score: float = float(best_face.det_score)

            # Compute simple quality score
            quality: float = self._compute_quality(face_bbox, det_score)

            return FaceResult(
                embedding=embedding,
                quality=quality,
                landmarks=landmarks,
                bbox=face_bbox,
                det_score=det_score,
            )

        except Exception as e:
            logger.debug("Failed to extract face features: {}", e)
            return None

    @staticmethod
    def _compute_quality(
            face_bbox: np.ndarray,
            det_score: float,
    ) -> float:
        """计算简单的人脸质量分数。

        基于:
        - 检测置信度
        - 人脸尺寸相对于最小要求

        这是一个粗略估计, QualityAssessor 提供更完整的评估。

        Args:
            face_bbox: 人脸框 (x1, y1, x2, y2)。
            det_score: 检测置信度。

        Returns:
            质量分数 [0, 1]。
        """
        face_w: float = face_bbox[2] - face_bbox[0]
        face_h: float = face_bbox[3] - face_bbox[1]
        face_size: float = min(float(face_w), float(face_h))

        # Size score: penalize very small faces
        size_score: float = min(1.0, face_size / get_config().face.min_face_size)

        # Combine with detection confidence
        quality: float = 0.6 * det_score + 0.4 * size_score

        return float(np.clip(quality, 0.0, 1.0))


def _face_area(face: Face) -> float:
    """计算人脸检测框面积。

    Args:
        face: InsightFace 检测到的 Face 对象。

    Returns:
        人脸框面积 (像素)。
    """
    bbox: np.ndarray = face.bbox
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
