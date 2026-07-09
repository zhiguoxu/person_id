"""
人脸特征提取器 — ArcFace / AdaFace 可切换

通过配置 ``recognition_backend`` 在 ArcFace 和 AdaFace 之间切换。
两者均使用 ONNX Runtime 直接推理, 不依赖 insightface.model_zoo。

从预对齐的 112×112 人脸提取 512 维 L2 归一化嵌入向量。

Tier1 SCRFD 已完成人脸检测 + 对齐 (aligned_face 112×112),
本模块只负责 embedding 提取, 不再重复检测。

支持的后端:
- arcface: InsightFace w600k_r50 (BGR→RGB, normalize [-1,1])
- adaface: AdaFace IR-101 (BGR→RGB, normalize [-1,1])
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.config import get_config, MODELS_DIR


class FaceExtractor:
    """人脸嵌入提取器 — 支持 ArcFace / AdaFace 后端切换。

    通过 ONNX Runtime 直接加载模型, 从预对齐的 112×112 人脸
    提取 512 维 L2 归一化特征向量。

    两种后端的预处理一致:
    - ArcFace: BGR → RGB → normalize to [-1, 1]
    - AdaFace: BGR → RGB → normalize to [-1, 1]
    """

    def __init__(self) -> None:
        """初始化人脸嵌入提取器。"""
        config = get_config().face
        self._backend = config.recognition_backend

        # 选择模型文件
        if self._backend == "arcface":
            model_file = config.arcface_model
        elif self._backend == "adaface":
            model_file = config.adaface_model
        else:
            raise ValueError(
                f"Unknown recognition_backend: '{self._backend}'. "
                "Supported: 'arcface', 'adaface'"
            )

        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face recognition model not found: {model_path}. "
                "Run: bash download_models.sh"
            )

        try:
            import onnxruntime as ort

            ctx_id = get_config().hardware.insightface_ctx_id
            providers = self._get_providers(ctx_id)

            self._session = ort.InferenceSession(
                str(model_path), providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name

            logger.info(
                "FaceExtractor 已加载: backend={}, model={}, ctx_id={}",
                self._backend, model_file, ctx_id,
            )
        except ImportError:
            logger.error(
                "未安装 onnxruntime。"
                "请通过以下命令安装: pip install onnxruntime-gpu"
            )
            raise
        except Exception as e:
            logger.error("FaceExtractor 初始化失败: {}", e)
            raise

    @property
    def default_channel_order(self) -> str:
        """backend 默认送入模型的通道顺序 (arcface / adaface 均为 rgb)。"""
        return "rgb"

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

    def _preprocess(
        self, aligned_face: np.ndarray, channel_order: str | None = None,
    ) -> np.ndarray:
        """预处理对齐人脸 → ONNX 输入 tensor。

        Args:
            aligned_face: 112×112 BGR 对齐人脸。
            channel_order: 强制指定送入模型的通道顺序 ``"bgr"`` / ``"rgb"``;
                为 ``None`` 时按 backend 默认 (arcface / adaface 均为 RGB)。

        Returns:
            (1, 3, 112, 112) float32 tensor, 值域 [-1, 1]。
        """
        img = aligned_face.astype(np.float32)

        # 未指定时按 backend 默认 (arcface / adaface 均为 rgb)
        order = channel_order or self.default_channel_order

        if order == "rgb":
            # 输入是 BGR, 翻转为 RGB
            img = img[:, :, ::-1].copy()

        # 统一归一化到 [-1, 1]
        img = (img - 127.5) / 127.5
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # 添加 batch 维度
        return np.expand_dims(img, 0)

    def extract_embedding(
        self, aligned_face: np.ndarray, channel_order: str | None = None,
    ) -> np.ndarray | None:
        """从预对齐的 112×112 人脸提取嵌入。

        Args:
            aligned_face: 112×112 BGR 对齐人脸 (来自 Tier1 norm_crop)。
            channel_order: 强制指定送入模型的通道顺序 ``"bgr"`` / ``"rgb"``;
                为 ``None`` 时按 backend 默认。

        Returns:
            512 维 L2 归一化嵌入向量, 或 None (提取失败时)。
        """
        if self._session is None:
            logger.warning("FaceExtractor session 未初始化")
            return None

        try:
            blob = self._preprocess(aligned_face, channel_order)
            output = self._session.run(None, {self._input_name: blob})
            embedding = output[0].flatten().astype(np.float32)

            norm = float(np.linalg.norm(embedding))
            if norm < 1e-6:
                logger.debug("Face embedding 范数(norm)接近 0")
                return None
            embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.debug("提取 face embedding 失败: {}", e)
            return None
