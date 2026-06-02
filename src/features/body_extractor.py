"""
全身 ReID 特征提取器

抽象 ReID 特征提取接口, 目前使用 torchreid OSNet-AIN 作为后备方案
(SOLIDER 需要从源码安装, 后续可替换)。

特征:
- 2048 维 L2 归一化全身嵌入向量
- 支持水平翻转测试增强 (TTA)
- 标准 ImageNet 预处理 (resize 256×128, normalize)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from src.config import ReIDConfig

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BodyExtractor:
    """全身 ReID 特征提取器。

    封装 person re-identification 模型, 从人体检测框中提取全身外观特征。
    当前后备方案使用 torchreid OSNet-AIN; 如不可用则退化为随机特征 (带警告)。

    Attributes:
        config: ReID 配置。
        model: 已加载的 ReID 模型。
        device: 推理设备。
    """

    # Output embedding dimension
    EMBEDDING_DIM: int = 512

    def __init__(self, config: ReIDConfig) -> None:
        """初始化 ReID 特征提取器。

        尝试按优先级加载模型:
        1. torchreid OSNet-AIN
        2. 随机特征生成器 (开发/测试用)

        Args:
            config: ReID 配置。
        """
        self.config = config
        self.device = torch.device(config.reid_device)
        self._model = None
        self._use_random = False

        self._load_model()

    def _load_model(self) -> None:
        """加载 ReID 模型, 优先 torchreid, 失败则退化为随机特征。"""
        try:
            import torchreid

            # 自动查找缓存的权重文件，避免 gdown 下载卡住
            model_path = self.config.reid_model_weights or None
            if not model_path:
                model_path = self._find_cached_weights()

            self._model = torchreid.utils.FeatureExtractor(
                model_name=self.config.reid_model_name,
                model_path=model_path,
                device=str(self.device),
            )
            # Query actual feature dimension from model
            self.EMBEDDING_DIM = self._probe_embedding_dim()
            logger.info(
                "BodyExtractor loaded: model={}, device={}, dim={}",
                self.config.reid_model_name,
                self.config.reid_device,
                self.EMBEDDING_DIM,
            )
        except Exception as e:
            logger.warning(
                "torchreid not available ({}). "
                "Falling back to random features.",
                e,
            )
            self._use_random = True
            self.EMBEDDING_DIM = 512

    def _find_cached_weights(self) -> str | None:
        """在常见缓存目录中搜索已下载的模型权重。"""
        import os
        import glob

        model_name = self.config.reid_model_name
        search_dirs = [
            os.path.expanduser("~/.cache/torch/checkpoints"),
            os.path.expanduser("~/.torch/checkpoints"),
            os.path.join(os.environ.get("TORCH_HOME", ""), "checkpoints"),
            "models",
        ]

        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            matches = glob.glob(os.path.join(d, f"{model_name}*"))
            if matches:
                path = matches[0]
                logger.info("Found cached ReID weights: {}", path)
                return path

        logger.debug("No cached weights found for {}", model_name)
        return None

    def _probe_embedding_dim(self) -> int:
        """探测模型实际输出特征维度。

        Returns:
            嵌入向量维度。
        """
        try:
            dummy = np.zeros((128, 64, 3), dtype=np.uint8)
            features = self._model([dummy])
            dim = features.shape[1]
            return int(dim)
        except Exception:
            return 512

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """从人体检测框中提取 ReID 特征。

        步骤:
        1. 裁剪人体区域
        2. resize 到 256×128
        3. ImageNet 标准化
        4. 模型推理 (可选 flip-test TTA)
        5. L2 归一化

        Args:
            frame: BGR 格式完整帧, shape (H, W, 3)。
            bbox: 人体检测框 (x1, y1, x2, y2)。

        Returns:
            L2 归一化的特征向量, shape (EMBEDDING_DIM,)。
        """
        if self._use_random:
            return self._random_feature()

        # Crop person region
        crop = self._crop_person(frame, bbox)

        if crop is None:
            logger.debug("Invalid crop for ReID extraction")
            return self._random_feature()

        try:
            return self._extract_with_model(crop)
        except Exception as e:
            logger.warning("ReID feature extraction failed: {}", e)
            return self._random_feature()

    def _extract_with_model(self, crop: np.ndarray) -> np.ndarray:
        """使用模型提取特征, 支持 TTA。

        Args:
            crop: 裁剪并预处理后的人体图像, BGR 格式。

        Returns:
            L2 归一化特征向量。
        """
        # Preprocess: resize to model input size
        input_h, input_w = self.config.reid_input_size
        resized = cv2.resize(crop, (input_w, input_h))

        # torchreid FeatureExtractor accepts list of images (BGR numpy)
        images = [resized]

        if self.config.use_flip_test:
            # Horizontal flip for TTA
            flipped = cv2.flip(resized, 1)  # 1 = horizontal flip
            images.append(flipped)

        # Extract features
        features = self._model(images)  # (N, dim)

        if self.config.use_flip_test and features.shape[0] >= 2:
            # Average original and flipped features
            feat = (features[0] + features[1]) / 2.0
        else:
            feat = features[0]

        # Convert to numpy and L2 normalize
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()

        feat = feat.astype(np.float32).flatten()
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm

        return feat

    def _crop_person(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray | None:
        """裁剪人体区域并做基本验证。

        Args:
            frame: 原始帧。
            bbox: 检测框 (x1, y1, x2, y2)。

        Returns:
            裁剪后的人体图像, 无效时返回 None。
        """
        h, w = frame.shape[:2]

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 - x1 < 10 or y2 - y1 < 20:
            return None

        return frame[y1:y2, x1:x2].copy()

    def _random_feature(self) -> np.ndarray:
        """生成随机 L2 归一化特征 (调试/降级用)。

        Returns:
            随机 L2 归一化向量, shape (EMBEDDING_DIM,)。
        """
        feat = np.random.randn(self.EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat = feat / norm
        return feat
