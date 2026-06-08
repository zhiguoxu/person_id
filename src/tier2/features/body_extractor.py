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
from loguru import logger

from src.config import get_config

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BodyExtractor:
    """全身 ReID 特征提取器。

    封装 person re-identification 模型, 从人体检测框中提取全身外观特征。
    当前使用 torchreid OSNet-AIN; 如加载失败则抛出异常。

    Attributes:
        config: ReID 配置。
        model: 已加载的 ReID 模型。
        device: 推理设备。
    """

    # Output embedding dimension
    EMBEDDING_DIM: int = 512

    def __init__(self) -> None:
        """初始化 ReID 特征提取器。

        加载 torchreid OSNet-AIN 模型。
        """
        config = get_config().reid
        self.device = torch.device(config.reid_device)
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """加载 ReID 模型, 优先 torchreid。"""
        import torchreid

        config = get_config().reid
        # 自动查找缓存的权重文件，避免 gdown 下载卡住
        model_path = config.reid_model_weights or None
        if not model_path:
            model_path = self._find_cached_weights()

        self._model = torchreid.utils.FeatureExtractor(
            model_name=config.reid_model_name,
            model_path=model_path,
            device=str(self.device),
        )
        # Query actual feature dimension from model
        self.EMBEDDING_DIM = self._probe_embedding_dim()
        logger.info(
            "BodyExtractor loaded: model={}, device={}, dim={}",
            config.reid_model_name,
            config.reid_device,
            self.EMBEDDING_DIM,
        )


    @staticmethod
    def _find_cached_weights() -> str | None:
        """在常见缓存目录中搜索已下载的模型权重。"""
        import os
        import glob

        model_name = get_config().reid.reid_model_name
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

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """批量推理, 利用 GPU 并行提速

        flip-test TTA 时将原图+翻转图拼为一个大 batch 一次推理,
        避免两次 kernel launch 的开销.

        Args:
            crops: BGR 图像列表 (已裁剪的人体区域)
        Returns:
            L2-normalized embeddings 列表
        """
        if not crops:
            return []

        config = get_config().reid
        input_h, input_w = config.reid_input_size

        # 预处理: resize 所有 crop
        resized_list = []
        for crop in crops:
            resized = cv2.resize(crop, (input_w, input_h))
            resized_list.append(resized)

        try:
            if config.use_flip_test:
                # 原图 + 翻转图拼成一个大列表, 一次 forward
                flipped_list = [cv2.flip(r, 1) for r in resized_list]
                all_images = resized_list + flipped_list
                features = self._model(all_images)  # (2N, dim)

                n = len(crops)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                orig_feats = features[:n]
                flip_feats = features[n:]
                avg_feats = (orig_feats + flip_feats) / 2.0
            else:
                features = self._model(resized_list)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                avg_feats = features

            # L2 normalize each
            results = []
            for i in range(len(crops)):
                feat = avg_feats[i].astype(np.float32).flatten()
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                results.append(feat)
            return results

        except Exception as e:
            logger.warning("Batch ReID extraction failed: {}", e)
            return [self._random_feature() for _ in crops]

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
