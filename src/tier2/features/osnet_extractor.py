"""
OSNet-AIN ReID 特征提取器 (torchreid)

用于与 SOLIDER Swin-Small 做性能对比。
使用 torchreid 的 FeatureExtractor 接口。
"""
from __future__ import annotations

import os
import glob

import cv2
import numpy as np
import torch
from loguru import logger


class OSNetExtractor:
    """OSNet-AIN ReID 特征提取器。

    基于 torchreid 的 OSNet-AIN x1.0 模型。
    支持 ImageNet 预训练和 MSMT17 ReID 微调两种权重。

    Attributes:
        EMBEDDING_DIM: 输出嵌入维度 (OSNet: 512)
    """

    EMBEDDING_DIM: int = 512

    def __init__(self, device: str = 'cuda', model_path: str | None = None) -> None:
        self.device = torch.device(device)
        self._model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str | None = None) -> None:
        """加载 torchreid OSNet-AIN 模型。"""
        import torchreid

        if not model_path:
            model_path = self._find_weights()

        self._model = torchreid.utils.FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=model_path,
            device=str(self.device),
        )

        # 探测实际输出维度
        self.EMBEDDING_DIM = self._probe_dim()
        logger.info(
            "OSNetExtractor 已加载: weights={}, device={}, dim={}",
            os.path.basename(model_path) if model_path else 'default',
            self.device,
            self.EMBEDDING_DIM,
        )

    @staticmethod
    def _find_weights() -> str | None:
        """搜索 OSNet 权重文件, 优先 ReID 微调权重。"""
        search_dirs = [
            'models',
            os.path.expanduser('~/.cache/torch/checkpoints'),
            os.path.expanduser('~/.torch/checkpoints'),
        ]

        # 优先查找 ReID 微调权重 (msmt17)
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, '*osnet*msmt17*')):
                logger.info("找到 OSNet ReID weights: {}", f)
                return f

        # 退而求其次: ImageNet 预训练
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, '*osnet*')):
                logger.info("找到 OSNet ImageNet weights: {}", f)
                return f

        return None

    def _probe_dim(self) -> int:
        """探测输出维度。"""
        try:
            dummy = np.zeros((128, 64, 3), dtype=np.uint8)
            features = self._model([dummy])
            return int(features.shape[1])
        except Exception:
            return 512

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """批量提取特征。

        Args:
            crops: BGR 图像列表 (已裁剪的人体区域)

        Returns:
            L2-normalized embeddings 列表
        """
        if not crops:
            return []

        try:
            # torchreid FeatureExtractor 接受 BGR numpy 图像列表
            # 内部会自动 resize 到 256x128 并做 ImageNet normalize
            resized = [cv2.resize(c, (128, 256)) for c in crops]
            features = self._model(resized)  # (N, dim) tensor

            results = []
            for i in range(len(crops)):
                feat = features[i].cpu().numpy().astype(np.float32).flatten()
                norm = np.linalg.norm(feat)
                if norm > 1e-6:
                    feat = feat / norm
                results.append(feat)
            return results

        except Exception as e:
            # 返回空列表而非随机向量: 随机向量会污染相似度对比,
            # 调用方以 len(结果) 判断成败 (与 BodyExtractor 的失败契约一致)
            logger.warning("OSNet feature 提取失败: {}", e)
            return []
