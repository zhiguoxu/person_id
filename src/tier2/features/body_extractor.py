"""
全身 ReID 特征提取器

使用 SOLIDER (Swin-Small, CVPR 2023) 提取全身 ReID 特征。

特征:
- 768 维 L2 归一化全身嵌入向量 (Swin-Small)
- 支持水平翻转测试增强 (TTA)
- SOLIDER 标准预处理 (resize 384×128, normalize mean=0.5, std=0.5)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from loguru import logger

from src.config import get_config


class BodyExtractor:
    """全身 ReID 特征提取器。

    封装 SOLIDER Swin Transformer ReID 模型,
    从人体检测框中提取全身外观特征。

    Attributes:
        config: ReID 配置。
        model: 已加载的 SOLIDER ReID 模型。
        device: 推理设备。
        EMBEDDING_DIM: 输出嵌入维度 (Swin-Small: 768)。
    """

    # Output embedding dimension (Swin-Small default)
    EMBEDDING_DIM: int = 768

    def __init__(self) -> None:
        """初始化 SOLIDER ReID 特征提取器。"""
        hw_config = get_config().hardware
        self.device = torch.device(hw_config.device)
        self._model = None
        self._pixel_mean = None
        self._pixel_std = None

        self._load_model()

    def _load_model(self) -> None:
        """加载 SOLIDER ReID 模型。"""
        from src.tier2.features.solider.solider_model import SOLIDERReID

        config = get_config().reid

        # 查找权重文件
        weights_path = config.reid_model_weights
        if not weights_path:
            weights_path = SOLIDERReID.find_checkpoint(config.reid_model_name)

        if not weights_path:
            raise RuntimeError(
                "No SOLIDER checkpoint found. Refusing to start with random weights: "
                "random weights produce meaningless body embeddings that change on every "
                "process restart (in-session match ~0.99, post-restart match ~0.0). "
                "Download the SOLIDER Swin-Small ReID weights and place them at "
                "models/solider_swin_small_reid.pth, or set reid.reid_model_weights to "
                "the checkpoint path."
            )

        self._model = SOLIDERReID.from_checkpoint(
            checkpoint_path=weights_path,
            model_name=config.reid_model_name,
            device=str(self.device),
        )

        self.EMBEDDING_DIM = self._model.feat_dim

        # 预计算归一化参数 (转为 tensor, 用于批量预处理)
        self._pixel_mean = torch.tensor(
            config.reid_pixel_mean, dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)
        self._pixel_std = torch.tensor(
            config.reid_pixel_std, dtype=torch.float32, device=self.device,
        ).view(1, 3, 1, 1)

        logger.info(
            "BodyExtractor loaded: model={}, device={}, dim={}",
            config.reid_model_name,
            get_config().hardware.device,
            self.EMBEDDING_DIM,
        )

    def _preprocess_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        """批量预处理: resize + normalize → tensor.

        Args:
            crops: BGR 图像列表

        Returns:
            预处理后的 tensor, shape (N, 3, H, W)
        """
        config = get_config().reid
        input_h, input_w = config.reid_input_size

        batch = []
        for crop in crops:
            # Resize
            resized = cv2.resize(crop, (input_w, input_h))
            # BGR → RGB, HWC → CHW, uint8 → float32 [0, 1]
            img = resized[:, :, ::-1].copy()
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch.append(img)

        tensor = torch.from_numpy(np.stack(batch)).to(self.device)
        # Normalize: (x - mean) / std
        tensor = (tensor - self._pixel_mean) / self._pixel_std
        return tensor

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """批量推理, 利用 GPU 并行提速。

        flip-test TTA 时将原图+翻转图拼为一个大 batch 一次推理,
        避免两次 kernel launch 的开销。

        Args:
            crops: BGR 图像列表 (已裁剪的人体区域)
        Returns:
            L2-normalized embeddings 列表
        """
        if not crops:
            return []

        config = get_config().reid

        try:
            if config.use_flip_test:
                # 原图 + 水平翻转图
                flipped_crops = [cv2.flip(c, 1) for c in crops]
                all_crops = crops + flipped_crops

                tensor = self._preprocess_batch(all_crops)

                with torch.no_grad():
                    features = self._model(tensor)  # (2N, dim)

                features = features.cpu().numpy()
                n = len(crops)
                orig_feats = features[:n]
                flip_feats = features[n:]
                avg_feats = (orig_feats + flip_feats) / 2.0
            else:
                tensor = self._preprocess_batch(crops)

                with torch.no_grad():
                    features = self._model(tensor)  # (N, dim)

                avg_feats = features.cpu().numpy()

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
            logger.warning("Batch SOLIDER extraction failed: {}", e)
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
