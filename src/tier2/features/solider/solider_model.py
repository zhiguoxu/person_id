"""
SOLIDER ReID 模型

基于 TransReID 架构 (build_transformer), 使用 Swin Transformer 骨干网络。
支持从以下格式加载权重:
1. SOLIDER 原始预训练 checkpoint (包含 "teacher" 键, backbone 参数带 "backbone." 前缀)
2. SOLIDER-REID finetuned checkpoint (完整模型参数)

模型结构:
    Swin backbone → Global Average Pool → BNNeck → (Classifier)
    推理时返回 GAP 后的特征向量 (不经过 BNNeck)

参考:
    https://github.com/tinyvision/SOLIDER-REID
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from loguru import logger

from .swin_transformer import (
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
)

# Swin 变体 → (工厂函数, 最终输出维度)
_SWIN_VARIANTS = {
    'swin_tiny_patch4_window7_224': (swin_tiny_patch4_window7_224, 768),
    'swin_small_patch4_window7_224': (swin_small_patch4_window7_224, 768),
    'swin_base_patch4_window7_224': (swin_base_patch4_window7_224, 1024),
}

# 配置名 → Swin 变体映射
MODEL_NAME_MAP = {
    'solider_swin_tiny': 'swin_tiny_patch4_window7_224',
    'solider_swin_small': 'swin_small_patch4_window7_224',
    'solider_swin_base': 'swin_base_patch4_window7_224',
}


def _weights_init_kaiming(m):
    """Kaiming initialization for BNNeck."""
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class SOLIDERReID(nn.Module):
    """SOLIDER ReID 特征提取模型。

    基于 build_transformer 架构 (TransReID for Swin),
    使用全局平均池化 + BNNeck 提取特征。
    推理模式下直接返回 BNNeck 之前的全局特征向量。

    Args:
        model_name: 模型名 ('solider_swin_small' 等)
        num_classes: 分类头类别数 (仅训练时使用, 推理时忽略)
        pretrain_hw_ratio: 预训练 H/W 比率 (ReID 通常为 2)
        reduce_feat_dim: 是否降维
        feat_dim: 降维目标维度
    """

    def __init__(
        self,
        model_name: str = 'solider_swin_small',
        num_classes: int = 751,  # Market-1501 默认类别数
        pretrain_hw_ratio: int = 2,
        reduce_feat_dim: bool = False,
        feat_dim: int = 512,
    ):
        super().__init__()
        swin_type = MODEL_NAME_MAP.get(model_name, model_name)
        if swin_type not in _SWIN_VARIANTS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_NAME_MAP.keys())}"
            )

        factory, _ = _SWIN_VARIANTS[swin_type]

        # Swin backbone
        self.base = factory(
            out_indices=(3,),
            pretrain_hw_ratio=pretrain_hw_ratio,
        )

        # 从 backbone 查询实际输出维度
        self.in_planes = self.base.num_features[-1]
        self.feat_dim = self.in_planes

        # 可选降维层
        self.reduce_feat_dim = reduce_feat_dim
        if reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, feat_dim, bias=False)
            nn.init.xavier_uniform_(self.fcneck.weight)
            self.in_planes = feat_dim
            self.feat_dim = feat_dim

        # BNNeck (Batch Normalization bottleneck)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

        # Classification head (training only)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入图像, shape (B, 3, H, W), 已归一化

        Returns:
            训练模式: (cls_score, global_feat, featmaps)
            推理模式: global_feat, shape (B, feat_dim)
        """
        # Swin backbone → (global_feat, featmaps)
        global_feat, featmaps = self.base(x)

        # 可选降维
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.training:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return cls_score, global_feat, featmaps
        else:
            # 推理时返回 BNNeck 之后的特征
            # 实验验证: before BN cosine=0.937(无区分), after BN cosine=0.247(强区分)
            # defaults.py 默认 TEST.NECK_FEAT='after', BNNeck 论文也推荐 after+cosine
            feat = self.bottleneck(global_feat)
            return feat

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str = 'solider_swin_small',
        device: str = 'cpu',
    ) -> 'SOLIDERReID':
        """从 SOLIDER / SOLIDER-REID checkpoint 加载模型。

        自动处理三种 checkpoint 格式:
        1. SOLIDER 原始预训练: {\"teacher\": {\"backbone.xxx\": ...}}
        2. SOLIDER-REID finetuned: {\"model\": {\"base.xxx\": ..., \"bottleneck.xxx\": ...}}
        3. 直接 state_dict: {\"base.xxx\": ..., \"bottleneck.xxx\": ...}

        Args:
            checkpoint_path: checkpoint 文件路径
            model_name: 模型名
            device: 目标设备

        Returns:
            加载好权重的模型
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # --- 解包 checkpoint ---
        if 'teacher' in ckpt:
            # SOLIDER 原始预训练 checkpoint
            state_dict = ckpt['teacher']
            logger.info("检测到 SOLIDER 预训练 checkpoint (teacher 格式)")
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # --- 键名映射 ---
        # SOLIDER 原始权重: "backbone.xxx" → "base.xxx" (映射到 self.base)
        mapped_state = {}
        for k, v in state_dict.items():
            new_k = k
            # "backbone.xxx" → "base.xxx"
            if k.startswith('backbone.'):
                new_k = 'base.' + k[len('backbone.'):]
            mapped_state[new_k] = v

        # --- 探测 num_classes ---
        num_classes = 751
        if 'classifier.weight' in mapped_state:
            num_classes = mapped_state['classifier.weight'].shape[0]

        # --- 检测是否降维 ---
        reduce_feat_dim = 'fcneck.weight' in mapped_state
        feat_dim = 512
        if reduce_feat_dim:
            feat_dim = mapped_state['fcneck.weight'].shape[0]

        # --- 创建模型 ---
        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            reduce_feat_dim=reduce_feat_dim,
            feat_dim=feat_dim,
        )

        # --- 加载权重 ---
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []

        for k, v in mapped_state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    model_state[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_state[k].shape})")
            else:
                skipped_keys.append(k)

        model.load_state_dict(model_state, strict=False)

        logger.info(
            "SOLIDER checkpoint 已加载: {} 个 param 匹配, {} 个跳过, path={}",
            len(loaded_keys), len(skipped_keys), checkpoint_path,
        )
        if skipped_keys and len(skipped_keys) <= 20:
            logger.debug("跳过的 keys: {}", skipped_keys)

        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def find_checkpoint(model_name: str) -> str | None:
        """在常见目录中搜索 SOLIDER 权重文件。

        搜索顺序:
        1. models/ 目录
        2. ~/.cache/solider/

        Args:
            model_name: 模型名 (如 'solider_swin_small')

        Returns:
            权重文件路径, 未找到返回 None
        """
        from src.config import MODELS_DIR

        possible_names = [
            f"{model_name}_reid.pth",
            f"{model_name}.pth",
            "solider_swin_small_reid.pth",
            "swin_small_reid.pth",
        ]

        models_dir = str(MODELS_DIR)
        if os.path.isdir(models_dir):
            for name in possible_names:
                path = os.path.join(models_dir, name)
                if os.path.isfile(path):
                    logger.info("找到 SOLIDER weights: {}", path)
                    return path

        return None
