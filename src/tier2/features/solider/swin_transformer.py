"""
Swin Transformer Backbone — 自包含实现

从 SOLIDER-REID (tinyvision/SOLIDER-REID) 的 swin_transformer.py 精简而来,
移除了 mmcv 依赖, 使用纯 PyTorch 实现。

关键修改:
- 移除 mmcv.runner.load_checkpoint 依赖
- 移除训练相关代码 (drop_path 训练模式等)
- 保留与 SOLIDER-REID checkpoint 完全兼容的参数命名

Swin-Small 参数:
- embed_dims = 96
- depths = [2, 2, 18, 2]
- num_heads = [3, 6, 12, 24]
- window_size = 7
- 最终输出维度: 768 (= 96 * 8)

Reference:
    Liu et al., "Swin Transformer: Hierarchical Vision Transformer
    using Shifted Windows", ICCV 2021
"""
from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from typing import Sequence
import collections.abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ─────────────────── Utility Functions ─────────────────────────────────────

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Truncated normal initialization (no grad)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in trunc_normal_.',
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def trunc_normal_init(module, mean=0, std=1, a=-2, b=2, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ─────────────────── Adaptive Padding ──────────────────────────────────────

class AdaptivePadding(nn.Module):
    """Pads input so it gets fully covered by the convolution kernel."""

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        out_h = math.ceil(input_h / stride_h)
        out_w = math.ceil(input_w / stride_w)
        pad_h = max((out_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
        pad_w = max((out_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.shape[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                ])
        return x


# ─────────────────── Patch Embedding ───────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image to Patch Embedding using convolution."""

    def __init__(self, in_channels=3, embed_dims=96, conv_type='Conv2d',
                 kernel_size=4, stride=4, padding='corner', norm_layer=None):
        super().__init__()
        self.embed_dims = embed_dims

        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)

        self.adaptive_padding = AdaptivePadding(
            kernel_size=kernel_size, stride=stride, padding=padding,
        )

        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size, stride=stride,
        )
        self.norm = norm_layer(embed_dims) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.adaptive_padding(x)
        x = self.projection(x)  # (B, C, H, W)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, out_size


# ─────────────────── Patch Merging ─────────────────────────────────────────

class PatchMerging(nn.Module):
    """Merge patches to reduce spatial resolution (downsample 2x).

    使用 nn.Unfold 实现, 与原版 SOLIDER-REID 完全一致。
    注意: nn.Unfold 的通道拼接顺序与手动切片 x[::2,::2] 不同,
    必须使用 nn.Unfold 才能与 checkpoint 权重兼容。
    """

    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 使用 AdaptivePadding + nn.Unfold (与原版一致)
        self.adaptive_padding = AdaptivePadding(
            kernel_size=2, stride=2, padding='corner',
        )
        self.sampler = nn.Unfold(kernel_size=2, stride=2)
        self.norm = norm_layer(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)

    def forward(self, x, input_size):
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W, f'input size ({H}*{W}) != seq length ({L})'

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # AdaptivePadding 确保 H, W 被 kernel_size 整除
        x = self.adaptive_padding(x)
        H, W = x.shape[-2:]

        # nn.Unfold: (B, C, H, W) → (B, C*kernel_h*kernel_w, L)
        x = self.sampler(x)  # (B, 4C, H/2*W/2)
        out_h, out_w = H // 2, W // 2

        x = x.transpose(1, 2)  # (B, H/2*W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, (out_h, out_w)


# ─────────────────── Window Attention ──────────────────────────────────────

class WindowMSA(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA)."""

    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True,
                 attn_drop_rate=0., proj_drop_rate=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = embed_dims // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ShiftWindowMSA(nn.Module):
    """Shifted Window Multi-head Self Attention."""

    def __init__(self, embed_dims, num_heads, window_size, shift_size=0,
                 qkv_bias=True, attn_drop_rate=0., proj_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = nn.Identity()  # DropPath placeholder

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, f'query length {L} != H*W {H}*{W}'
        query = query.view(B, H, W, C)

        # Pad to multiples of window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # Cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # Compute attention mask
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Partition mask into windows
            mask_windows = self._window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            shifted_query = query
            attn_mask = None

        # Partition into windows
        query_windows = self._window_partition(shifted_query, self.window_size)
        query_windows = query_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_query = self._window_reverse(attn_windows, self.window_size, H_pad, W_pad)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_query, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_query

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    @staticmethod
    def _window_partition(x, window_size):
        """Partition into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def _window_reverse(windows, window_size, H, W):
        """Reverse window partition."""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


# ─────────────────── Swin Transformer Block ────────────────────────────────

class SwinBlock(nn.Module):
    """Basic Swin Transformer Block."""

    def __init__(self, embed_dims, num_heads, window_size=7, shift=False,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dims)
        shift_size = window_size // 2 if shift else 0
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(embed_dims)
        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        # 使用 SOLIDER 命名: ffn.layers.0.0 / ffn.layers.1
        self.ffn = nn.Module()
        self.ffn.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, mlp_hidden_dim), nn.GELU()),
            nn.Linear(mlp_hidden_dim, embed_dims),
        )
        self.ffn_drop = nn.Dropout(drop_rate)

    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = self.ffn.layers(x)
        x = self.ffn_drop(x)
        x = identity + x
        return x


# ─────────────────── Swin Stage ────────────────────────────────────────────

class SwinStage(nn.Module):
    """A single Swin Transformer stage."""

    def __init__(self, embed_dims, num_heads, depth, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,
                 downsample=None, out_channels=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                shift=(i % 2 == 1),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
            )
            self.blocks.append(block)

        self.downsample = None
        if downsample is not None:
            self.downsample = downsample(
                in_channels=embed_dims,
                out_channels=out_channels,
            )

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        # 返回 downsample 前后的特征 (与原版 SOLIDER-REID 一致)
        if self.downsample is not None:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


# ─────────────────── Swin Transformer ──────────────────────────────────────

class SwinTransformer(nn.Module):
    """Swin Transformer backbone.

    完全自包含实现, 无 mmcv 依赖。
    参数命名与 SOLIDER-REID checkpoint 兼容。

    Args:
        pretrain_img_size: 预训练图像尺寸 (用于 position embedding)
        embed_dims: 初始嵌入维度
        patch_size: Patch 大小
        window_size: 注意力窗口大小
        depths: 各 stage 的 block 数
        num_heads: 各 stage 的注意力头数
        mlp_ratio: MLP 隐藏层维度比率
        qkv_bias: 是否使用 QKV bias
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: DropPath rate
        out_indices: 输出特征的 stage 索引
        pretrain_hw_ratio: 预训练 H/W 比率 (用于 position bias 插值)
    """

    def __init__(
        self,
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        out_indices=(3,),
        pretrain_hw_ratio=1,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.embed_dims = embed_dims
        self.num_layers = len(depths)
        self.out_indices = out_indices

        # 各 stage 输出通道数列表 (供上层模块查询, e.g. build_transformer)
        self.num_features = [embed_dims * (2 ** i) for i in range(len(depths))]

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            in_channels=3,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_layer=nn.LayerNorm,
        )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # Build stages
        self.stages = nn.ModuleList()
        in_channels = embed_dims
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                downsample = PatchMerging
                out_channels = 2 * in_channels
            else:
                downsample = None
                out_channels = in_channels

            stage = SwinStage(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                depth=depths[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                downsample=downsample,
                out_channels=out_channels,
            )
            self.stages.append(stage)

            if downsample is not None:
                in_channels = out_channels

        # Layer norms for output stages
        for i in out_indices:
            dim = embed_dims * (2 ** i)
            norm = nn.LayerNorm(dim)
            self.add_module(f'norm{i}', norm)

        self.pretrain_hw_ratio = pretrain_hw_ratio

        # 全局平均池化 (与原版 SOLIDER-REID 一致, 使用 AdaptiveAvgPool2d)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── SOLIDER Semantic Embedding ────────────────────────────────
        # 原版 SOLIDER 在每个 stage 之间对特征做 channel-wise affine 变换:
        #   x = x * softplus(sw) + sb
        # 其中 sw, sb 由 semantic_weight 参数 (默认 0.2) 控制。
        # 这些权重在训练时冻结, 推理时使用固定的 [0.2, 0.8] 输入。
        # 不加入这个模块会导致特征坍缩 (所有输出余弦相似度 > 0.99)。
        self.semantic_weight = 0.2
        self.semantic_embed_w = nn.ModuleList()
        self.semantic_embed_b = nn.ModuleList()
        for i in range(len(depths)):
            # 原版: i >= len(depths)-1 时 clamp 到 len(depths)-2
            feat_idx = min(i, len(depths) - 2)
            out_dim = self.num_features[feat_idx + 1]
            sem_w = nn.Linear(2, out_dim)
            sem_b = nn.Linear(2, out_dim)
            # 冻结参数 (与原版一致)
            for param in sem_w.parameters():
                param.requires_grad = False
            for param in sem_b.parameters():
                param.requires_grad = False
            self.semantic_embed_w.append(sem_w)
            self.semantic_embed_b.append(sem_b)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """前向传播。

        与原版 SOLIDER-REID SwinTransformer.forward 完全一致:
        - stage 返回 downsample 前的特征用于 norm
        - stage 之间施加 semantic embedding 调制
        - out reshape 为 4D (B, C, H, W)
        - 使用 AdaptiveAvgPool2d 做全局池化

        Args:
            x: 输入图像, shape (B, 3, H, W)

        Returns:
            (global_feat, outs):
                global_feat: 全局平均池化特征, shape (B, C)
                outs: 各 stage 输出特征列表 (4D format)
        """
        x, hw_shape = self.patch_embed(x)
        x = self.drop_after_pos(x)

        # 构建 semantic weight 输入: [semantic_weight, 1 - semantic_weight]
        batch_size = x.shape[0]
        w = torch.ones(batch_size, 1, device=x.device) * self.semantic_weight
        semantic_input = torch.cat([w, 1 - w], dim=-1)  # (B, 2)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)

            # Semantic embedding 调制 (在 downsample 后的 x 上施加)
            sw = self.semantic_embed_w[i](semantic_input).unsqueeze(1)  # (B, 1, C)
            sb = self.semantic_embed_b[i](semantic_input).unsqueeze(1)  # (B, 1, C)
            x = x * self.softplus(sw) + sb

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                # Reshape to 4D: (B, N, C) → (B, C, H, W)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        # 全局平均池化
        x = self.avgpool(outs[-1])  # (B, C, 1, 1)
        global_feat = torch.flatten(x, 1)  # (B, C)

        return global_feat, outs

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """加载预训练权重。

        支持 SOLIDER-REID checkpoint 格式, 自动处理:
        1. 'model' / 'state_dict' 键名包装
        2. 不匹配的参数跳过
        3. Relative position bias 尺寸不匹配时的插值

        Args:
            checkpoint_path: 权重文件路径
            strict: 是否严格匹配所有参数
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 处理 relative_position_bias_table 尺寸不匹配
        model_state = self.state_dict()
        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k and k in model_state:
                src_shape = state_dict[k].shape
                dst_shape = model_state[k].shape
                if src_shape != dst_shape:
                    # 插值
                    src_size = int(src_shape[0] ** 0.5)
                    dst_size = int(dst_shape[0] ** 0.5)
                    extra_tokens = src_shape[0] - src_size ** 2

                    if extra_tokens == 0:
                        src_bias = state_dict[k].float()
                        src_bias = src_bias.reshape(src_size, src_size, -1).permute(2, 0, 1)
                        dst_bias = F.interpolate(
                            src_bias.unsqueeze(0),
                            size=(dst_size, dst_size),
                            mode='bicubic',
                            align_corners=False,
                        ).squeeze(0)
                        state_dict[k] = dst_bias.permute(1, 2, 0).reshape(-1, dst_shape[1])

        # 过滤不匹配的参数
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered[k] = v

        msg = self.load_state_dict(filtered, strict=False)
        return msg


# ─────────────────── Factory Functions ─────────────────────────────────────

def swin_tiny_patch4_window7_224(**kwargs):
    """Swin-Tiny: embed_dims=96, depths=[2,2,6,2]"""
    defaults = dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )
    defaults.update(kwargs)
    return SwinTransformer(**defaults)


def swin_small_patch4_window7_224(**kwargs):
    """Swin-Small: embed_dims=96, depths=[2,2,18,2]"""
    defaults = dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )
    defaults.update(kwargs)
    return SwinTransformer(**defaults)


def swin_base_patch4_window7_224(**kwargs):
    """Swin-Base: embed_dims=128, depths=[2,2,18,2]"""
    defaults = dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
    )
    defaults.update(kwargs)
    return SwinTransformer(**defaults)
