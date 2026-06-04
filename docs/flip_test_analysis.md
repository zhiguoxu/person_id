# Flip-Test TTA 收益评估

> 评估 ReID 推理阶段的水平翻转测试时增强 (flip-test TTA) 在人脸和人体特征提取中的成本与收益。

## 学术基准收益

### 人体 ReID (SOLIDER/TransReID)

| 数据集 | 指标 | 无 flip | 有 flip | 提升 |
|:--|:--|:--:|:--:|:--:|
| Market-1501 | mAP | ~93.0 | ~93.5 | **+0.5** |
| Market-1501 | Rank-1 | ~95.8 | ~96.0 | **+0.2** |
| MSMT17 | mAP | ~68.0 | ~69.5 | **+1.5** |
| MSMT17 | Rank-1 | ~85.0 | ~85.8 | **+0.8** |

> MSMT17 更难（更多干扰/遮挡），flip-test 收益更大。

### 人脸识别 (AdaFace)

| 数据集 | 指标 | 无 flip | 有 flip | 提升 |
|:--|:--|:--:|:--:|:--:|
| LFW | Accuracy | 99.82 | 99.83 | +0.01 (饱和) |
| CPLFW (跨姿态) | Accuracy | ~93.5 | ~94.0 | **+0.5** |
| CFP-FP (正侧面) | Accuracy | ~98.0 | ~98.5 | **+0.5** |
| IJB-C TAR@FAR=1e-4 | TAR | ~96.0 | ~96.8 | **+0.8** |

> 简单场景 (LFW) 几乎无收益；**跨姿态/困难场景收益 0.5-1.0%**。

## 计算开销

### 朴素实现 (两次独立推理)

| 模型 | 单帧 | 10 帧 (2次推理) |
|:--|:--:|:--:|
| SOLIDER Swin-B | ~4ms | ~60ms → ~120ms |
| AdaFace | ~2ms | ~20ms → ~40ms |

### 优化后 (原图+翻转拼为单次 batch 推理)

```python
# 原图 + 翻转图拼成一个大 batch, 一次前向传播
combined = torch.cat([batch_tensor, torch.flip(batch_tensor, dims=[3])], dim=0)  # (2N, C, H, W)
with torch.no_grad():
    all_features = self._model(combined)  # (2N, dim)
features = (all_features[:N] + all_features[N:]) / 2.0
```

| 模型 | 10 帧 (concat 2N batch) | vs 无 flip |
|:--|:--:|:--:|
| SOLIDER Swin-B | ~70ms | +17% (非 2x，batch 并行效率) |
| AdaFace | ~24ms | +20% |

## 关键问题：多帧聚合是否使 flip-test 冗余？

flip-test 和多帧聚合都在做「增加鲁棒性」，但方向不同：

| 维度 | flip-test | 多帧聚合 |
|:--|:--|:--|
| 鲁棒性来源 | 同一帧的左右对称 | 不同时刻的不同姿态 |
| 对抗的问题 | 单帧姿态偏移/不对称噪声 | 时序抖动/偶发遮挡 |
| 边际收益递减 | 帧越多 → flip 边际收益越低 | flip 质量越高 → 需要的帧越少 |

**结论：有一定冗余，但不完全替代。**

- 当帧数充足 (≥5 帧/桶) 时，多帧聚合已经很鲁棒，flip-test 的 +0.5% 边际收益很小
- 当帧数稀少 (1-2 帧/桶，track 冷启动) 时，flip-test 的收益更明显

## 决策

**默认关闭，配置可选。**

```python
@dataclass
class MultiFrameConfig:
    use_flip_test: bool = False    # 默认关闭, 多帧聚合已提供足够鲁棒性
```

| 方案 | 适用场景 | 说明 |
|:--|:--|:--|
| **默认关闭** | 常规运行 | 多帧聚合已提供足够鲁棒性，省 ~20% GPU |
| **配置可选** | `use_flip_test: bool = False` | 保留代码实现，通过配置开关控制 |
| **自适应开启** (可选优化) | track 冷启动首次 Tier2 | 帧少时自动开启，帧多后自动关闭 |
