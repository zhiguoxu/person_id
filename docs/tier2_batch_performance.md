# Tier2 批量质量评估性能分析

> 分析 `_batch_quality_assess` 在不同帧数下的耗时，评估 SCRFD batch 优化和 Tier1 过滤的各自收益。

## 帧数对比

| 场景 | 无 Tier1 过滤 | 有 Tier1 过滤 (0.25s 窗口) |
|:--|:--:|:--:|
| 1s 间隔, 10 FPS | **10 帧** | ~4 帧 |
| 5s 间隔, 10 FPS | **50 帧** | ~20 帧 |

> Tier1 的 `quality_hint` 窗口竞争以 0.25s 为周期，10 FPS 下每窗口 ~2.5 帧，只保留最优 1 帧，过滤率 ~60%。

## `_batch_quality_assess` 各操作耗时

### 1s 间隔 (IDENTIFYING/SUSPECTED/CONFLICT 基准间隔)

| 操作 | 类型 | 10 帧 (无过滤) | 4 帧 (有过滤) |
|:--|:--|:--:|:--:|
| crop + sharpness | CPU 串行 | 5ms | 2ms |
| SCRFD 串行 | GPU | 9ms (6帧×1.5ms) | 3.6ms |
| **SCRFD batch** | **GPU** | **~3ms** | **~2ms** |
| face_qa | CPU 串行 | 1.5ms | 0.6ms |
| **总计 (串行 SCRFD)** | | **~16ms** | **~6ms** |
| **总计 (batch SCRFD)** | | **~10ms** | **~5ms** |

> 1s 间隔下，无论是否 batch，耗时都在合理范围内。

### 5s 间隔 (CONFIDENT/STRANGER 基准间隔) — 差距显著

| 操作 | 类型 | 50 帧 (无过滤) | 20 帧 (有过滤) |
|:--|:--|:--:|:--:|
| crop + sharpness | CPU 串行 | **25ms** ← 瓶颈转移 | 10ms |
| SCRFD 串行 | GPU | 45ms (30帧×1.5ms) | 18ms |
| **SCRFD batch** | **GPU** | **~10ms** | **~5ms** |
| face_qa | CPU 串行 | 6ms | 2.5ms |
| **总计 (串行 SCRFD)** | | **~77ms** | **~31ms** |
| **总计 (batch SCRFD)** | | **~42ms** | **~18ms** |

> 50 帧时 SCRFD batch 从 45ms → 10ms，节省 35ms；但 CPU sharpness (25ms) 成为新瓶颈。

## 全 Tier2 链路耗时 (5s 间隔)

| 阶段 | 50 帧 (无过滤) | 20 帧 (有过滤) |
|:--|:--:|:--:|
| `_batch_quality_assess` | 42ms | 18ms |
| `QualityCache` 竞争入缓存 | <0.1ms | <0.1ms |
| `_extract_new_embeddings` (上限 ~5帧) | ~20ms | ~20ms |
| 聚合 + 匹配 + 决策 | ~7ms | ~7ms |
| **总计** | **~69ms** | **~45ms** |

> `_extract_new_embeddings` 不受输入帧数影响——受 `pool_size`（face 10 + body 10）约束，每次最多提取 ~5 帧新入缓存帧。

## 关键结论

### 1. SCRFD batch 优化 GPU 瓶颈

| 帧数 | 串行 | batch | 节省 |
|:--|:--:|:--:|:--:|
| 10 帧 | 9ms | 3ms | **-6ms (67%)** |
| 50 帧 | 45ms | 10ms | **-35ms (78%)** |

帧数越多，batch 收益越大。GPU kernel launch 开销被摊薄。

### 2. Tier1 过滤优化 CPU 瓶颈

SCRFD batch 化后，CPU 串行的 `compute_sharpness`（Laplacian 方差）成为新瓶颈。Tier1 过滤将帧数从 50 → 20，直接将 sharpness 耗时从 25ms → 10ms。

| | 无 Tier1 过滤 | 有 Tier1 过滤 | 节省 |
|:--|:--:|:--:|:--:|
| sharpness 耗时 | 25ms | 10ms | **-15ms (60%)** |
| 全链路耗时 | 69ms | 45ms | **-24ms (35%)** |

### 3. 两者互补，缺一不可

```
瓶颈分析:
  无 batch + 无过滤: SCRFD 是瓶颈 (45ms)     → 总计 ~77ms
  有 batch + 无过滤: sharpness 是瓶颈 (25ms)  → 总计 ~42ms  ← batch 解决了 GPU 瓶颈
  无 batch + 有过滤: SCRFD 仍是瓶颈 (18ms)    → 总计 ~31ms  ← 过滤减少了帧数
  有 batch + 有过滤: 无明显瓶颈               → 总计 ~18ms  ← 最优
```

**SCRFD batch 优化 GPU 瓶颈，Tier1 过滤优化 CPU 瓶颈。两者作用于不同瓶颈，组合效果最佳。**
