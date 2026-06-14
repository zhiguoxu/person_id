"""
多帧处理 — 帧缓冲与质量缓存

提供 Tier1 帧收集 (RecentBuffer) 和 Tier2 质量缓存 (QualityCache) 组件:
- RecentBuffer: Per-track 帧收集器 (时间窗口 + 质量竞争)
- CachedFrame: 质量缓存条目 (Tier2 质量评估后填充)
- QualityCache: Per-track 高质量帧缓存 (face/body 分离)

帧缓冲直接使用 TrackedPerson (包含 crop/face/quality/timestamp/frame_snapshot).
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_config
from src.pipeline.data_models import TrackedPerson


# ==============================================================================
# 帧缓冲 (Tier1 收集)
# ==============================================================================

class RecentBuffer(BaseModel):
    """Per-track 帧收集器 — 时间窗口 + 质量竞争
    
    每 min_interval(0.25s) 为一个时间窗口, 窗口内仅保留 quality_hint 最高的帧.
    避免纯间隔丢帧 (= 降帧率), 保证窗口内最优帧不被浪费.
    """

    frames: list[TrackedPerson] = Field(default_factory=list)

    def push(self, entry: TrackedPerson) -> bool:
        """Tier1 每帧调用 — 时间窗口 + 质量竞争
        
        前置校验: bbox 最小尺寸 (宽>=10, 高>=20), 保证下游 crop 合法.
        
        Returns:
            True 表示收集成功 (新增或替换).
            False 表示跳过 (bbox 太小 / 窗口内质量不够).
        """
        # bbox 最小尺寸校验 — 保证下游 crop 永远合法
        bbox = entry.detection.bbox
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w < 10 or h < 20:
            return False

        # 空 buffer: 直接入
        if not self.frames:
            self.frames.append(entry)
            return True

        last = self.frames[-1]
        elapsed = entry.timestamp - last.timestamp

        if elapsed >= get_config().multiframe.recent_min_interval:
            # 新时间窗口 → 追加
            self.frames.append(entry)
            return True
        else:
            # 同窗口 → 质量竞争: 高质量替换低质量
            if entry.combined_quality > last.combined_quality:
                self.frames[-1] = entry
                return True
            return False

    def drain(self) -> list[TrackedPerson]:
        """Tier2 调用: 取走所有帧, 清空"""
        result = list(self.frames)
        self.frames.clear()
        return result

    def __len__(self) -> int:
        return len(self.frames)


# ==============================================================================
# 质量缓存 (Tier2 使用)
# ==============================================================================

class CachedFrame(BaseModel):
    """质量缓存条目 — 由 Tier2 质量评估后填充

    face_pool 和 body_pool 共用同一结构。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry: TrackedPerson  # 原始帧数据
    quality: float = 0.0  # 质量评分
    embedding: np.ndarray | None = None  # 特征向量
    enrolled: bool = False  # 是否已入库 (防止重复入库)


class QualityCache(BaseModel):
    """Per-track 高质量帧缓存 — face/body 分离, 按质量竞争"""

    face_pool: list[CachedFrame] = Field(default_factory=list)
    body_pool: list[CachedFrame] = Field(default_factory=list)

    def clear(self) -> None:
        self.face_pool.clear()
        self.body_pool.clear()

    def try_add_face(self, frame: CachedFrame) -> bool:
        """尝试加入 face_pool, 返回 True 表示新入缓存"""
        return self._try_add(self.face_pool, get_config().multiframe.face_pool_size, frame)

    def try_add_body(self, frame: CachedFrame) -> bool:
        """尝试加入 body_pool, 返回 True 表示新入缓存"""
        return self._try_add(self.body_pool, get_config().multiframe.body_pool_size, frame)

    @staticmethod
    def _try_add(pool: list[CachedFrame], max_size: int, frame: CachedFrame) -> bool:
        """质量竞争入池: 未满直接加, 满了替换最差的。"""
        if len(pool) < max_size:
            pool.append(frame)
            pool.sort(key=lambda f: f.quality, reverse=True)
            return True

        if frame.quality > pool[-1].quality:
            pool[-1] = frame
            pool.sort(key=lambda f: f.quality, reverse=True)
            return True
        return False
