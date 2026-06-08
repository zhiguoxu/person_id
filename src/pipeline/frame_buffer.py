"""
多帧处理 — 帧缓冲与质量缓存

提供 Tier1 帧收集 (RecentBuffer) 和 Tier2 质量缓存 (QualityCache) 组件:
- BufferEntry: 帧缓冲条目 (持有 crop 裁剪, 非整帧引用)
- RecentBuffer: Per-track 帧收集器 (时间窗口 + 质量竞争)
- CachedFrame: 质量缓存条目 (Tier2 质量评估后填充)
- QualityCache: Per-track 高质量帧缓存 (face/body 分离)
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_config
from src.gallery.data_models import FaceResult, PoseBucket


# ==============================================================================
# 帧缓冲 (Tier1 收集)
# ==============================================================================

class BufferEntry(BaseModel):
    """帧缓冲条目 — Tier1 每帧生成"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float  # time.monotonic()
    crop: np.ndarray  # BGR 人体裁剪 (bbox 区域拷贝, 非整帧引用)
    bbox: np.ndarray  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # (17, 3) COCO keypoints
    pose_bucket: PoseBucket  # 从 keypoints 快速分类
    quality_hint: float  # 轻量质量预估 (0-1), 窗口内竞争用


class RecentBuffer(BaseModel):
    """Per-track 帧收集器 — 时间窗口 + 质量竞争
    
    每 min_interval(0.25s) 为一个时间窗口, 窗口内仅保留 quality_hint 最高的帧.
    避免纯间隔丢帧 (= 降帧率), 保证窗口内最优帧不被浪费.
    """

    frames: list[BufferEntry] = Field(default_factory=list)
    min_interval: float = Field(
        default_factory=lambda: get_config().multiframe.recent_min_interval,
    )

    def push(self, entry: BufferEntry) -> bool:
        """Tier1 每帧调用 — 时间窗口 + 质量竞争
        
        前置校验: bbox 最小尺寸 (宽>=10, 高>=20), 保证下游 crop 合法.
        
        Returns:
            True 表示收集成功 (新增或替换).
            False 表示跳过 (bbox 太小 / 窗口内质量不够).
        """
        # bbox 最小尺寸校验 — 保证下游 crop 永远合法
        w = entry.bbox[2] - entry.bbox[0]
        h = entry.bbox[3] - entry.bbox[1]
        if w < 10 or h < 20:
            return False

        # 空 buffer: 直接入
        if not self.frames:
            self.frames.append(entry)
            return True

        last = self.frames[-1]
        elapsed = entry.timestamp - last.timestamp

        if elapsed >= self.min_interval:
            # 新时间窗口 → 追加
            self.frames.append(entry)
            return True
        else:
            # 同窗口 → 质量竞争: 高质量替换低质量
            if entry.quality_hint > last.quality_hint:
                self.frames[-1] = entry
                return True
            return False

    def drain(self) -> list[BufferEntry]:
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
    """质量缓存条目 — 由 Tier2 质量评估后填充"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry: BufferEntry  # 原始帧数据

    # 质量评分 (Tier2 批量计算)
    face_quality: float = 0.0  # QualityAssessor 精确人脸质量
    body_quality: float = 0.0  # quality_hint + sharpness 人体质量
    face_result: FaceResult | None = None  # SCRFD 检测 + ArcFace 嵌入结果，用来缓存 quality asses 的 embedding 副产品

    # 特征缓存 (入缓存后提取)
    face_embedding: np.ndarray | None = None  # AdaFace
    body_embedding: np.ndarray | None = None  # SOLIDER


class QualityCache(BaseModel):
    """Per-track 高质量帧缓存 — face/body 分离, 按质量竞争"""

    face_pool: list[CachedFrame] = Field(default_factory=list)
    body_pool: list[CachedFrame] = Field(default_factory=list)
    face_pool_size: int = Field(
        default_factory=lambda: get_config().multiframe.face_pool_size,
    )
    body_pool_size: int = Field(
        default_factory=lambda: get_config().multiframe.body_pool_size,
    )

    def try_add_face(self, frame: CachedFrame) -> bool:
        """尝试加入 face_pool, 返回 True 表示新入缓存"""
        if frame.face_result is None:
            return False
        return self._try_add(
            self.face_pool, self.face_pool_size,
            frame, key=lambda f: f.face_quality,
        )

    def try_add_body(self, frame: CachedFrame) -> bool:
        """尝试加入 body_pool, 返回 True 表示新入缓存"""
        return self._try_add(
            self.body_pool, self.body_pool_size,
            frame, key=lambda f: f.body_quality,
        )

    @staticmethod
    def _try_add(
            pool: list[CachedFrame],
            max_size: int,
            frame: CachedFrame,
            key: Callable[[CachedFrame], float],
    ) -> bool:
        """通用质量竞争入池: 未满直接加, 满了替换最差的。"""
        if len(pool) < max_size:
            pool.append(frame)
            pool.sort(key=key, reverse=True)
            return True

        if key(frame) > key(pool[-1]):
            pool[-1] = frame
            pool.sort(key=key, reverse=True)
            return True
        return False
