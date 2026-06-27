"""
Tier3VLMProcessor — VLM 仲裁处理器

当 Tier2 ReID 产生 SUSPECTED 或 CONFLICT 结果时, 使用 VLM (视觉语言模型)
进行高精度仲裁. 独立于 Tier2 处理流程, 异步执行.

职责:
  1. 从 QualityCache 选取最佳 crop 作为查询图片
  2. 从 Gallery 收集候选人的参考图片
  3. 调用 VLM API 进行身份仲裁
  4. 通过 IdentityResolver 解析 VLM 响应, 产出最终 MatchResult
"""
from __future__ import annotations


import cv2
import numpy as np
from loguru import logger

from src.tier3 import resolver, get_vlm_arbitrator

from src.gallery.data_models import PersonProfile
from src.pipeline.data_models import (
    MatchCandidate,
    MatchResult,
)
from src.pipeline.frame_buffer import QualityCache
from src.config import get_config


class Tier3VLMProcessor:
    """VLM 仲裁处理器。

    所有子模块在构造时一次性创建，无需外部传入。
    """

    @staticmethod
    def select_query_crop(quality_cache: QualityCache) -> np.ndarray | None:
        """从 QualityCache 选取最佳 crop 用于 VLM 询问

        优先选择 body_pool (全身图更有信息量), 降级到 face_pool.

        Args:
            quality_cache: TrackState 的 QualityCache

        Returns:
            BGR crop 或 None (无可用帧)
        """
        if quality_cache.body_pool:
            # 取质量最高的 body 帧
            best = max(quality_cache.body_pool, key=lambda cf: cf.quality)
            return best.entry.crop
        if quality_cache.face_pool:
            best = max(quality_cache.face_pool, key=lambda cf: cf.quality)
            return best.entry.crop
        return None

    @staticmethod
    def collect_candidate_images(
            candidates: list[MatchCandidate],
            gallery: dict[str, PersonProfile],
            max_candidates: int = 3,
    ) -> list[tuple[str, bytes]]:
        """从 Gallery 收集候选人的参考图片

        Args:
            candidates: MatchCandidate 列表 (Tier2 输出)
            gallery: 当前底库
            max_candidates: 最多取多少个候选人

        Returns:
            [(person_id, jpeg_bytes), ...] 列表
        """
        result: list[tuple[str, bytes]] = []
        for c in candidates[:max_candidates]:
            profile: PersonProfile | None = gallery.get(c.person_id)
            if not profile:
                continue
            # 尝试从人脸特征中取 source_image (注册时保存的缩略图)
            found: bool = False
            for bucket_entries in profile.face_features.values():
                for entry in bucket_entries:
                    if entry.source_image:
                        result.append((c.person_id, entry.source_image))
                        found = True
                        break
                if found:
                    break
        return result

    @staticmethod
    async def process(
            track_id: int,
            match_result: MatchResult,
            quality_cache: QualityCache,
            gallery: dict[str, PersonProfile],
    ) -> MatchResult | None:
        """异步执行 VLM 仲裁

        Args:
            track_id: 追踪 ID
            match_result: Tier2 产出的匹配结果 (含候选人列表)
            quality_cache: 该 track 的 QualityCache
            gallery: 当前底库

        Returns:
            VLM 仲裁后的 MatchResult, 或 None (无可用数据/失败)
        """

        # 1. 选取查询 crop
        crop: np.ndarray | None = Tier3VLMProcessor.select_query_crop(quality_cache)
        if crop is None:
            logger.warning("Tier3 VLM: track_id={} 没有 cache frame，中止", track_id)
            return None

        # 2. 编码为 JPEG
        _, query_jpeg = cv2.imencode('.jpg', crop)
        query_bytes: bytes = query_jpeg.tobytes()

        # 3. 收集候选人图片
        cfg = get_config()
        candidate_images: list[tuple[str, bytes]] = Tier3VLMProcessor.collect_candidate_images(
            match_result.candidates, gallery, max_candidates=cfg.vlm.max_candidates
        )

        # 4. 调用 VLM
        logger.info("Tier3 VLM: 正在 arbitrate track_id={}，共 {} 个 candidates",
                    track_id, len(candidate_images))
        vlm_response = await get_vlm_arbitrator().arbitrate(query_bytes, candidate_images)

        # 5. 解析 VLM 响应
        result: MatchResult = resolver.resolve_vlm(vlm_response, match_result)

        logger.info("Tier3 VLM: track_id={} → status={}", track_id, result.status.value)
        return result
