"""
Tier2 批量质量评估与增量特征提取

两步走:
1. _batch_quality_assess(): 批量 SCRFD + FaceQualityAssessor + compute_sharpness
2. _extract_new_embeddings(): 对 is_extracted=False 的帧提取 AdaFace/SOLIDER
"""
from __future__ import annotations


import numpy as np
from loguru import logger

from src.pipeline.frame_buffer import BufferEntry, CachedFrame, QualityCache
from src.pipeline.quality_utils import compute_sharpness
from src.gallery.data_models import PoseBucket
from src.tier2.features import get_face_extractor, get_body_extractor, get_quality_assessor


class BatchExtractor:
    """Tier2 批量处理器 — 质量评估 + 增量特征提取"""

    @staticmethod
    def batch_quality_assess(new_frames: list[BufferEntry],
                             quality_cache: QualityCache) -> list[CachedFrame]:
        """对 RecentBuffer drain 出的新帧做批量质量评估, 并竞争入缓存
        
        1. 构造 CachedFrame + body_quality (CPU, 极快)
        2. 批量 SCRFD 人脸检测 (GPU, 主要耗时)  
        3. FaceQualityAssessor 精确评估 (CPU, 逐帧)
        4. 竞争入 QualityCache (face_pool + body_pool)
        
        Args:
            new_frames: RecentBuffer.drain() 返回的帧列表
            quality_cache: 该 track 的 QualityCache (会被原地修改)
        
        Returns:
            CachedFrame 列表, 已填充 face_quality/body_quality/face_det
        """
        if not new_frames:
            return []

        # 1. 构造 CachedFrame + body_quality
        cached_frames = []
        for entry in new_frames:
            cf = CachedFrame(entry=entry)
            cf.body_quality = 0.75 * entry.quality_hint + 0.25 * compute_sharpness(entry.crop)
            cached_frames.append(cf)

        # 2. 批量人脸检测 (跳过 BACK 姿态)
        face_indices = [i for i, cf in enumerate(cached_frames)
                        if cf.entry.pose_bucket != PoseBucket.BACK]

        if face_indices:
            for i in face_indices:
                cf = cached_frames[i]
                face_result = get_face_extractor().extract(
                    cf.entry.crop,
                    np.array([0, 0, cf.entry.crop.shape[1], cf.entry.crop.shape[0]])
                )
                if face_result is not None:
                    cf.face_det = face_result
                    cf.face_quality = get_quality_assessor().assess(
                        cf.entry.crop,
                        getattr(face_result, 'landmarks', None),
                        getattr(face_result, 'bbox', None),
                        cf.entry.keypoints
                    )

        # 4. 竞争入 QualityCache
        for cf in cached_frames:
            quality_cache.try_add_face(cf)
            quality_cache.try_add_body(cf)

        return cached_frames

    @staticmethod
    def extract_new_embeddings(cache: QualityCache) -> int:
        """对 QualityCache 中 is_extracted=False 的帧批量提取 embedding
        
        Returns:
            新提取的 embedding 数量
        """
        # 合并两个 pool, 去重 (同一 CachedFrame 可能同时在 face_pool 和 body_pool)
        pending = list({id(cf): cf for cf in cache.face_pool + cache.body_pool
                        if not cf.is_extracted}.values())
        if not pending:
            return 0

        # Batch body extraction
        body_crops = []
        body_indices = []
        for i, cf in enumerate(pending):
            if cf.body_embedding is None:
                body_crops.append(cf.entry.crop)
                body_indices.append(i)

        if body_crops:
            body_embs = get_body_extractor().extract_batch(body_crops)
            for idx, emb in zip(body_indices, body_embs):
                if emb is not None:
                    pending[idx].body_embedding = emb

        # Face embedding: 从 face_det 获取或主动提取

        face_pending = [cf for cf in pending if cf.face_det is not None and cf.face_embedding is None]
        if face_pending:
            for cf in face_pending:
                existing = getattr(cf.face_det, 'embedding', None)
                if existing is not None:
                    cf.face_embedding = existing

            still_pending = [cf for cf in face_pending if cf.face_embedding is None]
            for cf in still_pending:
                face_bbox = getattr(cf.face_det, 'bbox', None)
                if face_bbox is not None:
                    emb = get_face_extractor().extract(cf.entry.crop, face_bbox)
                    if emb is not None:
                        cf.face_embedding = getattr(emb, 'embedding', emb)

        # 标记为已提取
        for cf in pending:
            cf.is_extracted = True

        logger.debug("Extracted embeddings for {} frames ({} with face)",
                     len(pending),
                     sum(1 for cf in pending if cf.face_embedding is not None))

        return len(pending)
