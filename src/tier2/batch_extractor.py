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
from src.tier2.features import get_face_extractor, get_body_extractor
from src.tier2.features.face_quality_assessor import FaceQualityAssessor


class BatchExtractor:
    """Tier2 批量处理器 — 质量评估 + 增量特征提取"""

    @staticmethod
    def batch_quality_assess(new_frames: list[BufferEntry],
                             quality_cache: QualityCache) -> None:
        """对 RecentBuffer drain 出的新帧逐帧质量评估并竞争入缓存。

        每帧流程:
        1. 构造 CachedFrame + quality (CPU)
        2. SCRFD 人脸检测 + FaceQualityAssessor 精确评估
        3. 竞争入 QualityCache (face_pool + body_pool)

        Args:
            new_frames: RecentBuffer.drain() 返回的帧列表
            quality_cache: 该 track 的 QualityCache (会被原地修改)
        """
        if not new_frames:
            return

        face_extractor = get_face_extractor()

        for entry in new_frames:
            body_cf = CachedFrame(entry=entry)
            body_cf.quality = 0.75 * entry.quality_hint + 0.25 * compute_sharpness(entry.crop)
            # 竞争入缓存
            quality_cache.try_add_body(body_cf)

            # 人脸检测 (跳过 BACK 姿态)
            if entry.pose_bucket != PoseBucket.BACK:
                face_result = face_extractor.extract(
                    entry.crop,
                    np.array([0, 0, entry.crop.shape[1], entry.crop.shape[0]]),
                )
                if face_result is not None:
                    face_cf = CachedFrame(entry=entry)
                    face_cf.face_result = face_result
                    face_cf.quality = FaceQualityAssessor.assess(
                        entry.crop,
                        face_result.landmarks,
                        face_result.bbox,
                        entry.keypoints,
                    )
                    face_cf.embedding = face_result.embedding
                    quality_cache.try_add_face(face_cf)


    @staticmethod
    def extract_new_embeddings(cache: QualityCache) -> int:
        """对 QualityCache 中新入缓存的帧批量提取 embedding

        Returns:
            新提取的 body embedding 数量
        """

        # Body embedding: 仅处理 body_pool 中未提取的帧
        body_pending = [cf for cf in cache.body_pool if cf.embedding is None]
        if body_pending:
            body_crops = [cf.entry.crop for cf in body_pending]
            body_embs = get_body_extractor().extract_batch(body_crops)
            for cf, emb in zip(body_pending, body_embs):
                cf.embedding = emb

        # Face embedding: 仅处理 face_pool 中未提取的帧 (直接从 face_result 拷贝)
        face_pending = [cf for cf in cache.face_pool if cf.embedding is None]
        for cf in face_pending:
            cf.embedding = cf.face_result.embedding

        if body_pending or face_pending:
            logger.debug("Extracted embeddings for {} body frames and {} face frames",
                         len(body_pending), len(face_pending))

        return len(body_pending) + len(face_pending)
