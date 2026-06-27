"""
Tier2 批量质量评估与增量特征提取

两步走:
1. _batch_quality_assess(): 质量评估 (人脸检测+质量已在 Tier1 完成)
2. _extract_new_embeddings(): 对 embedding=None 的帧提取人脸(ArcFace/AdaFace)/SOLIDER 全身特征
"""
from __future__ import annotations


import numpy as np
from loguru import logger


from src.pipeline.frame_buffer import CachedFrame, QualityCache
from src.pipeline.data_models import TrackedPerson
from src.pipeline.quality_utils import compute_sharpness
from src.tier2.features import get_face_extractor, get_body_extractor


class BatchExtractor:
    """Tier2 批量处理器 — 质量评估 + 增量特征提取"""

    @staticmethod
    def batch_quality_assess(new_frames: list[TrackedPerson],
                             quality_cache: QualityCache) -> None:
        """对 RecentBuffer drain 出的新帧逐帧质量评估并竞争入缓存。

        每帧流程:
        1. 构造 CachedFrame + quality (CPU)
        2. 人脸: 直接从 entry 复制 (检测+质量已在 Tier1 完成)
        3. 竞争入 QualityCache (face_pool + body_pool)

        Args:
            new_frames: RecentBuffer.drain() 返回的帧列表
            quality_cache: 该 track 的 QualityCache (会被原地修改)
        """
        if not new_frames:
            return

        for entry in new_frames:
            body_cf = CachedFrame(entry=entry)
            body_cf.quality = 0.75 * entry.quality_hint + 0.25 * compute_sharpness(entry.crop)
            # 竞争入缓存
            quality_cache.try_add_body(body_cf)

            # Face: 直接从 entry 复制 (检测+质量已在 Tier1 完成)
            if entry.aligned_face is not None:
                face_cf = CachedFrame(entry=entry)
                face_cf.quality = entry.face_quality
                quality_cache.try_add_face(face_cf)


    @staticmethod
    def extract_new_embeddings(cache: QualityCache) -> int:
        """对 QualityCache 中新入缓存的帧批量提取 embedding

        Returns:
            新提取的 embedding 数量
        """

        # Body embedding: 仅处理 body_pool 中未提取的帧
        body_pending = [cf for cf in cache.body_pool if cf.embedding is None]
        if body_pending:
            body_crops = [cf.entry.crop for cf in body_pending]
            body_embs = get_body_extractor().extract_batch(body_crops)
            if body_embs is None:
                # 整批提取失败: 直接从 body_pool 移除这些帧 (按身份, 原地过滤),
                # 避免残留 embedding=None 的帧污染后续入库/匹配; 本轮计数归零。
                pending_ids = {id(cf) for cf in body_pending}
                cache.body_pool[:] = [
                    cf for cf in cache.body_pool if id(cf) not in pending_ids
                ]
                body_pending = []
            else:
                for cf, emb in zip(body_pending, body_embs):
                    cf.embedding = emb

        # Face embedding: 从 Tier1 对齐人脸直接提取 (ArcFace/AdaFace, 无需重跑 SCRFD)
        # 逐帧提取: 失败的帧从 face_pool 移除 (与 body 一致), 不残留 embedding=None。
        face_pending = [cf for cf in cache.face_pool if cf.embedding is None]
        if face_pending:
            face_extractor = get_face_extractor()
            face_failed = []
            for cf in face_pending:
                embedding = face_extractor.extract_embedding(cf.entry.aligned_face)
                if embedding is not None:
                    cf.embedding = embedding
                else:
                    face_failed.append(cf)  # 提取失败, 待移除
            if face_failed:
                failed_ids = {id(cf) for cf in face_failed}
                cache.face_pool[:] = [
                    cf for cf in cache.face_pool if id(cf) not in failed_ids
                ]
                # 计数只算成功提取的帧
                face_pending = [cf for cf in face_pending if cf.embedding is not None]

        if body_pending or face_pending:
            logger.debug("已提取 {} 个 body frame 和 {} 个 face frame 的 embedding",
                         len(body_pending), len(face_pending))

        return len(body_pending) + len(face_pending)
