"""
Tier 2 流水线 — 深度身份识别（异步）

当调度器判定某追踪目标需要深度识别时触发, 基于 Tier1 缓存的多帧数据:
1. drain RecentBuffer 取新帧
2. 批量质量评估, 竞争进入 QualityCache
3. 增量特征提取 (人脸 + 全身 ReID + 体型比例, 仅新入缓存帧)
4. 多帧聚合
5. 底库匹配 (分模态) + 多模态融合
6. 歧义消解, 返回最终 MatchResult (SUSPECTED/CONFLICT 由上层触发 VLM 仲裁)
"""
from __future__ import annotations

from loguru import logger

from src.gallery.data_models import PersonProfile
from src.pipeline.data_models import (
    IdentityStatus,
    MatchResult,
)
from src.pipeline.frame_buffer import RecentBuffer, QualityCache
from src.tier2.batch_extractor import BatchExtractor
from src.tier2.multi_frame_aggregator import MultiFrameAggregator
from src.gallery import matcher as gallery_matcher
from src.tier2 import resolver
from src.tier2 import multi_modal_fusion as fusion
from src.config import get_config


# ---------------------------------------------------------------------------
# Tier 2 Processor
# ---------------------------------------------------------------------------

class Tier2Processor:
    """
    Tier 2 深度身份识别处理器。

    异步处理单个追踪目标的精确身份识别: 从 Tier1 缓存的多帧数据
    到最终 MatchResult 输出。

    所有子模块均为静态调用，无实例状态。
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def process_multiframe(
            track_id: int,
            buffer: RecentBuffer,
            quality_cache: QualityCache,
            gallery: dict[str, PersonProfile]
    ) -> MatchResult | None:
        """多帧 Tier2 处理。

        新的处理流程:
        1. drain RecentBuffer → 新帧列表
        2. 批量质量评估 → QualityCache 竞争入缓存
        3. 增量特征提取 (仅新入缓存帧)
        4. 多帧聚合
        5. Gallery 匹配 + 融合 + 决策
        6. 按需 Gallery 更新

        Args:
            track_id: 追踪 ID
            buffer: 该 track 的 RecentBuffer
            quality_cache: 该 track 的 QualityCache
            gallery: 当前底库
        """
        import time as _time
        t0 = _time.perf_counter()

        # --- 1. Drain buffer ---
        frames = buffer.drain()

        # --- 2. Batch quality assess + cache ---
        BatchExtractor.batch_quality_assess(frames, quality_cache)

        # --- 3. Incremental feature extraction ---
        n_new = BatchExtractor.extract_new_embeddings(quality_cache)

        # 两阶段控制: 无新 embedding → query 数据没变, 匹配结果不会变
        if n_new == 0:
            return MatchResult(stale=True)

        # --- 4. Multi-frame aggregation ---
        aggregated = MultiFrameAggregator.aggregate_from_cache(quality_cache)

        # --- 5. Gallery matching (分模态) ---
        face_candidates = gallery_matcher.match_face(
            aggregated.face_per_pose, gallery
        )
        body_candidates = gallery_matcher.match_body(
            aggregated.body_per_pose, gallery
        )
        proportion_candidates = gallery_matcher.match_proportions(
            aggregated.proportions, gallery
        )

        # --- 6. Fusion ---
        candidates = fusion.fuse(
            face_candidates, body_candidates, proportion_candidates,
        )

        # --- 7. Identity resolution ---
        match_result = MatchResult(
            candidates=candidates,
            best_match=candidates[0] if candidates else None,
            status=IdentityStatus.IDENTIFYING,
        )
        result = resolver.resolve_reid(match_result)

        elapsed = (_time.perf_counter() - t0) * 1000

        # --- Diagnostic logging ---
        cfg = get_config().matching
        top3 = candidates[:3]
        top3_info = " | ".join(
            f"{c.display_name}(fused={c.fused_score:.3f}, "
            f"face={c.face_score or 0:.3f}[q={c.face_match_quality:.2f}], "
            f"body={c.body_score or 0:.3f}[q={c.body_match_quality:.2f}], "
            f"prop={c.proportion_score or 0:.3f})"
            for c in top3
        ) if top3 else "(no candidates)"

        logger.info(
            "Tier2 track={} | gallery={} | face_cand={} body_cand={} prop_cand={} "
            "| fused_top3: [{}] | status={} "
            "| thresholds A={:.2f} B={:.2f} C={:.2f} | {:.1f}ms",
            track_id, len(gallery),
            len(face_candidates), len(body_candidates), len(proportion_candidates),
            top3_info, result.status.value,
            cfg.A_threshold, cfg.B_threshold, cfg.C_threshold,
            elapsed,
        )

        return result
