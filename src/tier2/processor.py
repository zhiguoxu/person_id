"""
Tier 2 流水线 — 深度身份识别（异步）

当 Tier 1 判定某追踪目标需要深度识别时触发:
1. 精确 YOLO11x-pose 检测 (ROI)
2. 姿态分类
3. 人脸特征提取 (非背面)
4. 全身 ReID 特征提取
5. 体型比例提取
6. 时序聚合
7. 底库匹配 (人脸 + 全身 + 体型)
8. 多模态融合
9. 歧义消解 (ReID 阶段)
10. 若 SUSPECTED / CONFLICT → VLM 仲裁
11. 返回最终 MatchResult

每个阶段记录耗时到 PipelineDebug。
"""
from __future__ import annotations

from loguru import logger

from src.gallery.data_models import PersonProfile
from src.pipeline.data_models import (
    IdentityStatus,
    MatchResult,
    PipelineDebug,
)
from src.pipeline.frame_buffer import RecentBuffer, QualityCache
from src.tier2.batch_extractor import BatchExtractor
from src.tier2.multi_frame_aggregator import MultiFrameAggregator
from src.gallery import matcher as gallery_matcher
from src.tier2 import resolver
from src.tier2 import multi_modal_fusion as fusion


# ---------------------------------------------------------------------------
# Tier 2 Processor
# ---------------------------------------------------------------------------

class Tier2Processor:
    """
    Tier 2 深度身份识别处理器。

    异步处理单个追踪目标的精确身份识别：从 ROI 切割开始
    到最终 MatchResult 输出，沿途在 PipelineDebug 中记录
    每个阶段的状态和耗时。

    所有子模块在构造时一次性创建，无需外部传入。
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
    ) -> tuple[MatchResult | None, PipelineDebug]:
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
        debug = PipelineDebug()
        t0 = _time.perf_counter()

        # --- 1. Drain buffer ---
        frames = buffer.drain()

        # --- 2. Batch quality assess + cache ---
        BatchExtractor.batch_quality_assess(frames, quality_cache)

        # --- 3. Incremental feature extraction ---
        n_new = BatchExtractor.extract_new_embeddings(quality_cache)

        # 两阶段控制: 无新 embedding → query 数据没变, 匹配结果不会变
        if n_new == 0:
            return None, debug

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

        debug.matching.status = "done"

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
        logger.info(
            "Tier2 multiframe: track={}, status={}, n_frames={}, n_new={}, {:.1f}ms",
            track_id, result.status.value, len(frames), n_new, elapsed,
        )

        return result, debug
