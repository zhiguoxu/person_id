"""
Ambiguity Resolver (Tier 2) — ReID 阶段歧义消解

负责将匹配管线产出的 MatchResult 映射到明确的 IdentityStatus:
    DEFINITE   — 笃定: 唯一终态 (A 级)
    CONFIDENT  — 确定: 单人 ≥ B 且 margin 充足
    SUSPECTED  — 怀疑: C ≤ 最高 < B
    CONFLICT   — 冲突: 多人 ≥ B
    STRANGER   — 陌生: 所有候选 < C
"""
from __future__ import annotations

from loguru import logger

from src.config import get_config
from src.pipeline.data_models import IdentityStatus, MatchResult


def resolve_reid(match_result: MatchResult) -> MatchResult:
    """四级置信度判定 (无笃定陌生)"""
    candidates = match_result.candidates
    if not candidates:
        match_result.status = IdentityStatus.STRANGER
        return match_result

    top = match_result.top_score
    margin = match_result.margin
    cfg = get_config().matching

    # A 级: 笃定 — 唯一终态
    if top >= cfg.A_threshold and margin >= cfg.A_margin:
        match_result.status = IdentityStatus.DEFINITE
        match_result.best_match = candidates[0]
        logger.info("笃定 (DEFINITE): {} score={:.3f} margin={:.3f}",
                    candidates[0].display_name, top, margin)
        return match_result

    # B 级: 确定
    if top >= cfg.B_threshold:
        n_above_B = sum(1 for c in candidates if c.fused_score >= cfg.B_threshold)
        if n_above_B > 1:
            match_result.status = IdentityStatus.CONFLICT
            match_result.best_match = candidates[0]
            logger.info("冲突 (CONFLICT): {} candidates above B", n_above_B)
        elif margin >= cfg.B_margin:
            match_result.status = IdentityStatus.CONFIDENT
            match_result.best_match = candidates[0]
            logger.info("确定 (CONFIDENT): {} score={:.3f} margin={:.3f}",
                        candidates[0].display_name, top, margin)
        else:
            match_result.status = IdentityStatus.SUSPECTED
            match_result.best_match = candidates[0]
            logger.debug("降级为怀疑: margin={:.3f} < B_margin={:.3f}",
                         margin, cfg.B_margin)
        return match_result

    # C 级: 怀疑
    if top >= cfg.C_threshold:
        match_result.status = IdentityStatus.SUSPECTED
        match_result.best_match = candidates[0]
        return match_result

    # 低于 C: 陌生人
    match_result.status = IdentityStatus.STRANGER
    return match_result
