"""
Ambiguity Resolver (Tier 3) — VLM 阶段歧义消解

负责将 VLM 仲裁结果融合到当前的 MatchResult 中。
VLM 直接输出离散等级 (DEFINITE/CONFIDENT/SUSPECTED/STRANGER),
不再需要连续分数 + 阈值判定。
"""
from __future__ import annotations

from loguru import logger

from src.pipeline.data_models import IdentityStatus, MatchResult
from src.tier3.vlm_arbitrator import VLMResponse

_GRADE_TO_STATUS = {
    "DEFINITE": IdentityStatus.DEFINITE,
    "CONFIDENT": IdentityStatus.CONFIDENT,
    "SUSPECTED": IdentityStatus.SUSPECTED,
    "STRANGER": IdentityStatus.STRANGER,
}


def resolve_vlm(
        vlm_response: VLMResponse,
        match_result: MatchResult,
) -> MatchResult:
    """基于 VLM 仲裁结果进行歧义消解。

    VLM 直接输出离散等级和匹配的候选人 ID，无需阈值判定。

    Args:
        vlm_response: VLM 返回的结构化结果, 包含:
            - matched_candidate_id: str | None
            - grade: "DEFINITE" | "CONFIDENT" | "SUSPECTED" | "STRANGER"
            - reasoning: str
            - distinguishing_features: list[str]
        match_result: 当前匹配结果。

    Returns:
        更新了 status 和 best_match 的 MatchResult。
    """
    matched_id = vlm_response.matched_candidate_id
    grade = vlm_response.grade

    if not match_result.candidates:
        match_result.status = IdentityStatus.STRANGER
        match_result.best_match = None
        return match_result

    # 将等级映射为 IdentityStatus，无效等级降级为 STRANGER
    status = _GRADE_TO_STATUS.get(grade, IdentityStatus.STRANGER)

    # STRANGER: 直接返回
    if status == IdentityStatus.STRANGER:
        match_result.status = IdentityStatus.STRANGER
        match_result.best_match = None
        logger.info("VLM → STRANGER (reason: {})", vlm_response.reasoning[:80])
        return match_result

    # 在候选列表中查找 VLM 指认的人
    matched_candidate = None
    if matched_id:
        for c in match_result.candidates:
            if c.person_id == matched_id:
                matched_candidate = c
                break

    if not matched_candidate:
        # VLM 返回了一个不在候选列表中的 ID → 降级为 STRANGER
        match_result.status = IdentityStatus.STRANGER
        match_result.best_match = None
        logger.warning(
            "VLM matched_candidate_id='{}' not in candidates → STRANGER",
            matched_id,
        )
        return match_result

    match_result.status = status
    match_result.best_match = matched_candidate
    logger.info(
        "VLM → {}: {} (reason: {})",
        grade, matched_candidate.person_id, vlm_response.reasoning[:80],
    )

    return match_result
