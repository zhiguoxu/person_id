"""
Multi-Modal Fusion — 自适应多模态融合

将人脸、全身 ReID、体型比例三种模态的匹配分数自适应融合:
    1. 根据各模态是否可用动态调整权重
    2. sigmoid 门控根据 body_quality 动态调整 face/body 权重
    3. 合并各模态候选人的分数, 按融合分排序
"""
from __future__ import annotations

import math

from loguru import logger

from src.config import get_config
from src.gallery.data_models import MatchCandidate

# sigmoid 门控参数
_SIGMOID_K = 10.0  # 斜率
_SIGMOID_Q0 = 0.5  # 翻转点


def fuse(
        face_candidates: list[MatchCandidate],
        body_candidates: list[MatchCandidate],
        proportion_candidates: list[MatchCandidate]
) -> list[MatchCandidate]:
    """Per-Candidate 三模态质量门控融合

    每个候选人的 face_match_quality / body_match_quality
    来自 matcher, 反映实际产生该分数的 query 桶质量.

    sigmoid 门控:
      gate(q) = 1 / (1 + exp(-k × (q - q0)))
      q=0.0 → gate≈0.007 (几乎关闭)
      q=0.5 → gate=0.50  (半开)
      q=0.7 → gate≈0.88  (基本开启)

    Args:
        face_candidates: 人脸匹配结果
        body_candidates: 人体匹配结果
        proportion_candidates: 体型匹配结果
    """

    if not face_candidates and not body_candidates and not proportion_candidates:
        return []

    cfg = get_config().matching

    # 合并所有候选人
    candidate_map: dict[str, MatchCandidate] = {}

    for c in face_candidates:
        candidate_map[c.person_id] = MatchCandidate(person_id=c.person_id,
                                                    display_name=c.display_name,
                                                    face_score=c.face_score,
                                                    face_match_quality=c.face_match_quality)

    for c in body_candidates:
        m = candidate_map.setdefault(c.person_id, MatchCandidate(
            person_id=c.person_id, display_name=c.display_name))
        m.body_score = c.body_score
        m.body_match_quality = c.body_match_quality

    for c in proportion_candidates:
        m = candidate_map.setdefault(c.person_id, MatchCandidate(
            person_id=c.person_id, display_name=c.display_name))
        m.proportion_score = c.proportion_score

    # Per-candidate 门控融合
    for c in candidate_map.values():
        f_quality = c.face_match_quality if c.face_match_quality else 0.0
        b_quality = c.body_match_quality if c.body_match_quality else 0.0

        face_gate = _sigmoid_gate(f_quality)
        body_gate = _sigmoid_gate(b_quality)

        w_face_raw = cfg.face_base_weight * face_gate if c.face_score is not None else 0.0
        w_body_raw = cfg.body_base_weight * body_gate if c.body_score is not None else 0.0
        w_prop_raw = cfg.proportion_base_weight if c.proportion_score is not None else 0.0

        w_total = w_face_raw + w_body_raw + w_prop_raw
        if w_total < 1e-8:
            c.fused_score = 0.0
            continue

        w_face = w_face_raw / w_total
        w_body = w_body_raw / w_total
        w_prop = w_prop_raw / w_total

        score = 0.0
        if c.face_score is not None:
            score += w_face * c.face_score
        if c.body_score is not None:
            score += w_body * c.body_score
        if c.proportion_score is not None:
            score += w_prop * c.proportion_score
        c.fused_score = score

    result = sorted(candidate_map.values(),
                    key=lambda c: c.fused_score, reverse=True)

    if result:
        top = result[0]
        logger.debug("Fusion: per-candidate gate, "
                     "top={:.3f} (face_q={:.2f}, body_q={:.2f})",
                     top.fused_score,
                     top.face_match_quality or 0,
                     top.body_match_quality or 0)
    return result


def _sigmoid_gate(quality: float) -> float:
    """quality → gate value (0~1)"""
    return 1.0 / (1.0 + math.exp(-_SIGMOID_K * (quality - _SIGMOID_Q0)))
