"""
Gallery Matcher — 全桶交叉匹配 + Top-K Blend

提供三种模态的底库匹配能力:
- 人脸匹配: 查询端全桶 × Gallery 端全桶交叉匹配, 桶内质量加权质心
- 全身匹配: Top-K Blend (姿态分桶 + 服装库 γ 提升)
- 体型比例匹配: 利用高斯核相似度
"""
from __future__ import annotations

import math
import time

import numpy as np

from src.config import get_config
from src.gallery.data_models import (
    BodyProportions,
    PersonProfile,
    PoseBucket,
)
from src.pipeline.data_models import MatchCandidate


# ------------------------------------------------------------------
# 人脸匹配
# ------------------------------------------------------------------

def match_face(query_faces: dict[PoseBucket, tuple[np.ndarray, float]],
               gallery: dict[str, PersonProfile]) -> list[MatchCandidate]:
    """人脸匹配 — 查询端全桶×Gallery端全桶交叉匹配:
    1. 获取每个人的 per-bucket 质心 (缓存, 仅 enroll 时重算)
    2. 查询端每个桶 × Gallery 每个桶: 纯 cos_sim
    3. 取所有 (query_bucket, gallery_bucket) 组合中的最大分
    """
    if not query_faces or not gallery:
        return []

    candidates = []

    for person_id, profile in gallery.items():
        centroids = profile.get_face_centroids()
        if not centroids:
            continue

        best_score = -1.0
        best_quality = 0.0

        for g_bucket, centroid in centroids.items():
            for q_pose, (q_emb, q_quality) in query_faces.items():
                cos_sim = float(np.dot(q_emb, centroid))
                if cos_sim > best_score:
                    best_score = cos_sim
                    best_quality = q_quality

        if best_score > 0.0:
            candidates.append(MatchCandidate(
                person_id=person_id,
                display_name=profile.display_name,
                face_score=best_score,
                face_match_quality=best_quality,
            ))

    candidates.sort(key=lambda c: c.face_score or 0.0, reverse=True)
    return candidates


# ------------------------------------------------------------------
# 全身匹配 (Top-K Blend + 衣橱 γ 提升)
# ------------------------------------------------------------------

def match_body(query_bodies: dict[PoseBucket, tuple[np.ndarray, float]],
               gallery: dict[str, PersonProfile]) -> list[MatchCandidate]:
    """人体匹配 — Top-K Blend (姿态分桶 + 服装库 γ 提升)

    与人脸不同:
    - Gallery 端不做桶内质心 (换装导致多峰)
    - 使用 Top-K Blend: 0.7 * peak + 0.3 * depth_avg
    - wardrobe 补充 γ 提升
    """
    if not query_bodies or not gallery:
        return []

    config = get_config().matching
    blend_alpha = config.blend_alpha
    cross_pose = config.cross_pose_discount
    gamma = config.wardrobe_boost_gamma
    candidates = []

    for person_id, profile in gallery.items():
        # --- Top-K Blend: (cos_sim, entry_quality, is_same_pose) ---
        all_pairs: list[tuple[float, float, bool]] = []
        pair_q_map: list[float] = []  # 对应的 query 桶质量

        for g_bucket, entries in profile.body_features.items():
            for entry in entries:
                for q_pose, (q_emb, q_quality) in query_bodies.items():
                    sim = float(np.dot(q_emb, entry.embedding))
                    same_pose = (q_pose == g_bucket)
                    all_pairs.append((sim, entry.quality_score, same_pose))
                    pair_q_map.append(q_quality)

        if not all_pairs:
            # 如果 body_features 为空, 尝试仅用 wardrobe
            wardrobe_score = _wardrobe_max_sim(query_bodies, profile)
            if wardrobe_score > 0:
                candidates.append(MatchCandidate(
                    person_id=person_id,
                    display_name=profile.display_name,
                    body_score=wardrobe_score,
                ))
            continue

        body_base = _topk_blend(all_pairs, blend_alpha, cross_pose)

        # --- wardrobe γ 有限提升: L1 + γ × max(0, L2 - L1) ---
        wardrobe_sim = _wardrobe_max_sim(query_bodies, profile)
        body_score = body_base + gamma * max(0.0, wardrobe_sim - body_base)

        # 记录产生 peak 的 query 桶质量
        if all_pairs:
            peak_idx = max(range(len(all_pairs)), key=lambda i: all_pairs[i][0])
            best_q_quality = pair_q_map[peak_idx]
        else:
            best_q_quality = 0.0

        candidates.append(MatchCandidate(
            person_id=person_id,
            display_name=profile.display_name,
            body_score=body_score,
            body_match_quality=best_q_quality,
        ))

    candidates.sort(key=lambda c: c.body_score or 0.0, reverse=True)
    return candidates


# ------------------------------------------------------------------
# 体型比例匹配
# ------------------------------------------------------------------

def match_proportions(
        proportions: BodyProportions,
        gallery: dict[str, PersonProfile],
) -> list[MatchCandidate]:
    """使用体型比例高斯核相似度进行匹配。

    Args:
        proportions: 查询目标的体型比例。
        gallery: 底库。

    Returns:
        按 ``proportion_score`` 降序排列的 ``MatchCandidate`` 列表。
    """
    if proportions is None or len(gallery) == 0:
        return []

    candidates: list[MatchCandidate] = []

    for person_id, profile in gallery.items():
        if profile.body_proportions is None:
            continue

        score = BodyProportions.similarity(proportions, profile.body_proportions)
        candidates.append(
            MatchCandidate(
                person_id=person_id,
                display_name=profile.display_name,
                proportion_score=score,
            )
        )

    candidates.sort(key=lambda c: c.proportion_score or 0.0, reverse=True)
    return candidates


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _topk_blend(pairs: list[tuple[float, float, bool]], alpha: float,
                cross_pose_discount: float = 0.7) -> float:
    """Quality-Weighted Top-K Blend

    Args:
        pairs: [(cos_sim, entry_quality, is_same_pose)]
        alpha: peak 权重 (1-α 为 depth 权重)
        cross_pose_discount: 跨姿态投票权折扣

    K = ceil(√N), N=1→K=1(退化为max), N=4→K=2, N=9→K=3
    depth = Σ(sim × vote_weight) / Σ(vote_weight)  (top-K by sim)
    vote_weight = quality × (1.0 if same_pose else discount)
    score = α × peak + (1-α) × depth
    """
    if not pairs:
        return 0.0

    N = len(pairs)
    peak = max(s for s, _, _ in pairs)

    K = max(1, math.ceil(math.sqrt(N)))
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    top_k = sorted_pairs[:K]

    weighted_sum = 0.0
    weight_total = 0.0
    for sim, quality, same_pose in top_k:
        vote_w = quality * (1.0 if same_pose else cross_pose_discount)
        weighted_sum += sim * vote_w
        weight_total += vote_w

    depth = weighted_sum / weight_total if weight_total > 0 else peak
    return alpha * peak + (1.0 - alpha) * depth


def _wardrobe_max_sim(query_bodies: dict[PoseBucket, tuple[np.ndarray, float]],
                      profile: PersonProfile) -> float:
    """服装库最大相似度 (recency 加权)"""
    now = time.time()
    best = 0.0
    for outfit in profile.wardrobe:
        recency = outfit.recency_weight(now)
        for _, (q_emb, _) in query_bodies.items():
            sim = float(np.dot(q_emb, outfit.body_embedding)) * recency
            best = max(best, sim)
    return best
