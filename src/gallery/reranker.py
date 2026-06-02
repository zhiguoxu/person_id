"""
K-Reciprocal Re-ranking — 重排序模块

基于 Zhong et al. "Re-ranking Person Re-identification with k-Reciprocal
Encoding" (CVPR 2017) 的思想, 利用 k-互近邻集合构建 Jaccard 距离,
与原始距离加权融合以提升 Re-ID 精度。

当底库人数 < 5 时, 邻域不足, 自动跳过重排序。
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.gallery.data_models import MatchCandidate


class KReciprocalReranker:
    """K-互近邻重排序器。

    适用于底库规模较大 (≥ 5 人) 的场景。
    对小库自动降级为透传 (pass-through)。
    """

    def __init__(
        self,
        k1: int = 20,
        k2: int = 6,
        lambda_value: float = 0.3,
    ) -> None:
        """初始化重排序器。

        Args:
            k1: 第一轮 k-互近邻集合大小。
            k2: 局部查询扩展的邻域大小。
            lambda_value: 原始距离权重 (0–1)。
                最终距离 = lambda * original + (1 - lambda) * Jaccard。
        """
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        logger.info(
            "KReciprocalReranker initialized (k1={}, k2={}, λ={:.2f})",
            k1, k2, lambda_value,
        )

    def rerank(
        self,
        query_embedding: np.ndarray,
        gallery_embeddings: list[tuple[str, np.ndarray]],
        initial_scores: list[MatchCandidate],
    ) -> list[MatchCandidate]:
        """执行 k-互近邻重排序。

        Args:
            query_embedding: 查询特征向量 (L2 归一化)。
            gallery_embeddings: 底库特征列表, 每项为 (person_id, embedding)。
            initial_scores: 初始匹配候选列表 (用于获取 display_name 等元信息)。

        Returns:
            重排序后的 ``MatchCandidate`` 列表, 按 ``fused_score`` 降序排列。
            若底库 < 5 人, 原样返回 ``initial_scores``。
        """
        n_gallery = len(gallery_embeddings)

        # 人数不足时跳过重排序
        if n_gallery < 5:
            logger.debug(
                "Gallery size {} < 5, skipping re-ranking", n_gallery
            )
            return initial_scores

        # 构建 person_id → 元信息 的映射
        meta_map: dict[str, MatchCandidate] = {
            c.person_id: c for c in initial_scores
        }

        # ------------------------------------------------------------------
        # 1. 构建距离矩阵 (query + gallery)
        # ------------------------------------------------------------------
        gallery_ids = [gid for gid, _ in gallery_embeddings]
        gallery_feats = np.array(
            [emb for _, emb in gallery_embeddings], dtype=np.float32
        )

        # query 放在第 0 位
        all_feats = np.vstack(
            [query_embedding.reshape(1, -1).astype(np.float32), gallery_feats]
        )
        n = all_feats.shape[0]

        # 余弦相似度 → 欧氏距离 (L2-normalized 时: d² = 2 - 2*cos)
        sim_matrix = all_feats @ all_feats.T
        np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
        dist_matrix = np.sqrt(np.maximum(2.0 - 2.0 * sim_matrix, 0.0))

        # ------------------------------------------------------------------
        # 2. 为每个样本构建 k-互近邻集合
        # ------------------------------------------------------------------
        k1_actual = min(self.k1, n - 1)
        k2_actual = min(self.k2, n - 1)

        # 排序索引 (按距离升序)
        sorted_indices = np.argsort(dist_matrix, axis=1)

        # k-近邻集合
        knn_sets: list[set[int]] = []
        for i in range(n):
            knn_sets.append(set(sorted_indices[i, 1: k1_actual + 1].tolist()))

        # 互近邻: R(i) = { j ∈ KNN(i) | i ∈ KNN(j) }
        reciprocal_sets: list[set[int]] = []
        for i in range(n):
            r_set = set()
            for j in knn_sets[i]:
                if i in knn_sets[j]:
                    r_set.add(j)
            reciprocal_sets.append(r_set)

        # 扩展互近邻 (1/2 条件)
        expanded_sets: list[set[int]] = []
        for i in range(n):
            expanded = set(reciprocal_sets[i])
            for j in list(reciprocal_sets[i]):
                rj = reciprocal_sets[j]
                # 如果 j 的互近邻有超过 2/3 与 i 的互近邻重叠, 则合并
                if len(rj & reciprocal_sets[i]) >= len(rj) * 2 / 3:
                    expanded |= rj
            expanded_sets.append(expanded)

        # ------------------------------------------------------------------
        # 3. 局部查询扩展 (Local Query Expansion)
        # ------------------------------------------------------------------
        V = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            r_set = expanded_sets[i]
            if not r_set:
                continue
            neighbors = sorted(r_set, key=lambda x: dist_matrix[i, x])[:k2_actual]
            weights = np.exp(-dist_matrix[i, neighbors])
            weight_sum = weights.sum()
            if weight_sum > 1e-8:
                weights /= weight_sum
            for idx, nb in enumerate(neighbors):
                V[i, nb] = weights[idx]

        # ------------------------------------------------------------------
        # 4. 计算 Jaccard 距离
        # ------------------------------------------------------------------
        # Jaccard 距离 between query (index 0) 和 gallery
        query_v = V[0]
        jaccard_dists = np.zeros(n_gallery, dtype=np.float32)

        for gi in range(n_gallery):
            gallery_v = V[gi + 1]  # gallery 从 index 1 开始
            min_v = np.minimum(query_v, gallery_v)
            max_v = np.maximum(query_v, gallery_v)
            max_sum = max_v.sum()
            if max_sum > 1e-8:
                jaccard_dists[gi] = 1.0 - min_v.sum() / max_sum
            else:
                jaccard_dists[gi] = 1.0

        # ------------------------------------------------------------------
        # 5. 融合距离: λ * original + (1-λ) * Jaccard
        # ------------------------------------------------------------------
        original_dists = dist_matrix[0, 1:]  # query → gallery
        final_dists = (
            self.lambda_value * original_dists
            + (1.0 - self.lambda_value) * jaccard_dists
        )

        # 距离 → 相似度 [0, 1]
        max_dist = final_dists.max() if final_dists.max() > 1e-8 else 1.0
        final_scores = 1.0 - final_dists / (max_dist + 1e-8)

        # ------------------------------------------------------------------
        # 6. 构建重排序结果
        # ------------------------------------------------------------------
        ranked_indices = np.argsort(final_dists)
        reranked: list[MatchCandidate] = []

        for idx in ranked_indices:
            pid = gallery_ids[idx]
            meta = meta_map.get(pid)
            if meta is None:
                continue
            reranked.append(
                MatchCandidate(
                    person_id=pid,
                    display_name=meta.display_name,
                    face_score=meta.face_score,
                    body_score=meta.body_score,
                    proportion_score=meta.proportion_score,
                    fused_score=float(final_scores[idx]),
                )
            )

        logger.debug(
            "Re-ranking complete: {} candidates processed", len(reranked)
        )
        return reranked
