"""
Multi-Modal Fusion — 自适应多模态融合

将人脸、全身 ReID、体型比例三种模态的匹配分数自适应融合:
    1. 根据各模态是否可用动态调整权重
    2. 根据人脸质量调整人脸权重
    3. 人脸捷径: 高质量人脸 + 高匹配分 → 跳过融合直接确认
    4. 合并各模态候选人的分数, 按融合分排序
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import MatchingConfig
from src.gallery.data_models import MatchCandidate


class MultiModalFusion:
    """自适应多模态融合器。

    根据每帧实际可用的模态和质量, 动态计算融合权重,
    将三种匹配信号合成为统一的 fused_score。
    """

    def __init__(self, config: MatchingConfig) -> None:
        self._config = config
        logger.info(
            "MultiModalFusion initialized (face={:.2f}, body={:.2f}, prop={:.2f})",
            config.face_base_weight,
            config.body_base_weight,
            config.proportion_base_weight,
        )

    def fuse(
        self,
        face_result: Optional[list[MatchCandidate]],
        body_result: Optional[list[MatchCandidate]],
        proportion_result: Optional[list[MatchCandidate]],
        face_quality: float = 0.0,
    ) -> list[MatchCandidate]:
        """自适应融合三种模态的匹配结果。

        流程:
            1. 检查人脸捷径条件
            2. 确定各模态可用性, 计算自适应权重
            3. 对每个候选人合并各模态分数
            4. 按 fused_score 降序排序

        Args:
            face_result: 人脸匹配候选列表 (可为 None 或空)。
            body_result: 全身匹配候选列表 (可为 None 或空)。
            proportion_result: 体型匹配候选列表 (可为 None 或空)。
            face_quality: 当前帧人脸质量 [0, 1], 用于权重调整。

        Returns:
            按 ``fused_score`` 降序排列的融合候选列表。
        """
        face_candidates = face_result or []
        body_candidates = body_result or []
        proportion_candidates = proportion_result or []

        # ------------------------------------------------------------------
        # 人脸捷径: 高质量正脸 + 高匹配分 → 直接返回
        # ------------------------------------------------------------------
        if self._check_face_shortcut(face_candidates, face_quality):
            top = face_candidates[0]
            logger.debug(
                "Face shortcut activated for {} (score={:.3f}, quality={:.3f})",
                top.person_id,
                top.face_score or 0.0,
                face_quality,
            )
            return [
                MatchCandidate(
                    person_id=top.person_id,
                    display_name=top.display_name,
                    face_score=top.face_score,
                    body_score=top.body_score,
                    proportion_score=top.proportion_score,
                    fused_score=top.face_score or 0.0,
                )
            ]

        # ------------------------------------------------------------------
        # 计算自适应权重
        # ------------------------------------------------------------------
        weights = self._compute_adaptive_weights(
            has_face=len(face_candidates) > 0,
            has_body=len(body_candidates) > 0,
            has_proportion=len(proportion_candidates) > 0,
            face_quality=face_quality,
        )

        # ------------------------------------------------------------------
        # 合并候选人分数
        # ------------------------------------------------------------------
        # 收集所有候选人 ID
        candidate_map: dict[str, MatchCandidate] = {}

        for c in face_candidates:
            if c.person_id not in candidate_map:
                candidate_map[c.person_id] = MatchCandidate(
                    person_id=c.person_id,
                    display_name=c.display_name,
                )
            candidate_map[c.person_id].face_score = c.face_score

        for c in body_candidates:
            if c.person_id not in candidate_map:
                candidate_map[c.person_id] = MatchCandidate(
                    person_id=c.person_id,
                    display_name=c.display_name,
                )
            candidate_map[c.person_id].body_score = c.body_score

        for c in proportion_candidates:
            if c.person_id not in candidate_map:
                candidate_map[c.person_id] = MatchCandidate(
                    person_id=c.person_id,
                    display_name=c.display_name,
                )
            candidate_map[c.person_id].proportion_score = c.proportion_score

        # 计算融合分
        w_face, w_body, w_prop = weights
        for candidate in candidate_map.values():
            score = 0.0
            if candidate.face_score is not None:
                score += w_face * candidate.face_score
            if candidate.body_score is not None:
                score += w_body * candidate.body_score
            if candidate.proportion_score is not None:
                score += w_prop * candidate.proportion_score
            candidate.fused_score = score

        # 降序排序
        fused = sorted(
            candidate_map.values(),
            key=lambda c: c.fused_score,
            reverse=True,
        )

        if fused:
            logger.debug(
                "Fusion result: top={} (fused={:.3f}, face={}, body={}, prop={}), "
                "weights=(f={:.2f}, b={:.2f}, p={:.2f})",
                fused[0].person_id,
                fused[0].fused_score,
                f"{fused[0].face_score:.3f}" if fused[0].face_score else "N/A",
                f"{fused[0].body_score:.3f}" if fused[0].body_score else "N/A",
                f"{fused[0].proportion_score:.3f}" if fused[0].proportion_score else "N/A",
                w_face, w_body, w_prop,
            )

        return fused

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _check_face_shortcut(
        self,
        face_candidates: list[MatchCandidate],
        face_quality: float,
    ) -> bool:
        """检查是否满足人脸捷径条件。

        条件:
            - face_quality > 0.7 (face_shortcut_quality)
            - face_score > 0.75 (face_shortcut_threshold)

        Args:
            face_candidates: 人脸匹配候选列表。
            face_quality: 当前帧人脸质量分。

        Returns:
            是否触发捷径。
        """
        if not face_candidates:
            return False

        top_face = face_candidates[0]
        top_score = top_face.face_score or 0.0

        return (
            face_quality > self._config.face_shortcut_quality
            and top_score > self._config.face_shortcut_threshold
        )

    def _compute_adaptive_weights(
        self,
        has_face: bool,
        has_body: bool,
        has_proportion: bool,
        face_quality: float,
    ) -> tuple[float, float, float]:
        """计算自适应融合权重。

        规则:
            1. 不可用的模态权重设为 0
            2. 人脸权重根据质量缩放
            3. 所有权重归一化使其和为 1

        Args:
            has_face: 人脸模态是否可用。
            has_body: 全身模态是否可用。
            has_proportion: 体型模态是否可用。
            face_quality: 人脸质量 [0, 1]。

        Returns:
            (face_weight, body_weight, proportion_weight), 和为 1.0。
        """
        w_face = self._config.face_base_weight if has_face else 0.0
        w_body = self._config.body_base_weight if has_body else 0.0
        w_prop = self._config.proportion_base_weight if has_proportion else 0.0

        # 根据人脸质量缩放人脸权重
        if has_face and face_quality > 0.0:
            # 高质量人脸 → 放大人脸权重; 低质量 → 缩小
            quality_factor = 0.5 + face_quality  # [0.5, 1.5]
            w_face *= quality_factor

        # 归一化
        total = w_face + w_body + w_prop
        if total < 1e-8:
            # 所有模态不可用, 回退到均匀分布
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        return (w_face / total, w_body / total, w_prop / total)
