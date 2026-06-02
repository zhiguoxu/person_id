"""
Ambiguity Resolver — 歧义消解 (四重阈值体系)

负责将匹配管线产出的 MatchResult 映射到明确的 IdentityStatus:
    CONFIDENT  — 唯一候选 ≥ X 且远超第二名
    SUSPECTED  — 最高分在 [Y, X) 区间
    CONFLICT   — 多人 ≥ X 或差距不足
    STRANGER   — 所有候选 < Y

支持两阶段:
    1. resolve_reid — 基于 ReID 融合分
    2. resolve_vlm  — 基于 VLM 仲裁结果

额外提供 face_shortcut 快速通道。
"""
from __future__ import annotations

from loguru import logger

from src.config import Config
from src.gallery.data_models import IdentityStatus, MatchCandidate, MatchResult


class AmbiguityResolver:
    """歧义消解器 — 四重阈值体系。

    根据匹配分数和差距, 将候选人列表映射为
    CONFIDENT / SUSPECTED / CONFLICT / STRANGER 四种状态。
    """

    def __init__(self, config: Config) -> None:
        self._matching_cfg = config.matching
        logger.info(
            "AmbiguityResolver initialized "
            "(X_reid={:.2f}, Y_reid={:.2f}, X_vlm={:.2f}, Y_vlm={:.2f})",
            self._matching_cfg.reid_confident_threshold,
            self._matching_cfg.reid_suspected_threshold,
            self._matching_cfg.vlm_confident_threshold,
            self._matching_cfg.vlm_suspected_threshold,
        )

    # ------------------------------------------------------------------
    # ReID 阶段消解
    # ------------------------------------------------------------------

    def resolve_reid(self, match_result: MatchResult) -> MatchResult:
        """基于 ReID 融合分进行歧义消解。

        规则:
            1. 无候选 → STRANGER
            2. 最高分 < Y_reid → STRANGER
            3. 最高分 ≥ X_reid 且差距 ≥ confident_margin → CONFIDENT
            4. 多人 ≥ X_reid 或差距不足 → CONFLICT
            5. Y_reid ≤ 最高分 < X_reid → SUSPECTED

        Args:
            match_result: 融合后的匹配结果 (candidates 按 fused_score 降序)。

        Returns:
            更新了 status 和 best_match 的 MatchResult。
        """
        x_reid = self._matching_cfg.reid_confident_threshold
        y_reid = self._matching_cfg.reid_suspected_threshold
        margin = self._matching_cfg.confident_margin

        return self._apply_thresholds(
            match_result,
            x_threshold=x_reid,
            y_threshold=y_reid,
            required_margin=margin,
            stage="reid",
        )

    # ------------------------------------------------------------------
    # VLM 阶段消解
    # ------------------------------------------------------------------

    def resolve_vlm(
        self,
        vlm_response: dict,
        match_result: MatchResult,
    ) -> MatchResult:
        """基于 VLM 仲裁结果进行歧义消解。

        VLM 返回的 confidence 替代原始融合分作为判定依据。

        Args:
            vlm_response: VLM 返回的 JSON, 包含:
                - is_same_person: bool
                - confidence: float [0, 1]
                - reasoning: str
                - distinguishing_features: list[str]
            match_result: 当前匹配结果。

        Returns:
            更新了 status 和 best_match 的 MatchResult。
        """
        x_vlm = self._matching_cfg.vlm_confident_threshold
        y_vlm = self._matching_cfg.vlm_suspected_threshold

        is_same = vlm_response.get("is_same_person", False)
        vlm_confidence = vlm_response.get("confidence", 0.0)

        if not match_result.candidates:
            match_result.status = IdentityStatus.STRANGER
            match_result.best_match = None
            return match_result

        # VLM 判定为不同人 → 降级
        if not is_same:
            if vlm_confidence >= x_vlm:
                # VLM 高置信否定 → 陌生人
                match_result.status = IdentityStatus.STRANGER
                match_result.best_match = None
                logger.info(
                    "VLM confidently rejected match (conf={:.3f})", vlm_confidence
                )
            else:
                # VLM 不确定的否定 → 保留疑似
                match_result.status = IdentityStatus.SUSPECTED
                match_result.best_match = match_result.candidates[0]
                logger.info(
                    "VLM uncertain rejection (conf={:.3f}), keeping SUSPECTED",
                    vlm_confidence,
                )
            return match_result

        # VLM 判定为同一人
        if vlm_confidence >= x_vlm:
            match_result.status = IdentityStatus.CONFIDENT
            match_result.best_match = match_result.candidates[0]
            logger.info(
                "VLM confirmed identity: {} (conf={:.3f})",
                match_result.best_match.person_id,
                vlm_confidence,
            )
        elif vlm_confidence >= y_vlm:
            match_result.status = IdentityStatus.SUSPECTED
            match_result.best_match = match_result.candidates[0]
            logger.info(
                "VLM suspected identity: {} (conf={:.3f})",
                match_result.best_match.person_id,
                vlm_confidence,
            )
        else:
            match_result.status = IdentityStatus.STRANGER
            match_result.best_match = None
            logger.info(
                "VLM low confidence (conf={:.3f}), marking STRANGER",
                vlm_confidence,
            )

        return match_result

    # ------------------------------------------------------------------
    # 人脸快速通道
    # ------------------------------------------------------------------

    def check_face_shortcut(
        self, face_score: float, face_quality: float
    ) -> bool:
        """检查是否满足人脸快速通道条件。

        当人脸质量和匹配分数均很高时, 可跳过多模态融合直接确认。

        Args:
            face_score: 人脸匹配分 [0, 1]。
            face_quality: 人脸质量分 [0, 1]。

        Returns:
            True 表示可以直接确认身份。
        """
        quality_ok = face_quality > self._matching_cfg.face_shortcut_quality
        score_ok = face_score > self._matching_cfg.face_shortcut_threshold

        if quality_ok and score_ok:
            logger.debug(
                "Face shortcut triggered (quality={:.3f}, score={:.3f})",
                face_quality,
                face_score,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # 内部通用阈值逻辑
    # ------------------------------------------------------------------

    def _apply_thresholds(
        self,
        match_result: MatchResult,
        x_threshold: float,
        y_threshold: float,
        required_margin: float,
        stage: str,
    ) -> MatchResult:
        """通用四重阈值判定。

        Args:
            match_result: 当前匹配结果。
            x_threshold: 确信阈值 X。
            y_threshold: 疑似阈值 Y。
            required_margin: 第一名需领先第二名的最小差距。
            stage: 阶段名 (用于日志)。

        Returns:
            更新后的 MatchResult。
        """
        candidates = match_result.candidates

        # 无候选
        if not candidates:
            match_result.status = IdentityStatus.STRANGER
            match_result.best_match = None
            logger.debug("[{}] No candidates → STRANGER", stage)
            return match_result

        top = candidates[0]
        top_score = top.fused_score

        # 所有候选 < Y
        if top_score < y_threshold:
            match_result.status = IdentityStatus.STRANGER
            match_result.best_match = None
            logger.debug(
                "[{}] Top score {:.3f} < Y={:.3f} → STRANGER",
                stage, top_score, y_threshold,
            )
            return match_result

        # 计算超过 X 阈值的候选数量
        above_x = [c for c in candidates if c.fused_score >= x_threshold]

        if len(above_x) >= 2:
            # 多人 ≥ X → CONFLICT
            match_result.status = IdentityStatus.CONFLICT
            match_result.best_match = top
            logger.info(
                "[{}] {} candidates ≥ X={:.3f} → CONFLICT",
                stage, len(above_x), x_threshold,
            )
        elif len(above_x) == 1:
            # 唯一候选 ≥ X, 检查差距
            current_margin = match_result.margin
            if current_margin >= required_margin:
                match_result.status = IdentityStatus.CONFIDENT
                match_result.best_match = top
                logger.info(
                    "[{}] {} CONFIDENT (score={:.3f}, margin={:.3f})",
                    stage, top.person_id, top_score, current_margin,
                )
            else:
                match_result.status = IdentityStatus.CONFLICT
                match_result.best_match = top
                logger.info(
                    "[{}] {} margin {:.3f} < {:.3f} → CONFLICT",
                    stage, top.person_id, current_margin, required_margin,
                )
        else:
            # Y ≤ top < X
            match_result.status = IdentityStatus.SUSPECTED
            match_result.best_match = top
            logger.debug(
                "[{}] {} SUSPECTED (score={:.3f})",
                stage, top.person_id, top_score,
            )

        return match_result
