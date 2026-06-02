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
8. K-Reciprocal 重排序
9. 多模态融合
10. 歧义消解 (ReID 阶段)
11. 若 SUSPECTED / CONFLICT → VLM 仲裁
12. 返回最终 MatchResult

每个阶段记录耗时到 PipelineDebug。
"""
from __future__ import annotations

import time
from typing import Any, Optional, Protocol

import cv2
import numpy as np
from loguru import logger

from src.config import Config
from src.gallery.data_models import (
    BodyProportions,
    Detection,
    IdentityStatus,
    MatchCandidate,
    MatchResult,
    PersonProfile,
    PipelineDebug,
    PoseBucket,
    TrackedPerson,
)
from src.pipeline.temporal_aggregator import TemporalAggregator


# ---------------------------------------------------------------------------
# 依赖接口协议
# ---------------------------------------------------------------------------

class HeavyDetectorProtocol(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class FaceExtractorProtocol(Protocol):
    def extract(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[tuple[np.ndarray, float]]:
        """返回 (embedding, quality) 或 None。"""
        ...


class BodyExtractorProtocol(Protocol):
    def extract(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """返回 L2 归一化的 body embedding。"""
        ...


class GalleryMatcherProtocol(Protocol):
    def match(
        self,
        face_embedding: Optional[np.ndarray],
        body_embedding: Optional[np.ndarray],
        proportions: Optional[BodyProportions],
        gallery: dict[str, PersonProfile],
        pose_bucket: PoseBucket,
        face_quality: float,
    ) -> list[MatchCandidate]:
        """返回按 fused_score 降序的候选列表。"""
        ...


class GalleryUpdaterProtocol(Protocol):
    def update(
        self,
        profile: PersonProfile,
        face_embedding: Optional[np.ndarray],
        body_embedding: Optional[np.ndarray],
        proportions: Optional[BodyProportions],
        pose_bucket: PoseBucket,
        face_quality: float,
        body_quality: float,
    ) -> None: ...


class RerankerProtocol(Protocol):
    def rerank(
        self, candidates: list[MatchCandidate], body_embedding: Optional[np.ndarray],
        gallery: dict[str, PersonProfile],
    ) -> list[MatchCandidate]: ...


class FusionProtocol(Protocol):
    def fuse(
        self, candidates: list[MatchCandidate], face_quality: float
    ) -> list[MatchCandidate]: ...


class ResolverProtocol(Protocol):
    def resolve(
        self, candidates: list[MatchCandidate], face_quality: float
    ) -> MatchResult: ...


class VLMProtocol(Protocol):
    async def arbitrate(
        self,
        frame: np.ndarray,
        person_crop: np.ndarray,
        candidates: list[MatchCandidate],
        gallery: dict[str, PersonProfile],
    ) -> MatchResult: ...


# ---------------------------------------------------------------------------
# Tier 2 Processor
# ---------------------------------------------------------------------------

class Tier2Processor:
    """
    Tier 2 深度身份识别处理器。

    异步处理单个追踪目标的精确身份识别：从 ROI 切割开始
    到最终 MatchResult 输出，沿途在 PipelineDebug 中记录
    每个阶段的状态和耗时。

    Args:
        config: 全局配置。
        heavy_detector: 精确 YOLO 检测器 (Tier 2)。
        face_extractor: 人脸特征提取器。
        body_extractor: 全身 ReID 特征提取器。
        gallery_matcher: 底库匹配器。
        gallery_updater: 底库更新器。
        reranker: K-Reciprocal 重排序器。
        fusion: 多模态融合器。
        resolver: 歧义消解器。
        vlm: VLM 仲裁器。
        temporal_aggregator: 时序特征聚合器。
    """

    def __init__(
        self,
        config: Config,
        heavy_detector: Any,
        face_extractor: Any,
        body_extractor: Any,
        gallery_matcher: Any,
        gallery_updater: Any,
        reranker: Any,
        fusion: Any,
        resolver: Any,
        vlm: Any,
        temporal_aggregator: TemporalAggregator,
    ) -> None:
        self._config = config
        self._heavy_detector = heavy_detector
        self._face_extractor = face_extractor
        self._body_extractor = body_extractor
        self._gallery_matcher = gallery_matcher
        self._gallery_updater = gallery_updater
        self._reranker = reranker
        self._fusion = fusion
        self._resolver = resolver
        self._vlm = vlm
        self._temporal_aggregator = temporal_aggregator

        logger.info("Tier2Processor initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        gallery: dict[str, PersonProfile],
    ) -> tuple[MatchResult, PipelineDebug]:
        """
        对单个追踪目标执行深度身份识别。

        Args:
            frame: 完整 BGR 帧 (H, W, 3)。
            person: Tier 1 输出的追踪人物。
            gallery: 当前底库 (person_id → PersonProfile)。

        Returns:
            (MatchResult, PipelineDebug) 元组。
        """
        debug = PipelineDebug()
        bbox = person.detection.bbox
        track_id = person.track_id

        # --- 0. 裁剪 ROI ---
        person_crop = self._crop_roi(frame, bbox)

        # --- 1. 精确检测 ---
        detection = await self._stage_detection(person_crop, debug)
        if detection is None:
            # 使用 Tier 1 的检测结果作为后备
            detection = person.detection

        # --- 2. 姿态分类 ---
        pose_bucket = self._stage_pose(detection, debug)

        # --- 3. 人脸特征提取 ---
        face_embedding, face_quality = await self._stage_face(
            person_crop, detection, pose_bucket, debug
        )

        # --- 4. 全身 ReID 特征提取 ---
        body_embedding = await self._stage_body(person_crop, detection, debug)

        # --- 5. 体型比例提取 ---
        proportions = self._stage_proportions(detection, debug)

        # --- 6. 时序聚合 ---
        aggregated_body = self._stage_temporal(
            track_id, body_embedding, face_quality or 0.5, debug
        )

        # --- 7. 底库匹配 (分模态) ---
        face_candidates = None
        body_candidates = None
        proportion_candidates = None

        if face_embedding is not None and gallery:
            face_candidates = self._gallery_matcher.match_face(
                face_embedding, pose_bucket, gallery
            )
        if aggregated_body is not None and gallery:
            body_candidates = self._gallery_matcher.match_body(
                aggregated_body, gallery
            )
        if proportions is not None and gallery:
            proportion_candidates = self._gallery_matcher.match_proportions(
                proportions, gallery
            )

        debug.matching.status = "done"

        # --- 8. 多模态融合 ---
        candidates = self._stage_fusion(
            face_candidates, body_candidates, proportion_candidates,
            face_quality or 0.0, debug
        )

        # --- 9. K-Reciprocal 重排序 ---
        candidates = self._stage_rerank(candidates, aggregated_body, gallery, debug)

        # --- 10. 歧义消解 (ReID 阶段) ---
        result = self._stage_resolve(candidates, face_quality or 0.0, debug)

        # --- 11. VLM 仲裁 (按需) ---
        if result.status in (IdentityStatus.SUSPECTED, IdentityStatus.CONFLICT):
            result = await self._stage_vlm(
                frame, person_crop, result.candidates, gallery, debug
            )

        # 附加 debug 信息
        result.face_quality = face_quality or 0.0

        logger.debug(
            "Tier2 track_id={}: status={}, best={}({:.3f})",
            track_id,
            result.status.value,
            result.best_match.person_id if result.best_match else "none",
            result.best_match.fused_score if result.best_match else 0.0,
        )

        return result, debug

    def get_debug(self) -> PipelineDebug:
        """获取最近一次处理的调试信息。"""
        return self._last_debug if hasattr(self, "_last_debug") else PipelineDebug()

    # ------------------------------------------------------------------
    # Stage 实现
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_roi(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """裁剪感兴趣区域, 带边界保护。"""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))
        return frame[y1:y2, x1:x2].copy()

    async def _stage_detection(
        self, crop: np.ndarray, debug: PipelineDebug
    ) -> Optional[Detection]:
        """Stage 1: 精确检测。"""
        debug.detection.status = "running"
        t0 = time.perf_counter()
        try:
            detections = self._heavy_detector.detect(crop)
            elapsed = (time.perf_counter() - t0) * 1000
            debug.detection.time_ms = elapsed
            if detections:
                debug.detection.status = "done"
                debug.detection.details = {"count": len(detections)}
                return detections[0]  # 取最大的检测
            debug.detection.status = "done"
            debug.detection.details = {"count": 0, "fallback": True}
            return None
        except Exception:
            debug.detection.status = "done"
            debug.detection.time_ms = (time.perf_counter() - t0) * 1000
            debug.detection.details = {"error": "detection_failed"}
            logger.exception("Tier2 heavy detection failed")
            return None

    def _stage_pose(
        self, detection: Detection, debug: PipelineDebug
    ) -> PoseBucket:
        """Stage 2: 姿态分类 (从关键点推断)。"""
        debug.pose.status = "running"
        t0 = time.perf_counter()
        try:
            pose_bucket = detection.pose_bucket
            if pose_bucket == PoseBucket.UNKNOWN:
                pose_bucket = self._classify_pose(detection.keypoints)
            debug.pose.status = "done"
            debug.pose.time_ms = (time.perf_counter() - t0) * 1000
            debug.pose.details = {"bucket": pose_bucket.value}
            return pose_bucket
        except Exception:
            debug.pose.status = "done"
            debug.pose.time_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Pose classification failed")
            return PoseBucket.UNKNOWN

    async def _stage_face(
        self,
        crop: np.ndarray,
        detection: Detection,
        pose_bucket: PoseBucket,
        debug: PipelineDebug,
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        """Stage 3: 人脸特征提取（背面跳过）。"""
        if pose_bucket == PoseBucket.BACK:
            debug.face.status = "skipped"
            debug.face.details = {"reason": "back_pose"}
            return None, None

        debug.face.status = "running"
        t0 = time.perf_counter()
        try:
            result = self._face_extractor.extract(crop, detection.bbox)
            elapsed = (time.perf_counter() - t0) * 1000
            debug.face.time_ms = elapsed

            if result is not None:
                debug.face.status = "done"
                debug.face.details = {"quality": round(result.quality, 3)}
                return result.embedding, result.quality
            else:
                debug.face.status = "done"
                debug.face.details = {"found": False}
                return None, None
        except Exception:
            debug.face.status = "done"
            debug.face.time_ms = (time.perf_counter() - t0) * 1000
            debug.face.details = {"error": "extraction_failed"}
            logger.exception("Face extraction failed")
            return None, None

    async def _stage_body(
        self,
        crop: np.ndarray,
        detection: Detection,
        debug: PipelineDebug,
    ) -> Optional[np.ndarray]:
        """Stage 4: 全身 ReID 特征提取。"""
        debug.reid.status = "running"
        t0 = time.perf_counter()
        try:
            embedding = self._body_extractor.extract(crop, detection.bbox)
            elapsed = (time.perf_counter() - t0) * 1000
            debug.reid.time_ms = elapsed
            debug.reid.status = "done"
            debug.reid.details = {
                "dim": embedding.shape[0] if embedding is not None else 0,
            }
            return embedding
        except Exception:
            debug.reid.status = "done"
            debug.reid.time_ms = (time.perf_counter() - t0) * 1000
            debug.reid.details = {"error": "extraction_failed"}
            logger.exception("Body ReID extraction failed")
            return None

    def _stage_proportions(
        self, detection: Detection, debug: PipelineDebug
    ) -> Optional[BodyProportions]:
        """Stage 5: 体型比例提取。"""
        try:
            proportions = BodyProportions.from_keypoints(detection.keypoints)
            return proportions
        except Exception:
            logger.debug("Proportions extraction failed (keypoints insufficient)")
            return None

    def _stage_temporal(
        self,
        track_id: int,
        body_embedding: Optional[np.ndarray],
        quality: float,
        debug: PipelineDebug,
    ) -> Optional[np.ndarray]:
        """Stage 6: 时序聚合。"""
        if body_embedding is None:
            return None
        try:
            return self._temporal_aggregator.add_and_get(
                track_id, body_embedding, quality
            )
        except Exception:
            logger.exception("Temporal aggregation failed")
            return body_embedding

    async def _stage_matching(
        self,
        face_embedding: Optional[np.ndarray],
        body_embedding: Optional[np.ndarray],
        proportions: Optional[BodyProportions],
        gallery: dict[str, PersonProfile],
        pose_bucket: PoseBucket,
        face_quality: float,
        debug: PipelineDebug,
    ) -> list[MatchCandidate]:
        """Stage 7: 底库匹配 (分别匹配人脸/全身/体型, 合并候选)。"""
        debug.matching.status = "running"
        t0 = time.perf_counter()
        try:
            # 收集各模态候选 {person_id → MatchCandidate}
            merged: dict[str, MatchCandidate] = {}

            # 人脸匹配
            if face_embedding is not None:
                for c in self._gallery_matcher.match_face(face_embedding, pose_bucket, gallery):
                    merged[c.person_id] = MatchCandidate(
                        person_id=c.person_id,
                        display_name=c.display_name,
                        face_score=c.face_score,
                    )

            # 全身匹配
            if body_embedding is not None:
                for c in self._gallery_matcher.match_body(body_embedding, gallery):
                    if c.person_id in merged:
                        merged[c.person_id].body_score = c.body_score
                    else:
                        merged[c.person_id] = MatchCandidate(
                            person_id=c.person_id,
                            display_name=c.display_name,
                            body_score=c.body_score,
                        )

            # 体型比例匹配
            if proportions is not None:
                for c in self._gallery_matcher.match_proportions(proportions, gallery):
                    if c.person_id in merged:
                        merged[c.person_id].proportion_score = c.proportion_score
                    else:
                        merged[c.person_id] = MatchCandidate(
                            person_id=c.person_id,
                            display_name=c.display_name,
                            proportion_score=c.proportion_score,
                        )

            candidates = list(merged.values())
            # 按最高可用分降序排列
            candidates.sort(
                key=lambda c: max(c.face_score or 0.0, c.body_score or 0.0, c.proportion_score or 0.0),
                reverse=True,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            debug.matching.time_ms = elapsed
            debug.matching.status = "done"
            debug.matching.details = {
                "candidates": len(candidates),
                "top_score": (
                    round(max(
                        candidates[0].face_score or 0.0,
                        candidates[0].body_score or 0.0,
                    ), 3) if candidates else 0.0
                ),
            }
            return candidates
        except Exception:
            debug.matching.status = "done"
            debug.matching.time_ms = (time.perf_counter() - t0) * 1000
            debug.matching.details = {"error": "matching_failed"}
            logger.exception("Gallery matching failed")
            return []

    def _stage_rerank(
        self,
        candidates: list[MatchCandidate],
        body_embedding: Optional[np.ndarray],
        gallery: dict[str, PersonProfile],
        debug: PipelineDebug,
    ) -> list[MatchCandidate]:
        """Stage 8: K-Reciprocal 重排序。"""
        if not candidates or body_embedding is None:
            return candidates
        try:
            # rerank(query_embedding, gallery_embeddings, initial_scores)
            gallery_embeddings: list[tuple[str, np.ndarray]] = []
            for pid, profile in gallery.items():
                for outfit in profile.wardrobe:
                    gallery_embeddings.append((pid, outfit.body_embedding))
            if not gallery_embeddings:
                return candidates
            return self._reranker.rerank(body_embedding, gallery_embeddings, candidates)
        except Exception:
            logger.exception("Reranking failed, using original order")
            return candidates

    def _stage_fusion(
        self,
        face_candidates: Optional[list[MatchCandidate]],
        body_candidates: Optional[list[MatchCandidate]],
        proportion_candidates: Optional[list[MatchCandidate]],
        face_quality: float,
        debug: PipelineDebug,
    ) -> list[MatchCandidate]:
        """Stage 9: 多模态融合。"""
        if not face_candidates and not body_candidates and not proportion_candidates:
            return []
        try:
            return self._fusion.fuse(face_candidates, body_candidates, proportion_candidates, face_quality)
        except Exception:
            logger.exception("Fusion failed, using raw scores")
            # 回退: 合并所有候选
            all_c = (face_candidates or []) + (body_candidates or []) + (proportion_candidates or [])
            return all_c

    def _stage_resolve(
        self,
        candidates: list[MatchCandidate],
        face_quality: float,
        debug: PipelineDebug,
    ) -> MatchResult:
        """Stage 10: 歧义消解 (ReID 阶段)。"""
        debug.identity.status = "running"
        t0 = time.perf_counter()
        try:
            # 构建初始 MatchResult
            match_result = MatchResult(
                candidates=candidates,
                best_match=candidates[0] if candidates else None,
                status=IdentityStatus.IDENTIFYING,
                face_quality=face_quality,
            )
            result = self._resolver.resolve_reid(match_result)
            elapsed = (time.perf_counter() - t0) * 1000
            debug.identity.time_ms = elapsed
            debug.identity.status = "done"
            debug.identity.details = {
                "status": result.status.value,
                "margin": round(result.margin, 3),
            }
            return result
        except Exception:
            debug.identity.status = "done"
            debug.identity.time_ms = (time.perf_counter() - t0) * 1000
            debug.identity.details = {"error": "resolve_failed"}
            logger.exception("Identity resolution failed")
            return MatchResult(
                candidates=candidates,
                status=IdentityStatus.STRANGER,
            )

    async def _stage_vlm(
        self,
        frame: np.ndarray,
        crop: np.ndarray,
        candidates: list[MatchCandidate],
        gallery: dict[str, PersonProfile],
        debug: PipelineDebug,
    ) -> MatchResult:
        """Stage 11: VLM 仲裁。"""
        try:
            logger.info("VLM arbitration triggered ({} candidates)", len(candidates))

            # VLM 需要 JPEG 字节
            _, query_jpeg = cv2.imencode('.jpg', crop)
            query_bytes = query_jpeg.tobytes()

            # 构造候选图片列表 (从底库取缩略图)
            candidate_images: list[tuple[str, bytes]] = []
            for c in candidates[:3]:  # 最多 3 个候选
                profile = gallery.get(c.person_id)
                if profile:
                    # 尝试从人脸特征中取缩略图
                    for bucket_entries in profile.face_features.values():
                        for entry in bucket_entries:
                            if entry.source_image:
                                candidate_images.append((c.person_id, entry.source_image))
                                break
                        if any(pid == c.person_id for pid, _ in candidate_images):
                            break

            vlm_response = await self._vlm.arbitrate(query_bytes, candidate_images)

            # 用 resolver 的 VLM 阶段处理结果
            match_result = MatchResult(
                candidates=candidates,
                best_match=candidates[0] if candidates else None,
                status=IdentityStatus.SUSPECTED,
            )
            result = self._resolver.resolve_vlm(vlm_response, match_result)
            return result
        except Exception:
            logger.exception("VLM arbitration failed, falling back to ReID result")
            return MatchResult(
                candidates=candidates,
                best_match=candidates[0] if candidates else None,
                status=IdentityStatus.SUSPECTED if candidates else IdentityStatus.STRANGER,
            )

    # ------------------------------------------------------------------
    # Pose classification helper
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_pose(keypoints: np.ndarray) -> PoseBucket:
        """
        从 COCO 17 关键点推断姿态桶。

        使用鼻子、眼睛、耳朵的可见性判断:
        - 鼻子 + 双眼可见 → FRONTAL
        - 仅左眼/左耳可见 → LEFT
        - 仅右眼/右耳可见 → RIGHT
        - 关键点大量缺失 → BACK
        """
        conf_thresh = 0.3
        if keypoints.shape[0] < 5:
            return PoseBucket.UNKNOWN

        nose_vis = keypoints[0, 2] > conf_thresh
        leye_vis = keypoints[1, 2] > conf_thresh
        reye_vis = keypoints[2, 2] > conf_thresh
        lear_vis = keypoints[3, 2] > conf_thresh
        rear_vis = keypoints[4, 2] > conf_thresh

        # 双眼 + 鼻子可见 → 正面
        if nose_vis and leye_vis and reye_vis:
            return PoseBucket.FRONTAL

        # 单侧可见
        left_score = int(leye_vis) + int(lear_vis)
        right_score = int(reye_vis) + int(rear_vis)

        if left_score > right_score and left_score >= 1:
            return PoseBucket.LEFT
        if right_score > left_score and right_score >= 1:
            return PoseBucket.RIGHT

        # 面部关键点均不可见 → 背面
        if not nose_vis and not leye_vis and not reye_vis:
            return PoseBucket.BACK

        return PoseBucket.UNKNOWN
