"""
Per-track 持久状态

每个被追踪的人拥有一个 TrackState, 管理:
- 帧缓冲 (RecentBuffer) + 质量缓存 (QualityCache)
- 跨帧调度时间戳
- 身份结果 (IdentityResult)
- VLM 异步任务生命周期
- 识别调度 (resolve)
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.gallery.data_models import PersonProfile, PoseBucket
from src.pipeline.data_models import (
    IdentityResult,
    IdentityStatus,
    MatchResult,
    TrackedPerson,
)
from src.pipeline.frame_buffer import BufferEntry, QualityCache, RecentBuffer
from src.pipeline.quality_utils import compute_blur_score, compute_quality_hint
from src.tier1.face_detector_light import get_face_detector_light
from src.tier2.features.edifiqa import get_edifiqa
from src.pipeline.scheduler import Tier2Action, should_trigger_tier2, should_trigger_vlm
from src.tier2.processor import Tier2Processor
from src.tier3.processor import Tier3VLMProcessor


class TrackState(BaseModel):
    """Per-track 持久状态: 缓冲帧 + 质量缓存 + 调度 + VLM 任务。

    TrackedPerson 每帧重建, 但调度状态需要跨帧保持, 所以放在这里。
    VLM 异步任务也归属于 track, 清理 track 时自动取消。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    buffer: RecentBuffer = Field(default_factory=RecentBuffer)
    quality_cache: QualityCache = Field(default_factory=QualityCache)
    person: TrackedPerson

    # --- 跨帧持久状态 ---
    identity_result: IdentityResult = Field(default_factory=IdentityResult)
    is_current_target: bool = False  # 是否为当前注意力目标
    # 上次 Tier 2 处理时间，如果要提高响应实时性，需减少 last_tier2_time 的初始时间
    last_tier2_time: float = Field(default_factory=time.monotonic)
    last_vlm_time: float = 0.0  # 上次 VLM 处理时间
    tier2_count: int = 0  # Tier2 累计执行次数
    force_probe: bool = False  # 强制触发 Tier2 (如身份冲突后)

    # --- VLM 异步任务 (归属于 track 生命周期) ---
    vlm_task: asyncio.Task | None = Field(default=None, exclude=True)  # type: ignore[type-arg]
    vlm_result: MatchResult | None = None  # VLM 完成后暂存, 下帧应用

    # ==================================================================
    # VLM 生命周期
    # ==================================================================

    def cancel_vlm(self) -> None:
        """取消 pending VLM 任务并清理结果。"""
        if self.vlm_task is not None and not self.vlm_task.done():
            self.vlm_task.cancel()
        self.vlm_task = None
        self.vlm_result = None

    async def _run_vlm(
            self, match_result: MatchResult, gallery: dict[str, PersonProfile],
    ) -> None:
        """异步执行 Tier 3 VLM 仲裁，结果存入 self.vlm_result。"""
        track_id = self.person.track_id
        self.last_vlm_time = time.monotonic()
        try:
            result = await Tier3VLMProcessor.process(
                track_id=track_id,
                match_result=match_result,
                quality_cache=self.quality_cache,
                gallery=gallery,
            )
            if result is not None:
                self.vlm_result = result
        except asyncio.CancelledError:
            logger.debug("VLM task cancelled for track_id={}", track_id)
        except Exception as e:
            logger.exception("VLM processing failed for track_id={}: {}", track_id, e)
        finally:
            self.vlm_task = None

    # ==================================================================
    # 识别调度
    # ==================================================================

    def process_frame(
            self, frame: np.ndarray, gallery: dict[str, PersonProfile],
    ) -> tuple[MatchResult | None, bool, float, float]:
        """单帧步进: 注入图像 + 执行调度。

        Args:
            frame: 当前帧。
            gallery: 底库。

        Returns:
            (MatchResult | None, is_enrich, face_detect_ms, face_assess_ms)
        """
        face_detect_ms, face_assess_ms = self.feed_frame(frame)
        match_result, is_enrich = self.resolve(gallery)
        return match_result, is_enrich, face_detect_ms, face_assess_ms

    # ==================================================================
    # 帧缓存管理
    # ==================================================================

    def feed_frame(self, frame: np.ndarray) -> tuple[float, float]:
        """将 Tier1 检测结果裁剪并推入 per-track RecentBuffer。

        Returns:
            (face_detect_ms, face_assess_ms) 人脸检测和质量评估各自耗时。
        """
        det = self.person.detection
        if det is None or det.bbox is None:
            return 0.0, 0.0

        h_f, w_f = frame.shape[:2]
        x1 = max(0, int(det.bbox[0]))
        y1 = max(0, int(det.bbox[1]))
        x2 = min(w_f, int(det.bbox[2]))
        y2 = min(h_f, int(det.bbox[3]))
        crop = frame[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return 0.0, 0.0

        q_hint = compute_quality_hint(det.bbox, det.keypoints, (h_f, w_f))

        local_kps = det.keypoints.copy() if det.keypoints is not None else None
        if local_kps is not None:
            local_kps[:, 0] -= x1
            local_kps[:, 1] -= y1

        # 轻量人脸检测 + eDifFIQA + blur 补充 (非 BACK 姿态)
        face_quality = 0.0
        aligned_face = None
        face_bbox = None
        face_kps = None
        face_detect_ms = 0.0
        face_assess_ms = 0.0
        if det.pose_bucket != PoseBucket.BACK:
            t_det = time.perf_counter()
            result = get_face_detector_light().get_aligned_face(crop)
            face_detect_ms = (time.perf_counter() - t_det) * 1000
            if result is not None:
                aligned_face, face_bbox, face_kps = result
                t_assess = time.perf_counter()
                edifiqa_score = get_edifiqa().predict(aligned_face)
                blur = compute_blur_score(aligned_face)
                face_assess_ms = (time.perf_counter() - t_assess) * 1000
                face_quality = 0.8 * edifiqa_score + 0.2 * blur

        entry = BufferEntry(
            timestamp=time.time(),
            crop=crop,
            bbox=det.bbox,
            keypoints=local_kps,
            pose_bucket=det.pose_bucket,
            quality_hint=q_hint,
            face_quality=face_quality,
            aligned_face=aligned_face,
            face_bbox=face_bbox,
            face_kps=face_kps,
        )
        self.buffer.push(entry)
        return face_detect_ms, face_assess_ms

    def resolve(self, gallery: dict[str, PersonProfile],
                ) -> tuple[MatchResult | None, bool]:
        """识别阶段: 消费 VLM 结果 / 执行 Tier2 / 触发 VLM, 更新 identity_result。

        内部自动管理 VLM 生命周期, 调用方无需感知 VLM 存在。

        Args:
            gallery: 当前底库 (Tier2 匹配需要)。

        Returns:
            (MatchResult | None, is_enrich)
            MatchResult 为 None 表示本帧无新结果。
            is_enrich 为 True 表示本次是后台富化 (不更新身份/不发事件)。
        """
        track_id = self.person.track_id

        # --- 1. 消费上一轮 VLM 异步结果 (优先级高于 Tier2) ---
        vlm_result = self.vlm_result
        if vlm_result is not None:
            self.vlm_result = None  # 消费掉
            if self.identity_result.status in (IdentityStatus.SUSPECTED, IdentityStatus.CONFLICT):
                self.update_identity(vlm_result)
                # VLM 结果已应用, 本帧跳过 Tier2, 避免覆盖仲裁结论
                return vlm_result, False
            else:
                logger.debug(
                    "VLM result for track_id={} discarded: status already {}",
                    track_id, self.identity_result.status.value,
                )

        # --- 2. Tier2 调度 ---
        if len(self.buffer) == 0:
            return None, False

        action = should_trigger_tier2(self)
        if action == Tier2Action.SKIP:
            return None, False

        # 更新调度状态
        self.force_probe = False
        self.last_tier2_time = time.monotonic()
        self.tier2_count += 1

        # --- 3. Tier2 执行 ---
        result, _ = Tier2Processor.process_multiframe(
            track_id=track_id,
            buffer=self.buffer,
            quality_cache=self.quality_cache,
            gallery=gallery,
        )

        if result is None:
            return None, False

        if action == Tier2Action.TRIGGER_ENRICH:
            # ENRICH 不更新身份, 也不触发 VLM
            if result.best_match.person_id != self.identity_result.person_id:
                logger.error("TRIGGER_ENRICH 触发人物变更")
                return None, False
            return result, True

        # --- 4. 更新身份 + 非 DEFINITE 时尝试 VLM ---
        self.update_identity(result)
        if should_trigger_vlm(self):
            self.cancel_vlm()
            self.vlm_task = asyncio.create_task(self._run_vlm(result, gallery))

        return result, False

    def update_identity(self, result: MatchResult) -> None:
        """将 MatchResult 写入 identity_result (纯状态更新, 无副作用)。"""
        best = result.best_match
        if best:
            self.identity_result = IdentityResult(
                **best.model_dump(),
                status=result.status,
            )
        else:
            self.identity_result = IdentityResult(status=result.status)
