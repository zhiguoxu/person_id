"""
视觉流水线主编排器

负责初始化所有子模块、协调 Tier1 / Tier2 / Tier3 处理流程、
管理底库与 per-track 状态、发出系统事件，以及选择注意力目标。

通过 ``await VisionOrchestrator.create(config)`` 创建实例，
所有子模块在构造时一次性加载完毕。
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import cv2
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_config
from src.gallery.data_models import (
    FeatureEntry,
    GalleryUpdateResult,
    PersonProfile, PoseBucket,
)
from src.pipeline.data_models import (
    ConfirmResult,
    EventType,
    IdentityResult,
    IdentityStatus,
    MatchResult,
    RegisterFailureReason,
    SystemEvent,
    TrackedPerson,
)
from src.gallery.persistence import get_gallery_persistence
from src.pipeline.frame_buffer import CachedFrame
from src.tier2.features.ediffiqa import get_ediffiqa_enroll
from src.tier2.multi_frame_aggregator import MultiFrameAggregator

from src.tier1.processor import Tier1Processor

from src.pipeline.track_state import TrackState


# ------------------------------------------------------------------
# 主编排器
# ------------------------------------------------------------------

class VisionOrchestrator(BaseModel):
    """视觉流水线主编排器。

    统一管理检测、追踪、特征提取、匹配、Gallery 及事件系统。
    每个 Tier 处理器自行创建所需子模块，编排器仅负责协调。

    使用 ``await VisionOrchestrator.create()`` 创建实例。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- 摄像头标识 ---
    camera_id: str

    # --- Pipeline 处理器 ---
    tier1: Tier1Processor = Field(default_factory=Tier1Processor)

    # --- Gallery ---
    gallery: dict[str, PersonProfile] = Field(default_factory=dict)

    # --- Per-track 状态 ---
    tracks: dict[int, TrackState] = Field(default_factory=dict)

    # --- 异步任务 ---
    save_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)

    # --- 事件系统 ---
    event_log: list[SystemEvent] = Field(default_factory=list)
    new_events: list[SystemEvent] = Field(default_factory=list)
    max_events: int = 200
    last_event_time: dict[str, float] = Field(default_factory=dict)
    event_cooldown_sec: float = 10.0

    # --- 帧计数与注意力 ---
    frame_count: int = 0
    current_target_id: int | None = None

    # ==================================================================
    # Factory
    # ==================================================================

    @classmethod
    async def create(cls, camera_id: str) -> VisionOrchestrator:
        """工厂方法：创建指定摄像头的 VisionOrchestrator。

        Args:
            camera_id: 摄像头标识，用于隔离 Gallery 和追踪状态。
        """
        logger.info("正在初始化 VisionOrchestrator (camera={}) ...", camera_id)
        orch = cls(camera_id=camera_id)
        await orch._load_gallery_from_db()

        logger.info(
            "VisionOrchestrator [{}] 已初始化 (gallery: {} 人)",
            camera_id, len(orch.gallery),
        )
        return orch

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def shutdown(self) -> None:
        """清理资源: 取消异步任务、保存底库。"""
        logger.info("正在关闭 VisionOrchestrator ...")

        for state in self.tracks.values():
            state.cancel_vlm()

        self.tracks.clear()
        logger.info("VisionOrchestrator 关闭完成")

    def reset_attention(self) -> None:
        """断流时清空运行时轨迹与注意力目标 (gallery 人物档案不动)。

        轨迹清理与目标重选都只在 process_frame 里发生, 断流后不再有帧进来,
        不清的话 current_identity 会一直返回最后一帧镜头前的人。轨迹 id 是
        流内概念, 重连后 tracker 产生的是新轨迹, 旧轨迹不可能续上, 清掉无损。
        """
        if not self.tracks and self.current_target_id is None:
            return
        for state in self.tracks.values():
            state.cancel_vlm()
        self.tracks.clear()
        self.current_target_id = None
        logger.info(
            "已清空运行时轨迹与注意力目标: camera={} (断流)", self.camera_id,
        )

    # ==================================================================
    # Frame Processing
    # ==================================================================

    async def process_frame(self, frame: np.ndarray) -> dict[str, object]:
        """处理一帧图像并返回结果。

        Args:
            frame: BGR 图像 (H, W, 3)。

        Returns:
            包含 tracked_persons, current_target, events, pipeline_debug 的字典。
        """
        self.frame_count += 1
        t0 = time.perf_counter()

        # --- Tier 1: 检测 + 追踪 + 注意力 + 人脸 ---
        persons = self.tier1.process(frame)
        tier1_ms = (time.perf_counter() - t0) * 1000

        # --- 人脸耗时直接从 persons 汇总 ---
        total_face_detect_ms = sum(p.face_detect_ms for p in persons)
        total_face_assess_ms = sum(p.face_assess_ms for p in persons)

        # --- 刷新 TrackState ---
        self._refresh_track_states(persons)

        # --- track 逐个进处理 ---
        gallery_dirty = False
        for state in self.tracks.values():
            result, is_enrich = state.process_frame(frame, self.gallery)
            if result is None:
                continue

            # 无新 embedding → 发 DATA_STALE 事件 (前端可清缓存, 节流)
            if result.stale:
                now = time.time()
                stale_key = f"stale_{state.person.track_id}"
                if now - self.last_event_time.get(stale_key, 0) > self.event_cooldown_sec:
                    self.last_event_time[stale_key] = now
                    self._emit_event(
                        EventType.DATA_STALE,
                        track_id=state.person.track_id,
                        source="reid",
                        message="No new embeddings, quality cache unchanged",
                    )
                continue

            self._emit_match_event(state.person.track_id, result)

            if result.status == IdentityStatus.DEFINITE:
                gallery_dirty = True
                changes = self._update_gallery_person(result.best_match.person_id, state)
                asyncio.ensure_future(
                    self._save_gallery_person_incremental(result.best_match.person_id, changes)
                )

        # 统一：设 force_probe
        if gallery_dirty:
            self._on_gallery_updated()

        # --- 注意力选择 ---
        self._select_target()

        # --- 构建响应 ---
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response = self._build_response(elapsed_ms)
        response["pipeline_debug"] = self._build_debug(tier1_ms, elapsed_ms, total_face_detect_ms, total_face_assess_ms)
        return response

    # ==================================================================
    # Gallery 删除
    # ==================================================================

    def delete_person(self, person_id: str) -> None:
        """同步删除: 清理内存 gallery 和 track 状态。

        gallery.pop 必须第一行执行 (同步, 不可被打断),
        后续所有 save 路径通过 `person_id not in self.gallery` 判断是否已删除。
        DB 删除由调用方 (routes.py) 单独处理。
        """
        removed = self.gallery.pop(person_id, None)

        reset_count = 0
        for state in self.tracks.values():
            if state.identity_result.person_id == person_id:
                state.identity_result = IdentityResult(
                    status=IdentityStatus.IDENTIFYING,
                )
                state.quality_cache.clear()
                state.force_probe = True
                reset_count += 1

        logger.info(
            "delete_person: person_id={}, gallery_removed={}, tracks_reset={}",
            person_id, removed is not None, reset_count,
        )

    async def delete_person_from_db(self, person_id: str) -> None:
        """在 save_lock 保护下删除 DB 数据, 防止与入库 commit 交叉。"""
        async with self.save_lock:
            persistence = get_gallery_persistence()
            await persistence.delete_profile(person_id, self.camera_id)

    # ==================================================================
    # Human-in-the-loop
    # ==================================================================

    async def confirm_identity(
            self, track_id: int, person_id: str | None, name: str,
            min_face_quality: float | None = None,
    ) -> ConfirmResult:
        """人工确认身份。

        Args:
            track_id: 需要确认的轨迹 ID。
            person_id: 底库人物 ID（如不存在或为 None 则自动创建）。
            name: 显示名称。
            min_face_quality: 人脸入库质量门槛下限(可选)。与配置的 enroll 阈值
                取较大值生效(只提高不降低), 供主动注册流程用更高标准采集底片。

        Returns:
            ConfirmResult: 成功时携带最终入库的 person_id; 预期内失败
            (没看到人/没看清脸等) 时携带原因码与可读信息, 不抛异常。
        """
        logger.info(
            "人工已确认: track_id={} → person_id={}, name={}",
            track_id, person_id, name,
        )

        state = self.tracks.get(track_id)
        if state is None:
            return ConfirmResult.fail(
                RegisterFailureReason.NO_TARGET,
                f"Track {track_id} not found, cannot confirm",
            )

        # 检查是否有人脸数据 (必须有至少一个带 embedding 的人脸帧)
        has_face = any(
            cf.embedding is not None for cf in state.quality_cache.face_pool
        )
        if not has_face:
            return ConfirmResult.fail(
                RegisterFailureReason.NO_FACE,
                "入库失败：没有人脸数据。请等待目标正面朝向摄像头后重试。",
            )

        is_new_person = False
        if not person_id:
            # 传空表示创建新用户
            profile = PersonProfile.create_new(display_name=name)
            person_id = profile.person_id
            self.gallery[person_id] = profile
            is_new_person = True
            logger.info("已为 {} 创建新 profile (未提供 ID)", person_id)
        elif person_id not in self.gallery:
            # 传了 ID 但库里没有，视为错误
            logger.error("gallery 中未找到 Person ID {}", person_id)
            return ConfirmResult.fail(
                RegisterFailureReason.UNKNOWN_PERSON_ID,
                f"Person ID {person_id} not found in gallery",
            )

        profile = self.gallery[person_id]
        profile.display_name = name

        # 将当前 track 缓存的特征立即入库
        changes = self._update_gallery_person(person_id, state,
                                              min_face_quality=min_face_quality)

        # 新建用户必须真正写入至少一条人脸特征, 否则会产生"有 person 行但无人脸
        # 特征"的空人物 (embedding 存在但质量/尺寸未达入库门槛)。回滚并报质量不足。
        enrolled_face = any(op.kind == "face" for op in changes.feature_ops)
        if is_new_person and not enrolled_face:
            self.gallery.pop(person_id, None)
            logger.warning(
                "已回滚新 profile {}: 没有 face feature 通过 enroll threshold "
                "(face quality/尺寸不足)",
                person_id,
            )
            return ConfirmResult.fail(
                RegisterFailureReason.LOW_FACE_QUALITY,
                "入库失败：人脸画面不够清晰 (可能太远或角度偏)，未能记录有效特征。"
                "请正对摄像头并靠近一点后重试。",
            )

        # 统一走 save_lock + 共享 session 落库 (含 upsert_person_row)
        await self._save_gallery_person_incremental(person_id, changes)

        face_count = profile.total_face_features()
        body_count = sum(len(v) for v in profile.body_features.values())
        wardrobe_count = len(profile.wardrobe)
        logger.info(
            "已为 {} enroll feature: face={}, body={}, wardrobe={}",
            person_id, face_count, body_count, wardrobe_count,
        )

        state.identity_result = IdentityResult(
            person_id=person_id,
            display_name=name,
            status=IdentityStatus.DEFINITE,
            fused_score=1.0,
        )

        # Gallery 变动, 通知其他 track 重新匹配
        self._on_gallery_updated()

        self._emit_event(
            EventType.HUMAN_CONFIRMED,
            track_id=track_id,
            person_id=person_id,
            display_name=name,
            fused_score=1.0,
            source="human",
            message=f"Human confirmed {name}",
        )

        return ConfirmResult.ok(person_id)

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def events(self) -> list[SystemEvent]:
        """事件日志。"""
        return self.event_log

    def drain_new_events(self) -> list[SystemEvent]:
        """取出并清空待广播事件队列。"""
        events = self.new_events
        self.new_events = []
        return events

    # ==================================================================
    # Per-track 状态管理 (原 CameraPipeline)
    # ==================================================================
    def _refresh_track_states(self, persons: list[TrackedPerson]) -> None:
        """同步活跃轨迹状态，清理失效轨迹。"""
        active_ids = set()

        # 创建或更新当前帧的 active states
        for person in persons:
            tid = person.track_id
            if tid not in self.tracks:
                self.tracks[tid] = TrackState(person=person)
            else:
                # 每帧更新 person 引用
                self.tracks[tid].person = person
            active_ids.add(person.track_id)

        # 清理失效 states
        stale = set(self.tracks.keys()) - active_ids
        for tid in stale:
            self.tracks[tid].cancel_vlm()
            del self.tracks[tid]

    def _on_gallery_updated(self) -> None:
        """Gallery 更新后，所有非 DEFINITE 的 track 标记 force_probe。"""
        for state in self.tracks.values():
            if state.identity_result.status != IdentityStatus.DEFINITE:
                state.force_probe = True

    def _update_gallery_person(
            self,
            person_id: str,
            state: TrackState,
            min_face_quality: float | None = None,
    ) -> GalleryUpdateResult:
        """统一的 Gallery 更新：从 QualityCache 写入特征到底库。

        Args:
            min_face_quality: 人脸入库质量门槛下限(可选), 与配置阈值取较大值
                生效; 只作用于人脸, 人体门槛不受影响。

        Returns:
            GalleryUpdateResult: 包含了变动的特征和衣橱更新标记。
        """
        changes = GalleryUpdateResult()

        profile = self.gallery.get(person_id)
        if not profile:
            return changes

        cache = state.quality_cache
        gallery_cfg = get_config().gallery
        min_face_size = get_config().face.min_face_size

        # --- 人脸 / 人体特征入库 ---
        body_best: dict[PoseBucket, tuple[CachedFrame, float]] = {}
        enroll_scorer = get_ediffiqa_enroll()

        for pool, enroll_fn, with_face in [
            (cache.face_pool, profile.enroll_face, True),
            (cache.body_pool, profile.enroll_body_feature, False),
        ]:
            # 人脸与人体使用各自的入库质量门槛 (人脸要求更高);
            # 调用方可给人脸门槛下限(min_face_quality), 与配置阈值取较大值生效
            if with_face:
                quality_threshold = gallery_cfg.face_quality_enroll_threshold
                if min_face_quality is not None:
                    quality_threshold = max(quality_threshold, min_face_quality)
            else:
                quality_threshold = gallery_cfg.body_quality_enroll_threshold
            # 每个姿态桶选最高质量的未入库帧
            best_frames: dict[PoseBucket, tuple[CachedFrame, float]] = {}
            for cf in pool:
                if cf.enrolled:
                    continue
                # 人脸: 尺寸门槛 (短边像素) + 入库专用模型重评质量
                quality = cf.quality
                if with_face:
                    fb = cf.entry.face_bbox
                    if fb is None:
                        continue
                    face_short_edge = min(float(fb[2] - fb[0]), float(fb[3] - fb[1]))
                    if face_short_edge < min_face_size:
                        continue  # 脸太小, 分辨率不足, 不入库
                    if cf.entry.aligned_face is not None:
                        quality = enroll_scorer.predict(cf.entry.aligned_face)
                if quality >= quality_threshold:
                    pose = cf.entry.detection.pose_bucket
                    if pose not in best_frames or quality > best_frames[pose][1]:
                        best_frames[pose] = (cf, quality)

            for pose, (cf, quality) in best_frames.items():
                overlay_bbox = None
                if with_face:
                    # 人脸特征: source_image = crop, overlay_bbox = face_bbox (crop 坐标系)
                    if cf.entry.face_bbox is not None:
                        x1, y1, x2, y2 = cf.entry.face_bbox[:4].tolist()
                        h, w = cf.entry.crop.shape[:2]
                        overlay_bbox = [
                            max(0.0, x1), max(0.0, y1),
                            min(float(w), x2), min(float(h), y2),
                        ]
                    _, buf = cv2.imencode('.png', cf.entry.crop)
                    source_image = buf.tobytes()
                else:
                    # 人体特征: source_image = 全帧原图, overlay_bbox = body bbox (全帧坐标系)
                    _, buf = cv2.imencode('.png', cf.entry.frame_snapshot)
                    source_image = buf.tobytes()
                    bx = cf.entry.detection.bbox
                    overlay_bbox = [
                        float(bx[0]), float(bx[1]),
                        float(bx[2]), float(bx[3]),
                    ]

                entry = FeatureEntry(
                    embedding=cf.embedding,
                    pose_bucket=pose,
                    quality_score=quality,
                    timestamp=cf.entry.timestamp,
                    source_image=source_image,
                    overlay_bbox=overlay_bbox,
                )
                if op := enroll_fn(entry):
                    changes.feature_ops.append(op)
                    cf.enrolled = True

            if not with_face:
                body_best = best_frames

        # 服装: 最高质量 body embedding
        if body_best:
            best_cf, best_q = max(body_best.values(), key=lambda t: t[1])
            changes.wardrobe_op = profile.enroll_outfit(
                best_cf.embedding, best_q,
            )

        # 体型比例 (存在 PersonRow 中, 由 upsert_person_row 保存)
        proportions = MultiFrameAggregator.aggregate_proportions(cache.body_pool)
        if proportions is not None:
            profile.update_proportions(proportions)

        logger.info("已更新 gallery, person_id={}", person_id)
        return changes

    def _emit_match_event(
            self, track_id: int, result: MatchResult,
    ) -> None:
        """每次 Tier2 执行完后发出匹配事件 (含候选详情)。"""
        # 构建候选详情
        candidates_detail = []
        for c in result.candidates[:5]:  # top 5
            candidates_detail.append({
                "person_id": c.person_id,
                "display_name": c.display_name,
                "fused_score": round(c.fused_score, 3),
                "face_score": round(c.face_score, 3) if c.face_score is not None else None,
                "body_score": round(c.body_score, 3) if c.body_score is not None else None,
                "proportion_score": round(c.proportion_score, 3) if c.proportion_score is not None else None,
                "face_match_quality": round(c.face_match_quality, 2),
                "body_match_quality": round(c.body_match_quality, 2),
                "face_weight": round(c.face_weight, 2),
                "body_weight": round(c.body_weight, 2),
            })

        # 映射 status → event_type
        status_event_map = {
            IdentityStatus.DEFINITE: EventType.IDENTITY_DEFINITE,
            IdentityStatus.CONFIDENT: EventType.IDENTITY_CONFIDENT,
            IdentityStatus.SUSPECTED: EventType.IDENTITY_SUSPECTED,
            IdentityStatus.CONFLICT: EventType.IDENTITY_CONFLICT,
            IdentityStatus.STRANGER: EventType.NEW_PERSON,
        }
        event_type = status_event_map.get(result.status, EventType.NEW_PERSON)

        best = result.best_match
        self._emit_event(
            event_type,
            track_id=track_id,
            person_id=best.person_id if best else None,
            display_name=best.display_name if best else None,
            fused_score=best.fused_score if best else None,
            source="reid",
            message=f"{result.status.value}",
            candidates=candidates_detail,
        )

    def _select_target(self) -> None:
        """选择注意力最高的目标（带滞后防抖）。

        当前目标享有 _HYSTERESIS_MARGIN 的防抖优势，
        只有当新目标的分数超过当前目标分数 + margin 时才会切换，
        避免两人分数接近时目标来回跳动。
        """

        # 目标切换滞后阈值
        _HYSTERESIS_MARGIN: float = 0.1

        active_states = self.tracks.values()
        if not active_states:
            self.current_target_id = None
            return

        for s in active_states:
            s.is_current_target = False

        best = max(active_states, key=lambda s: s.person.attention_score)

        # 如果当前有目标且仍然活跃，应用滞后判断
        if self.current_target_id is not None:
            current_state = next(
                (s for s in active_states if s.person.track_id == self.current_target_id),
                None,
            )
            if current_state is not None:
                current_score = current_state.person.attention_score
                best_score = best.person.attention_score
                # 新目标必须超过当前目标 + margin 才切换
                if best_score < current_score + _HYSTERESIS_MARGIN:
                    best = current_state

        best.is_current_target = True
        self.current_target_id = best.person.track_id

    # ==================================================================
    # 响应构建
    # ==================================================================

    def _build_response(self, elapsed_ms: float) -> dict[str, object]:
        """构建 API 响应字典。"""
        tracked = []
        for s in self.tracks.values():
            tracked.append({
                "person": s.person,
                "identity_result": s.identity_result,
                "is_current_target": s.is_current_target,
            })

        current_target_info = None
        if self.current_target_id is not None:
            ts = self.tracks.get(self.current_target_id)
            if ts is not None:
                tir = ts.identity_result
                current_target_info = {
                    "track_id": self.current_target_id,
                    "person_id": tir.person_id,
                    "display_name": tir.display_name,
                }

        return {
            "frame_id": self.frame_count,
            "tracked_persons": tracked,
            "current_target": current_target_info,
            "pending_vlm": [s.person.track_id for s in self.tracks.values() if s.vlm_task is not None],
            "gallery_size": len(self.gallery),
            "processing_ms": round(elapsed_ms, 1),
        }

    def _build_debug(
            self,
            tier1_ms: float,
            total_ms: float,
            face_detect_ms: float = 0.0,
            face_assess_ms: float = 0.0,
    ) -> dict[str, object]:
        """构建 pipeline_debug 字典。

        tier1_ms 已包含 face_detect + face_assess, 这里做拆分展示:
        - detection: 纯检测+追踪+注意力 (= tier1_ms - face)
        - face_detect: 人脸检测耗时 (Tier1 子阶段)
        - face_assess: 质量评估耗时 (Tier1 子阶段)
        - reid: Tier2 耗时 (= total - tier1)
        """
        active_states = self.tracks.values()
        n_states = len(active_states)
        n_vlm_pending = sum(1 for s in active_states if s.vlm_task is not None)

        # tier1_ms 已包含 face, 拆出纯检测时间
        detect_ms = tier1_ms - face_detect_ms - face_assess_ms
        reid_ms = total_ms - tier1_ms

        # 人脸检测统计
        n_face_detected = sum(
            1 for s in active_states
            if s.person.detection and s.person.detection.pose_bucket != PoseBucket.BACK
        )
        face_detect_status = "done" if face_detect_ms > 0 else "skipped"
        face_assess_status = "done" if face_assess_ms > 0 else "skipped"

        if reid_ms > 1.0:  # 有显著 ReID 耗时
            return {
                "detection": {"status": "done", "time_ms": round(detect_ms, 1), "details": {"count": n_states}},
                "face_detect": {"status": face_detect_status, "time_ms": round(face_detect_ms, 1),
                                "details": {"detected": n_face_detected, "total": n_states}},
                "face_assess": {"status": face_assess_status, "time_ms": round(face_assess_ms, 1),
                                "details": {
                                    "results": [
                                        {"track_id": s.person.track_id,
                                         "face_match_quality": s.identity_result.face_match_quality,
                                         "body_match_quality": s.identity_result.body_match_quality}
                                        for s in active_states],
                                }},
                "reid": {"status": "done", "time_ms": round(max(0, reid_ms), 1),
                         "details": {"multiframe": True}},
            }

        pending_status = "running" if n_vlm_pending > 0 else "pending"
        return {
            "detection": {"status": "done", "time_ms": round(detect_ms, 1), "details": {"count": n_states}},
            "face_detect": {"status": face_detect_status, "time_ms": round(face_detect_ms, 1),
                            "details": {"detected": n_face_detected, "total": n_states}},
            "face_assess": {"status": face_assess_status, "time_ms": round(face_assess_ms, 1), "details": {}},
            "reid": {"status": pending_status, "time_ms": 0, "details": {}},
        }

    # ==================================================================
    # 事件系统
    # ==================================================================

    def _emit_event(
            self,
            event_type: EventType,
            track_id: int | None = None,
            person_id: str | None = None,
            display_name: str | None = None,
            fused_score: float | None = None,
            source: str = "system",
            message: str = "",
            candidates: list[dict] | None = None,
    ) -> None:
        """发出系统事件并加入日志。"""
        event = SystemEvent(
            event_type=event_type,
            track_id=track_id,
            person_id=person_id,
            display_name=display_name,
            fused_score=fused_score,
            source=source,
            message=message,
            candidates=candidates or [],
        )
        self.event_log.append(event)
        self.new_events.append(event)

        if len(self.event_log) > self.max_events:
            self.event_log = self.event_log[-self.max_events:]

        logger.info(
            "Event: {} | track={} person={} | {}",
            event_type.value, track_id, person_id, message,
        )

    # ==================================================================
    # Gallery 持久化
    # ==================================================================

    async def _load_gallery_from_db(self) -> None:
        """从 SQLite 加载当前摄像头的底库。"""
        persistence = get_gallery_persistence()
        self.gallery = await persistence.load_all_profiles(self.camera_id)

    async def _save_gallery_person_incremental(
            self,
            person_id: str,
            changes: GalleryUpdateResult,
    ) -> None:
        """增量持久化: 只写入 _update_gallery 实际变动的部分。

        使用共享 session 将所有 DB 操作合并为单一事务,
        commit 前检查 gallery 成员关系防止竞态写回。
        """
        # 快速检查: 不在 gallery 中则跳过 (已删除或从未存在)
        profile = self.gallery.get(person_id)
        if not profile:
            return

        persistence = get_gallery_persistence()
        from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession

        async with self.save_lock:
            # 拿锁后再检查 (delete 可能在等锁期间执行)
            if person_id not in self.gallery:
                return

            async with _AsyncSession(persistence.engine) as session:
                # 1. persons 表
                person_row = await persistence.get_person_row(
                    session, profile.person_id, self.camera_id,
                )
                persistence.upsert_person_row_in(
                    session, profile, self.camera_id, person_row,
                )

                # 2. 特征操作
                for op in changes.feature_ops:
                    if op.evicted is None:
                        persistence.add_feature_in(
                            session, person_id, op.entry, op.kind,
                        )
                    else:
                        await persistence.replace_feature_in(
                            session, person_id, op.entry, op.evicted, op.kind,
                        )

                # 3. 衣橱
                wop = changes.wardrobe_op
                if wop is not None:
                    if wop.previous is not None:
                        await persistence.update_outfit_in(
                            session, person_id, wop.previous, wop.outfit,
                        )
                    elif wop.evicted is not None:
                        await persistence.replace_outfit_in(
                            session, person_id, wop.outfit, wop.evicted,
                        )
                    else:
                        persistence.add_outfit_in(
                            session, person_id, wop.outfit,
                        )

                # 提交前最终检查
                if person_id not in self.gallery:
                    logger.debug(
                        "已中止保存 {} (事务期间被删除)",
                        person_id,
                    )
                    return  # session close 自动 rollback

                await session.commit()

        logger.debug(
            "增量保存 {}: {} 个 feature ops, wardrobe={}",
            person_id,
            len(changes.feature_ops),
            changes.wardrobe_op is not None,
        )
