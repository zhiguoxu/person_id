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
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_config
from src.gallery.data_models import (
    EventType,
    IdentityResult,
    IdentityStatus,
    MatchResult,
    PersonProfile,
    SystemEvent,
    TrackedPerson,
)
from src.gallery.data_models import FeatureEntry
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
        logger.info("Initializing VisionOrchestrator (camera={}) ...", camera_id)
        orch = cls(camera_id=camera_id)
        await orch._load_gallery_from_db()

        logger.info(
            "VisionOrchestrator [{}] initialized (gallery: {} persons)",
            camera_id, len(orch.gallery),
        )
        return orch

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def shutdown(self) -> None:
        """清理资源: 取消异步任务、保存底库。"""
        logger.info("Shutting down VisionOrchestrator ...")

        for state in self.tracks.values():
            state.cancel_vlm()

        await self._save_gallery_to_db()
        self.tracks.clear()
        logger.info("VisionOrchestrator shutdown complete")

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

        # --- Tier 1: 检测 + 追踪 + 注意力 ---
        persons = self.tier1.process(frame)
        tier1_ms = (time.perf_counter() - t0) * 1000

        # --- 刷新 TrackState ---
        self._refresh_track_states(persons)

        # --- track 逐个进处理 ---
        gallery_dirty = False
        for state in self.tracks.values():
            result, is_enrich = state.process_frame(frame, self.gallery)
            if result is None:
                continue

            if not is_enrich:
                self._emit_match_event(state.person.track_id, result)

            if result.status == IdentityStatus.DEFINITE:
                gallery_dirty = True
                self._update_gallery(result, state)

        # 统一：设 force_probe + 异步落盘
        if gallery_dirty:
            self._on_gallery_updated()
            asyncio.ensure_future(self._save_gallery_to_db())

        # --- 注意力选择 ---
        self._select_target()

        # --- 构建响应 ---
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response = self._build_response(elapsed_ms)
        response["pipeline_debug"] = self._build_debug(tier1_ms, elapsed_ms)
        return response

    # ==================================================================
    # Human-in-the-loop
    # ==================================================================

    async def confirm_identity(
            self, track_id: int, person_id: str, name: str
    ) -> None:
        """人工确认身份。

        Args:
            track_id: 需要确认的轨迹 ID。
            person_id: 底库人物 ID（如不存在则自动创建）。
            name: 显示名称。
        """
        logger.info(
            "Human confirmed: track_id={} → person_id={}, name={}",
            track_id, person_id, name,
        )

        if person_id not in self.gallery:
            profile = PersonProfile.create_new(display_name=name)
            profile.person_id = person_id
            self.gallery[person_id] = profile
            logger.info("Created new profile for {}", person_id)

        profile = self.gallery[person_id]
        profile.display_name = name
        profile.touch()

        state = self.tracks.get(track_id)
        if state is not None:
            state.identity_result = IdentityResult(
                person_id=person_id,
                display_name=name,
                status=IdentityStatus.CONFIDENT,
                confidence=1.0,
            )

        self._emit_event(
            EventType.HUMAN_CONFIRMED,
            track_id=track_id,
            person_id=person_id,
            display_name=name,
            confidence=1.0,
            source="human",
            message=f"Human confirmed {name}",
        )

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
        """同步活跃轨迹状态，清理失效轨迹，返回当前活跃的 TrackState 列表。"""
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

    def _on_gallery_updated(self) -> None:
        """Gallery 更新后，所有非 DEFINITE 的 track 标记 force_probe。"""
        for state in self.tracks.values():
            if state.identity_result.status != IdentityStatus.DEFINITE:
                state.force_probe = True

    def _update_gallery(
            self,
            result: MatchResult,
            state: TrackState,
    ) -> None:
        """统一的 Gallery 更新：从 QualityCache 写入特征到底库。

        Tier2 和 Tier3 的 DEFINITE 结果均通过此方法更新 Gallery。
        """
        if not result.best_match:
            return

        pid = result.best_match.person_id
        profile = self.gallery.get(pid)
        if not profile:
            return

        cache = state.quality_cache
        gallery_cfg = get_config().gallery

        # 人脸: 每个姿态桶取质量最高的帧
        face_best: dict = {}
        if cache.face_pool:
            for cf in cache.face_pool:
                if cf.face_quality >= gallery_cfg.quality_enroll_threshold:
                    pose = cf.entry.pose_bucket
                    if pose not in face_best or cf.face_quality > face_best[pose].face_quality:
                        face_best[pose] = cf

        for pose, cf in face_best.items():
            profile.enroll_face(FeatureEntry(
                embedding=cf.face_embedding,
                pose_bucket=pose,
                quality_score=cf.face_quality,
                timestamp=cf.entry.timestamp,
            ))

        # 人体: 每个姿态桶取质量最高的帧
        body_best: dict = {}
        if cache.body_pool:
            for cf in cache.body_pool:
                pose = cf.entry.pose_bucket
                if pose not in body_best or cf.body_quality > body_best[pose].body_quality:
                    body_best[pose] = cf

        for pose, cf in body_best.items():
            profile.enroll_body_feature(FeatureEntry(
                embedding=cf.body_embedding,
                pose_bucket=pose,
                quality_score=cf.body_quality,
                timestamp=cf.entry.timestamp,
            ))

        # 服装: 最高质量 body embedding
        if body_best:
            best_cf = max(body_best.values(), key=lambda cf: cf.body_quality)
            profile.enroll_outfit(best_cf.body_embedding, best_cf.body_quality)

        # 体型比例
        proportions = MultiFrameAggregator.aggregate_proportions(cache.body_pool)
        if proportions is not None:
            profile.update_proportions(proportions)

        profile.touch(time.time())
        logger.info("Gallery updated for person_id={}", pid)

    def _emit_match_event(
            self, track_id: int, result: MatchResult,
    ) -> None:
        """根据匹配结果发出事件（带全局冷却）。"""
        now = time.time()
        event_key = f"{result.status.value}:{track_id}"
        if (now - self.last_event_time.get(event_key, 0)) <= self.event_cooldown_sec:
            return

        self.last_event_time[event_key] = now

        if result.status == IdentityStatus.CONFIDENT and result.best_match:
            self._emit_event(
                EventType.IDENTITY_CONFIRMED,
                track_id=track_id,
                person_id=result.best_match.person_id,
                display_name=result.best_match.display_name,
                confidence=result.best_match.fused_score,
                source="reid",
            )
        elif result.status == IdentityStatus.CONFLICT:
            self._emit_event(
                EventType.IDENTITY_CONFLICT,
                track_id=track_id,
                source="reid",
                message=f"Conflict among {len(result.candidates)} candidates",
            )
        elif result.status == IdentityStatus.STRANGER:
            self._emit_event(
                EventType.NEW_PERSON,
                track_id=track_id,
                source="system",
                message="New person detected (stranger)",
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
                if best_score < current_score + self._HYSTERESIS_MARGIN:
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
    ) -> dict[str, object]:
        """构建 pipeline_debug 字典。"""
        active_states = self.tracks.values()
        n_states = len(active_states)
        n_identified = sum(1 for s in active_states if s.identity_result.person_id)
        n_identifying = sum(1 for s in active_states if s.identity_result.status == IdentityStatus.IDENTIFYING)
        n_vlm_pending = sum(1 for s in active_states if s.vlm_task is not None)
        has_tier2 = total_ms - tier1_ms > 1.0  # 有显著 tier2 耗时

        if has_tier2:
            tier2_ms = total_ms - tier1_ms
            return {
                "detection": {"status": "done", "time_ms": round(tier1_ms, 1), "details": {"count": n_states}},
                "pose": {"status": "done", "time_ms": 0, "details": {}},
                "face": {"status": "done", "time_ms": 0, "details": {
                    "results": [
                        {"track_id": s.person.track_id,
                         "extracted": s.identity_result.face_quality is not None,
                         "quality": s.identity_result.face_quality}
                        for s in active_states],
                }},
                "reid": {"status": "done", "time_ms": round(tier2_ms, 1),
                         "details": {"multiframe": True}},
                "matching": {"status": "done", "time_ms": 0, "details": {
                    "results": [
                        {"track_id": s.person.track_id,
                         "decision": s.identity_result.status.value,
                         "matched_id": s.identity_result.person_id,
                         "candidates": []}
                        for s in active_states],
                }},
                "identity": {"status": "done", "time_ms": 0,
                             "details": {"confirmed": n_identified, "identifying": n_identifying,
                                         "vlm_pending": n_vlm_pending}},
            }

        pending_status = "running" if n_vlm_pending > 0 else "pending"
        return {
            "detection": {"status": "done", "time_ms": round(tier1_ms, 1), "details": {"count": n_states}},
            "pose": {"status": pending_status, "time_ms": 0, "details": {}},
            "face": {"status": pending_status, "time_ms": 0, "details": {}},
            "reid": {"status": pending_status, "time_ms": 0, "details": {}},
            "matching": {"status": pending_status, "time_ms": 0, "details": {}},
            "identity": {"status": pending_status, "time_ms": 0, "details": {
                "confirmed": n_identified, "identifying": n_identifying, "vlm_pending": n_vlm_pending,
            }},
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
            confidence: float | None = None,
            source: str = "system",
            message: str = "",
    ) -> None:
        """发出系统事件并加入日志。"""
        event = SystemEvent(
            event_type=event_type,
            track_id=track_id,
            person_id=person_id,
            display_name=display_name,
            confidence=confidence,
            source=source,
            message=message,
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
        try:
            from src.gallery import load_gallery
            self.gallery = await load_gallery(
                get_config().server.gallery_db_path,
                camera_id=self.camera_id,
            )
            logger.info(
                "[{}] Loaded {} persons from DB",
                self.camera_id, len(self.gallery),
            )
        except ImportError:
            logger.warning("Gallery persistence not available; starting with empty gallery")
            self.gallery = {}
        except Exception:
            logger.exception("Failed to load gallery from DB")
            self.gallery = {}

    async def _save_gallery_to_db(self) -> None:
        """保存当前摄像头的底库到 SQLite（串行化，防止并发写入）。"""
        async with self.save_lock:
            try:
                from src.gallery import save_gallery
                await save_gallery(
                    get_config().server.gallery_db_path,
                    self.gallery,
                    camera_id=self.camera_id,
                )
                logger.info(
                    "[{}] Saved {} persons to DB",
                    self.camera_id, len(self.gallery),
                )
            except ImportError:
                logger.warning("Gallery persistence not available; data not saved")
            except Exception:
                logger.exception("Failed to save gallery to DB")
