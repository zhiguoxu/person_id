"""
视觉编排器 — 主流水线入口

将 Tier 1（逐帧追踪）与 Tier 2（深度识别）串联，管理底库、
事件系统、注意力选择，并为 API 层提供统一调用入口。

典型调用链:
  frame → Tier1.process() → 判定哪些 track 需要 Tier2 →
  Tier2.process()（异步） → 更新缓存 / 底库 → 注意力选择 →
  构造响应 dict
"""
from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from src.config import Config
from src.gallery.data_models import (
    EventType,
    IdentityStatus,
    MatchResult,
    PersonProfile,
    PipelineDebug,
    SystemEvent,
    TrackedPerson,
)
from src.pipeline.temporal_aggregator import TemporalAggregator
from src.pipeline.tier1 import Tier1Processor
from src.pipeline.tier2 import Tier2Processor


class VisionOrchestrator:
    """
    视觉流水线主编排器。

    负责初始化所有子模块、协调 Tier1 / Tier2 处理流程、
    管理底库、发出系统事件，以及选择注意力目标。

    Args:
        config: 全局配置对象。
    """

    def __init__(self, config: Config) -> None:
        self._config = config

        # --- Sub-modules (延迟初始化) ---
        self._detector_fast: Any = None
        self._detector_heavy: Any = None
        self._tracker: Any = None
        self._face_extractor: Any = None
        self._body_extractor: Any = None
        self._gallery_matcher: Any = None
        self._gallery_updater: Any = None
        self._reranker: Any = None
        self._fusion: Any = None
        self._resolver: Any = None
        self._vlm: Any = None
        self._attention_engine: Any = None

        # --- Pipeline processors ---
        self._tier1: Optional[Tier1Processor] = None
        self._tier2: Optional[Tier2Processor] = None
        self._temporal_aggregator = TemporalAggregator(
            window_size=config.temporal.window_size
        )

        # --- Gallery ---
        self._gallery: dict[str, PersonProfile] = {}

        # --- State ---
        self._pending_tier2: dict[int, asyncio.Task] = {}
        self._tier2_results: dict[int, MatchResult] = {}
        self._tier2_debug: dict[int, Any] = {}  # track_id → PipelineDebug
        self._current_target_id: Optional[int] = None
        self._event_log: list[SystemEvent] = []
        self._new_events: list[SystemEvent] = []  # 待广播队列
        self._max_events = 200  # 保留最近 N 条事件

        # --- Frame counter ---
        self._frame_count = 0
        self._initialized = False
        # 事件去重: status_value → 上次发送时间
        self._last_event_time: dict[str, float] = {}
        self._event_cooldown_sec = 10.0  # 同类事件全局冷却 10 秒

        logger.info("VisionOrchestrator created")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def initialize(self) -> None:
        """
        初始化所有子模块: 加载模型、连接数据库。

        应在开始处理帧之前调用一次。
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized, skipping")
            return

        logger.info("Initializing VisionOrchestrator ...")

        # --- 检测器 ---
        try:
            from src.detection import create_detector  # type: ignore

            self._detector_fast = create_detector(
                self._config, tier="fast"
            )
            self._detector_heavy = create_detector(
                self._config, tier="heavy"
            )
            logger.info("Detectors loaded")
        except Exception as e:
            logger.warning(
                "Detection module not available ({}); using stub detectors", e
            )
            self._detector_fast = _StubDetector()
            self._detector_heavy = _StubDetector()

        # --- 追踪器 ---
        try:
            from src.tracking import create_tracker  # type: ignore

            self._tracker = create_tracker(self._config)
            logger.info("Tracker loaded")
        except Exception as e:
            logger.warning("Tracking module not available ({}); using stub tracker", e)
            self._tracker = _StubTracker()

        # --- 人脸提取 ---
        try:
            from src.features import create_face_extractor  # type: ignore

            self._face_extractor = create_face_extractor(self._config)
            logger.info("Face extractor loaded")
        except Exception as e:
            logger.warning("Face extractor not available ({}); using stub", e)
            self._face_extractor = _StubFaceExtractor()

        # --- Body ReID ---
        try:
            from src.features import create_body_extractor  # type: ignore

            self._body_extractor = create_body_extractor(self._config)
            logger.info("Body extractor loaded")
        except Exception as e:
            logger.warning("Body extractor not available ({}); using stub", e)
            self._body_extractor = _StubBodyExtractor()

        # --- Gallery 匹配 / 更新 ---
        try:
            from src.gallery import (  # type: ignore
                create_gallery_matcher,
                create_gallery_updater,
            )

            self._gallery_matcher = create_gallery_matcher(self._config)
            self._gallery_updater = create_gallery_updater(self._config)
            logger.info("Gallery matcher/updater loaded")
        except Exception as e:
            logger.warning("Gallery module not available ({}); using stubs", e)
            self._gallery_matcher = _StubMatcher()
            self._gallery_updater = _StubUpdater()

        # --- Reranker / Fusion / Resolver ---
        try:
            from src.identity import (  # type: ignore
                create_reranker,
                create_fusion,
                create_resolver,
            )

            self._reranker = create_reranker(self._config)
            self._fusion = create_fusion(self._config)
            self._resolver = create_resolver(self._config)
            logger.info("Identity modules loaded")
        except Exception as e:
            logger.warning("Identity modules not available ({}); using stubs", e)
            self._reranker = _StubReranker()
            self._fusion = _StubFusion()
            self._resolver = _StubResolver()

        # --- VLM ---
        try:
            from src.perception import create_vlm  # type: ignore

            self._vlm = create_vlm(self._config)
            logger.info("VLM loaded")
        except Exception as e:
            logger.warning("VLM module not available ({}); using stub", e)
            self._vlm = _StubVLM()

        # --- Attention ---
        try:
            from src.attention import create_attention_engine  # type: ignore

            self._attention_engine = create_attention_engine(self._config)
            logger.info("Attention engine loaded")
        except Exception as e:
            logger.warning("Attention engine not available ({})", e)
            self._attention_engine = None

        # --- 构建 Tier 处理器 ---
        self._tier1 = Tier1Processor(
            config=self._config,
            detector=self._detector_fast,
            tracker=self._tracker,
            attention_engine=self._attention_engine,
        )

        self._tier2 = Tier2Processor(
            config=self._config,
            heavy_detector=self._detector_heavy,
            face_extractor=self._face_extractor,
            body_extractor=self._body_extractor,
            gallery_matcher=self._gallery_matcher,
            gallery_updater=self._gallery_updater,
            reranker=self._reranker,
            fusion=self._fusion,
            resolver=self._resolver,
            vlm=self._vlm,
            temporal_aggregator=self._temporal_aggregator,
        )

        # --- 加载底库 ---
        await self._load_gallery_from_db()

        self._initialized = True
        logger.info(
            "VisionOrchestrator initialized (gallery: {} persons)",
            len(self._gallery),
        )

    async def shutdown(self) -> None:
        """清理资源: 取消异步任务、保存底库、释放模型。"""
        logger.info("Shutting down VisionOrchestrator ...")

        # 取消所有挂起的 Tier 2 任务
        for task in self._pending_tier2.values():
            task.cancel()
        self._pending_tier2.clear()

        # 持久化底库
        await self._save_gallery_to_db()

        # 清理
        self._temporal_aggregator.clear()
        self._initialized = False
        logger.info("VisionOrchestrator shutdown complete")

    # ==================================================================
    # Frame Processing
    # ==================================================================

    async def process_frame(self, frame: np.ndarray) -> dict:
        """
        处理一帧图像并返回结果字典。

        Args:
            frame: BGR 图像 (H, W, 3)。

        Returns:
            包含 tracked_persons, current_target, events 的字典。
        """
        if not self._initialized:
            return {"error": "not_initialized", "tracked_persons": []}

        self._frame_count += 1
        t0 = time.perf_counter()

        # --- Tier 1 ---
        assert self._tier1 is not None
        t_det = time.perf_counter()
        persons = self._tier1.process(frame)
        tier1_ms = (time.perf_counter() - t_det) * 1000

        # --- 判定需要 Tier 2 的 track ---
        tier2_tracks = self._select_tier2_candidates(persons)

        # --- 启动 Tier 2 异步任务 ---
        for person in tier2_tracks:
            if person.track_id not in self._pending_tier2:
                task = asyncio.create_task(
                    self._run_tier2(frame, person)
                )
                self._pending_tier2[person.track_id] = task

        # --- 收集已完成的 Tier 2 结果 ---
        self._collect_tier2_results()

        # --- 应用 Tier 2 结果到 persons, 收集 debug ---
        latest_t2_debug = None
        for person in persons:
            result = self._tier2_results.pop(person.track_id, None)
            if result is not None:
                self._apply_match_result(person, result)
                # 取最新完成的 Tier 2 debug
                t2_dbg = self._tier2_debug.pop(person.track_id, None)
                if t2_dbg is not None:
                    latest_t2_debug = t2_dbg

        # --- 注意力选择 ---
        current_target = self._select_target(persons)

        # --- 构建响应 ---
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # --- 构建 pipeline_debug ---
        has_tier2 = latest_t2_debug is not None
        has_pending = len(self._pending_tier2) > 0
        n_identified = sum(1 for p in persons if p.person_id)
        n_identifying = sum(1 for p in persons if p.identity_status == IdentityStatus.IDENTIFYING)

        if has_tier2:
            # 使用真实的 Tier 2 调试数据
            t2 = latest_t2_debug
            pipeline_debug = {
                "detection": {
                    "status": "done",
                    "time_ms": round(tier1_ms, 1),
                    "details": {"count": len(persons)},
                },
                "pose": {
                    "status": t2.pose.status,
                    "time_ms": round(t2.pose.time_ms, 1),
                    "details": t2.pose.details,
                },
                "face": {
                    "status": t2.face.status,
                    "time_ms": round(t2.face.time_ms, 1),
                    "details": {
                        "results": [
                            {
                                "track_id": p.track_id,
                                "extracted": p.face_quality is not None,
                                "quality": p.face_quality,
                            }
                            for p in persons
                        ]
                    },
                },
                "reid": {
                    "status": t2.reid.status,
                    "time_ms": round(t2.reid.time_ms, 1),
                    "details": t2.reid.details,
                },
                "matching": {
                    "status": t2.matching.status,
                    "time_ms": round(t2.matching.time_ms, 1),
                    "details": {
                        "results": [
                            {
                                "track_id": p.track_id,
                                "decision": p.identity_status.value,
                                "matched_id": p.person_id,
                                "candidates": [],
                            }
                            for p in persons
                        ]
                    },
                },
                "identity": {
                    "status": t2.identity.status,
                    "time_ms": round(t2.identity.time_ms, 1),
                    "details": {
                        "confirmed": n_identified,
                        "identifying": n_identifying,
                        "vlm_pending": len(self._pending_tier2),
                    },
                },
            }
        else:
            # Tier 2 未完成, 显示 pending 状态
            pending_status = "running" if has_pending else "pending"
            pipeline_debug = {
                "detection": {
                    "status": "done",
                    "time_ms": round(tier1_ms, 1),
                    "details": {"count": len(persons)},
                },
                "pose": {"status": pending_status, "time_ms": 0, "details": {}},
                "face": {"status": pending_status, "time_ms": 0, "details": {}},
                "reid": {"status": pending_status, "time_ms": 0, "details": {}},
                "matching": {"status": pending_status, "time_ms": 0, "details": {}},
                "identity": {
                    "status": pending_status,
                    "time_ms": 0,
                    "details": {
                        "confirmed": n_identified,
                        "identifying": n_identifying,
                        "vlm_pending": len(self._pending_tier2),
                    },
                },
            }

        response = self._build_response(persons, current_target, elapsed_ms)
        response["pipeline_debug"] = pipeline_debug

        return response

    async def process_frame_with_debug(self, frame: np.ndarray) -> dict:
        """
        与 process_frame 相同，但包含详细调试信息和缩略图。

        Args:
            frame: BGR 图像 (H, W, 3)。

        Returns:
            带 pipeline_debug 和 thumbnails 的完整响应字典。
        """
        response = await self.process_frame(frame)

        # 附加缩略图
        for person_data in response.get("tracked_persons", []):
            track_id = person_data.get("track_id")
            if track_id is None:
                continue
            # 生成 base64 缩略图
            bbox = person_data.get("bbox")
            if bbox:
                thumb = self._generate_thumbnail(frame, bbox)
                person_data["thumbnail_b64"] = thumb

        response["debug"] = True
        return response

    # ==================================================================
    # Human-in-the-loop
    # ==================================================================

    async def confirm_identity(
        self, track_id: int, person_id: str, name: str
    ) -> None:
        """
        人工确认身份（Human-in-the-loop）。

        Args:
            track_id: 需要确认的轨迹 ID。
            person_id: 底库人物 ID（如不存在则自动创建）。
            name: 显示名称。
        """
        logger.info(
            "Human confirmed: track_id={} → person_id={}, name={}",
            track_id,
            person_id,
            name,
        )

        # 确保底库中存在该 profile
        if person_id not in self._gallery:
            profile = PersonProfile.create_new(display_name=name)
            profile.person_id = person_id
            self._gallery[person_id] = profile
            logger.info("Created new profile for {}", person_id)

        profile = self._gallery[person_id]
        profile.display_name = name
        profile.touch()

        # 更新 Tier 1 缓存
        if self._tier1:
            self._tier1.update_identity(
                track_id=track_id,
                person_id=person_id,
                display_name=name,
                status=IdentityStatus.CONFIDENT,
                confidence=1.0,
            )

        # 发出事件
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
    # Gallery access
    # ==================================================================

    @property
    def gallery(self) -> dict[str, PersonProfile]:
        """当前底库。"""
        return self._gallery

    @property
    def events(self) -> list[SystemEvent]:
        """事件日志。"""
        return self._event_log

    def drain_new_events(self) -> list[SystemEvent]:
        """取出并清空待广播事件队列。"""
        events = self._new_events
        self._new_events = []
        return events

    @property
    def config(self) -> Config:
        """当前配置。"""
        return self._config

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _select_tier2_candidates(
        self, persons: list[TrackedPerson]
    ) -> list[TrackedPerson]:
        """判定哪些 track 需要 Tier 2 深度识别。"""
        now = time.time()
        refresh_interval = self._config.tracking.tier2_refresh_interval_sec
        candidates: list[TrackedPerson] = []

        for p in persons:
            # 跳过已在处理中的
            if p.track_id in self._pending_tier2:
                continue

            needs_tier2 = False

            # 新目标 / 未识别
            if p.identity_status == IdentityStatus.IDENTIFYING:
                needs_tier2 = True

            # 疑似 / 冲突需要刷新
            elif p.identity_status in (
                IdentityStatus.SUSPECTED,
                IdentityStatus.CONFLICT,
            ):
                needs_tier2 = True

            # 定期刷新
            elif (now - p.last_tier2_time) > refresh_interval:
                needs_tier2 = True

            if needs_tier2:
                candidates.append(p)

        return candidates

    async def _run_tier2(
        self, frame: np.ndarray, person: TrackedPerson
    ) -> None:
        """异步运行 Tier 2 并存储结果。"""
        try:
            assert self._tier2 is not None
            result, debug = await self._tier2.process(frame, person, self._gallery)
            self._tier2_results[person.track_id] = result
            self._tier2_debug[person.track_id] = debug
        except asyncio.CancelledError:
            logger.debug("Tier2 task cancelled for track_id={}", person.track_id)
        except Exception:
            logger.exception(
                "Tier2 processing failed for track_id={}", person.track_id
            )
        finally:
            self._pending_tier2.pop(person.track_id, None)

    def _collect_tier2_results(self) -> None:
        """收集已完成的 Tier 2 任务。"""
        done_ids = []
        for track_id, task in self._pending_tier2.items():
            if task.done():
                done_ids.append(track_id)
                # 异常在 _run_tier2 中已处理
        for track_id in done_ids:
            self._pending_tier2.pop(track_id, None)

    def _apply_match_result(
        self, person: TrackedPerson, result: MatchResult
    ) -> None:
        """将 Tier 2 匹配结果应用到 TrackedPerson 和缓存。"""
        person.identity_status = result.status
        person.face_quality = result.face_quality

        if result.best_match:
            person.person_id = result.best_match.person_id
            person.display_name = result.best_match.display_name
            person.confidence = result.best_match.fused_score
        else:
            person.person_id = None
            person.display_name = None
            person.confidence = 0.0

        # 更新 Tier 1 缓存
        if self._tier1:
            self._tier1.update_identity(
                track_id=person.track_id,
                person_id=person.person_id,
                display_name=person.display_name,
                status=person.identity_status,
                confidence=person.confidence,
                face_quality=person.face_quality,
            )

        # 发出事件 (全局冷却: 同类事件 N 秒内只发一次)
        now = time.time()
        event_key = result.status.value  # 按状态全局冷却
        last_time = self._last_event_time.get(event_key, 0)

        if (now - last_time) > self._event_cooldown_sec:
            self._last_event_time[event_key] = now

            if result.status == IdentityStatus.CONFIDENT and result.best_match:
                self._emit_event(
                    EventType.IDENTITY_CONFIRMED,
                    track_id=person.track_id,
                    person_id=result.best_match.person_id,
                    display_name=result.best_match.display_name,
                    confidence=result.best_match.fused_score,
                    source="reid",
                )
            elif result.status == IdentityStatus.CONFLICT:
                self._emit_event(
                    EventType.IDENTITY_CONFLICT,
                    track_id=person.track_id,
                    source="reid",
                    message=f"Conflict among {len(result.candidates)} candidates",
                )
            elif result.status == IdentityStatus.STRANGER:
                self._emit_event(
                    EventType.NEW_PERSON,
                    track_id=person.track_id,
                    source="system",
                    message="New person detected (stranger)",
                )

    def _select_target(
        self, persons: list[TrackedPerson]
    ) -> Optional[TrackedPerson]:
        """选择注意力最高的目标。"""
        if not persons:
            self._current_target_id = None
            return None

        # 按注意力分降序排列
        best = max(persons, key=lambda p: p.attention_score)
        best.is_current_target = True
        self._current_target_id = best.track_id

        # 取消其他人的目标标记
        for p in persons:
            if p.track_id != best.track_id:
                p.is_current_target = False

        return best

    def _build_response(
        self,
        persons: list[TrackedPerson],
        target: Optional[TrackedPerson],
        elapsed_ms: float,
    ) -> dict:
        """构建 API 响应字典。"""
        tracked = []
        for p in persons:
            tracked.append(
                {
                    "track_id": p.track_id,
                    "bbox": p.detection.bbox.tolist() if p.detection.bbox is not None else None,
                    "keypoints": p.detection.keypoints.tolist() if p.detection.keypoints is not None else None,
                    "pose_bucket": p.detection.pose_bucket.value if p.detection.pose_bucket else None,
                    "person_id": p.person_id,
                    "display_name": p.display_name,
                    "identity_status": p.identity_status.value,
                    "confidence": round(p.confidence, 3),
                    "attention_score": round(p.attention_score, 3),
                    "is_current_target": p.is_current_target,
                    "face_quality": (
                        round(p.face_quality, 3)
                        if p.face_quality is not None
                        else None
                    ),
                    "trail": p.trail[-20:],  # 最近 20 点
                }
            )

        return {
            "frame_id": self._frame_count,
            "tracked_persons": tracked,
            "current_target": (
                {
                    "track_id": target.track_id,
                    "person_id": target.person_id,
                    "display_name": target.display_name,
                }
                if target
                else None
            ),
            "pending_tier2": list(self._pending_tier2.keys()),
            "gallery_size": len(self._gallery),
            "processing_ms": round(elapsed_ms, 1),
        }

    def _emit_event(
        self,
        event_type: EventType,
        track_id: Optional[int] = None,
        person_id: Optional[str] = None,
        display_name: Optional[str] = None,
        confidence: Optional[float] = None,
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
        self._event_log.append(event)
        self._new_events.append(event)  # 加入待广播队列

        # 限制日志大小
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events:]

        logger.info(
            "Event: {} | track={} person={} | {}",
            event_type.value,
            track_id,
            person_id,
            message,
        )

    @staticmethod
    def _generate_thumbnail(
        frame: np.ndarray, bbox: list, max_size: int = 96
    ) -> str:
        """生成 base64 编码的 JPEG 缩略图。"""
        try:
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return ""

            # 缩放
            ch, cw = crop.shape[:2]
            scale = min(max_size / max(ch, cw, 1), 1.0)
            if scale < 1.0:
                crop = cv2.resize(
                    crop, (int(cw * scale), int(ch * scale))
                )

            _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Gallery persistence
    # ------------------------------------------------------------------

    async def _load_gallery_from_db(self) -> None:
        """从 SQLite 加载底库（stub — 待 gallery 持久化模块实现）。"""
        try:
            from src.gallery import load_gallery  # type: ignore

            self._gallery = await load_gallery(
                self._config.server.gallery_db_path
            )
            logger.info(
                "Loaded {} persons from DB", len(self._gallery)
            )
        except ImportError:
            logger.warning(
                "Gallery persistence not available; starting with empty gallery"
            )
            self._gallery = {}
        except Exception:
            logger.exception("Failed to load gallery from DB")
            self._gallery = {}

    async def _save_gallery_to_db(self) -> None:
        """保存底库到 SQLite（stub — 待 gallery 持久化模块实现）。"""
        try:
            from src.gallery import save_gallery  # type: ignore

            await save_gallery(
                self._config.server.gallery_db_path, self._gallery
            )
            logger.info("Saved {} persons to DB", len(self._gallery))
        except ImportError:
            logger.warning("Gallery persistence not available; data not saved")
        except Exception:
            logger.exception("Failed to save gallery to DB")


# ======================================================================
# Stub implementations (used when sub-modules are not yet available)
# ======================================================================


class _StubDetector:
    """占位检测器。"""

    def detect(self, frame: np.ndarray) -> list:
        return []


class _StubTracker:
    """占位追踪器。"""

    def update(self, detections: list, frame: np.ndarray) -> list:
        return []


class _StubFaceExtractor:
    """占位人脸提取器。"""

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        return None


class _StubBodyExtractor:
    """占位 ReID 提取器。"""

    def extract(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        return None


class _StubMatcher:
    """占位匹配器。"""

    def match(self, *args: Any, **kwargs: Any) -> list:
        return []


class _StubUpdater:
    """占位更新器。"""

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass


class _StubReranker:
    """占位重排序器。"""

    def rerank(self, candidates: list, *args: Any) -> list:
        return candidates


class _StubFusion:
    """占位融合器。"""

    def fuse(self, candidates: list, *args: Any) -> list:
        return candidates


class _StubResolver:
    """占位消解器。"""

    def resolve_reid(self, match_result: MatchResult) -> MatchResult:
        from src.gallery.data_models import IdentityStatus
        match_result.status = IdentityStatus.STRANGER
        return match_result

    def resolve_vlm(self, vlm_response: dict, match_result: MatchResult) -> MatchResult:
        from src.gallery.data_models import IdentityStatus
        match_result.status = IdentityStatus.STRANGER
        return match_result


class _StubVLM:
    """占位 VLM。"""

    async def arbitrate(self, *args: Any, **kwargs: Any) -> MatchResult:
        from src.gallery.data_models import IdentityStatus, MatchResult

        return MatchResult(
            candidates=[],
            status=IdentityStatus.STRANGER,
        )
