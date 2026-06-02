"""
REST API 路由 — 配置管理、底库查询、身份确认

提供 HTTP 端点供前端/外部系统调用:
- GET/PUT /api/config — 配置查询与更新
- GET /api/gallery/persons — 人物列表
- GET /api/gallery/person/{person_id} — 人物详情
- POST /api/vision/confirm_identity — 人工确认身份
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ConfirmIdentityRequest,
    FeatureEntryInfo,
    OutfitInfo,
    PersonDetailResponse,
    PersonListResponse,
    PersonSummary,
    TunableParam,
)

router = APIRouter(prefix="/api", tags=["api"])

# 全局引用 (由 server.py 在 startup 时注入)
_orchestrator: Any = None


def set_orchestrator(orchestrator: Any) -> None:
    """注入 VisionOrchestrator 引用 (由 server.py 调用)。"""
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator() -> Any:
    """获取编排器实例。"""
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="System not initialized",
        )
    return _orchestrator


# ==============================================================================
# Config endpoints
# ==============================================================================

@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """获取所有可调参数及其当前值、范围。"""
    orch = _get_orchestrator()
    tunable = orch.config.get_tunable_params()
    params = {
        key: TunableParam(**info) for key, info in tunable.items()
    }
    return ConfigResponse(params=params)


@router.put("/config", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    """更新可调参数。"""
    orch = _get_orchestrator()
    updated = orch.config.update_from_dict(request.updates)
    if updated:
        logger.info("Config updated via REST: {}", updated)
    return ConfigUpdateResponse(updated_keys=updated)


# ==============================================================================
# Gallery endpoints
# ==============================================================================

@router.get("/gallery/persons", response_model=PersonListResponse)
async def list_persons() -> PersonListResponse:
    """列出底库中所有人物。"""
    orch = _get_orchestrator()
    gallery = orch.gallery

    persons = []
    for pid, profile in gallery.items():
        persons.append(
            PersonSummary(
                person_id=profile.person_id,
                display_name=profile.display_name,
                face_count=profile.total_face_features(),
                outfit_count=len(profile.wardrobe),
                last_seen=profile.last_seen,
                total_appearances=profile.total_appearances,
            )
        )

    # 按最后出现时间倒序
    persons.sort(key=lambda p: p.last_seen, reverse=True)

    return PersonListResponse(persons=persons, total=len(persons))


@router.get("/gallery/person/{person_id}", response_model=PersonDetailResponse)
async def get_person(person_id: str) -> PersonDetailResponse:
    """获取单个人物的详细信息。"""
    orch = _get_orchestrator()
    gallery = orch.gallery

    if person_id not in gallery:
        raise HTTPException(status_code=404, detail="Person not found")

    profile = gallery[person_id]

    # 转换人脸特征
    face_features: dict[str, list[FeatureEntryInfo]] = {}
    for bucket, entries in profile.face_features.items():
        bucket_key = bucket.value if hasattr(bucket, "value") else str(bucket)
        face_features[bucket_key] = [
            FeatureEntryInfo(
                pose_bucket=bucket_key,
                quality_score=round(entry.quality_score, 3),
                timestamp=entry.timestamp,
            )
            for entry in entries
        ]

    # 转换衣橱
    wardrobe = [
        OutfitInfo(
            quality_score=round(outfit.quality_score, 3),
            first_seen=outfit.first_seen,
            last_seen=outfit.last_seen,
            seen_count=outfit.seen_count,
        )
        for outfit in profile.wardrobe
    ]

    return PersonDetailResponse(
        person_id=profile.person_id,
        display_name=profile.display_name,
        face_features=face_features,
        wardrobe=wardrobe,
        has_proportions=profile.body_proportions is not None,
        vlm_description=profile.vlm_description,
        created_at=profile.created_at,
        last_seen=profile.last_seen,
        total_appearances=profile.total_appearances,
    )


# ==============================================================================
# Vision control endpoints
# ==============================================================================

@router.post("/vision/confirm_identity")
async def confirm_identity(request: ConfirmIdentityRequest) -> dict:
    """
    人工确认身份 (Human-in-the-loop)。

    将指定 track_id 绑定到 person_id，更新底库和缓存。
    """
    orch = _get_orchestrator()

    try:
        await orch.confirm_identity(
            track_id=request.track_id,
            person_id=request.person_id,
            name=request.name,
        )
        return {
            "status": "confirmed",
            "track_id": request.track_id,
            "person_id": request.person_id,
            "name": request.name,
        }
    except Exception as e:
        logger.exception("Confirm identity failed")
        raise HTTPException(
            status_code=500,
            detail=f"Confirmation failed: {str(e)}",
        ) from e
