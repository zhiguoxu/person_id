"""
REST API 路由 — 配置管理、底库查询、身份确认

多摄像头架构:
- GET/PUT /api/config — 全局配置（不分摄像头）
- GET /api/{camera_id}/gallery/persons — 指定摄像头的人物列表
- GET /api/{camera_id}/gallery/person/{person_id} — 人物详情
- POST /api/{camera_id}/vision/confirm_identity — 人工确认身份
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.config import get_config as _get_config

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


def _get_camera_orchestrator(camera_id: str):
    """获取指定摄像头的编排器。"""
    from src.api.server import get_camera_orchestrator
    orch = get_camera_orchestrator(camera_id)
    if orch is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found or not connected",
        )
    return orch


# ==============================================================================
# Config endpoints (全局, 不分摄像头)
# ==============================================================================

@router.get("/config", response_model=ConfigResponse)
async def get_config_endpoint() -> ConfigResponse:
    """获取所有可调参数及其当前值、范围。"""
    tunable = _get_config().get_tunable_params()
    params = {
        key: TunableParam(**info) for key, info in tunable.items()
    }
    return ConfigResponse(params=params)


@router.put("/config", response_model=ConfigUpdateResponse)
async def update_config_endpoint(request: ConfigUpdateRequest) -> ConfigUpdateResponse:
    """更新可调参数。"""
    updated = _get_config().update_from_dict(request.updates)
    if updated:
        logger.info("Config updated via REST: {}", updated)
    return ConfigUpdateResponse(updated_keys=updated)


# ==============================================================================
# Camera list
# ==============================================================================

@router.get("/cameras")
async def list_cameras() -> dict[str, list[str]]:
    """列出当前活跃的摄像头。"""
    from src.api.server import camera_registry
    return {"cameras": list(camera_registry.keys())}


# ==============================================================================
# Gallery endpoints (per-camera)
# ==============================================================================

@router.get("/{camera_id}/gallery/persons", response_model=PersonListResponse)
async def list_persons(camera_id: str) -> PersonListResponse:
    """列出指定摄像头底库中所有人物。"""
    orch = _get_camera_orchestrator(camera_id)
    gallery = orch.gallery

    persons = []
    for pid, profile in gallery.items():
        persons.append(
            PersonSummary(
                person_id=profile.person_id,
                display_name=profile.display_name,
                face_count=profile.total_face_features(),
                outfit_count=len(profile.wardrobe),
                last_updated=profile.last_updated,
                update_count=profile.update_count,
            )
        )

    persons.sort(key=lambda p: p.last_updated, reverse=True)
    return PersonListResponse(persons=persons, total=len(persons))


@router.get("/{camera_id}/gallery/person/{person_id}", response_model=PersonDetailResponse)
async def get_person(camera_id: str, person_id: str) -> PersonDetailResponse:
    """获取指定摄像头中单个人物的详细信息。"""
    orch = _get_camera_orchestrator(camera_id)
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
        last_updated=profile.last_updated,
        update_count=profile.update_count,
    )


# ==============================================================================
# Vision control endpoints (per-camera)
# ==============================================================================

@router.post("/{camera_id}/vision/confirm_identity")
async def confirm_identity(
    camera_id: str, request: ConfirmIdentityRequest
) -> dict:
    """
    人工确认身份 (Human-in-the-loop)。

    将指定 track_id 绑定到 person_id，更新底库和缓存。
    """
    orch = _get_camera_orchestrator(camera_id)

    try:
        await orch.confirm_identity(
            track_id=request.track_id,
            person_id=request.person_id,
            name=request.name,
        )
        return {
            "status": "confirmed",
            "camera_id": camera_id,
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
