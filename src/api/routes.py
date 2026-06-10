"""
REST API 路由 — 配置管理、底库查询、身份确认

多摄像头架构:
- GET/PUT /api/config — 全局配置（不分摄像头）
- GET /api/{camera_id}/gallery/persons — 指定摄像头的人物列表
- GET /api/{camera_id}/gallery/person/{person_id} — 人物详情
- POST /api/{camera_id}/vision/confirm_identity — 人工确认身份
"""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from loguru import logger
from src.api.registry import get_camera_orchestrator
from src.config import get_config as _get_config

from src.api.schemas import (
    CachedFrameInfo,
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ConfirmIdentityRequest,
    FeatureEntryInfo,
    OutfitInfo,
    PersonDetailResponse,
    PersonListResponse,
    PersonSummary,
    QualityCacheResponse,
    RenamePersonRequest,
    TunableParam,
)
from src.pipeline.frame_buffer import CachedFrame

router = APIRouter(prefix="/api", tags=["api"])


def _get_camera_orchestrator(camera_id: str):
    """获取指定摄像头的编排器。"""
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
    from src.api.registry import camera_registry
    return {"cameras": list(camera_registry.keys())}


# ==============================================================================
# Gallery endpoints (per-camera)
# ==============================================================================

@router.get("/{camera_id}/gallery/persons", response_model=PersonListResponse)
async def list_persons(camera_id: str) -> PersonListResponse:
    """列出指定摄像头底库中所有人物。

    优先从活跃的 orchestrator 内存读取, 不在线时从数据库加载。
    """
    # 尝试从活跃 orchestrator 获取 (最快)
    if (orch := get_camera_orchestrator(camera_id)) is not None:
        gallery = orch.gallery
    else:
        # Camera 不在线, 直接从数据库加载
        from src.gallery.persistence import get_gallery_persistence
        try:
            gallery = await get_gallery_persistence().load_all_profiles(camera_id)
        except Exception as e:
            logger.warning("Failed to load gallery from DB for camera {}: {}", camera_id, e)
            gallery = {}

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
                source_image_b64=base64.b64encode(entry.source_image).decode('ascii') if entry.source_image else None,
                face_bbox=entry.face_bbox,
            )
            for entry in entries
        ]

    # 转换体态特征
    body_features: dict[str, list[FeatureEntryInfo]] = {}
    for bucket, entries in profile.body_features.items():
        bucket_key = bucket.value if hasattr(bucket, 'value') else str(bucket)
        body_features[bucket_key] = [
            FeatureEntryInfo(
                pose_bucket=bucket_key,
                quality_score=round(entry.quality_score, 3),
                timestamp=entry.timestamp,
                source_image_b64=base64.b64encode(entry.source_image).decode('ascii') if entry.source_image else None,
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
        body_features=body_features,
        wardrobe=wardrobe,
        body_proportions=profile.body_proportions,
        vlm_description=profile.vlm_description,
        created_at=profile.created_at,
        last_updated=profile.last_updated,
        update_count=profile.update_count,
    )


@router.patch("/{camera_id}/gallery/person/{person_id}")
async def rename_person(
        camera_id: str, person_id: str, request: RenamePersonRequest,
) -> dict:
    """
    重命名人物 display_name。

    同时更新内存 gallery 和数据库。
    """
    from src.gallery.persistence import get_gallery_persistence

    new_name = request.display_name.strip()

    # 更新内存中的 profile
    orch = get_camera_orchestrator(camera_id)
    if orch is not None:
        gallery = orch.gallery
        if person_id not in gallery:
            raise HTTPException(status_code=404, detail="Person not found")
        profile = gallery[person_id]
        profile.display_name = new_name
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found or not connected",
        )

    # 持久化到数据库
    try:
        persistence = get_gallery_persistence()
        await persistence.upsert_person_row(profile, camera_id)
    except Exception as e:
        logger.exception("Failed to persist rename for {}", person_id)
        raise HTTPException(
            status_code=500,
            detail=f"Rename failed: {str(e)}",
        ) from e

    logger.info("Renamed person {} to '{}' (camera={})", person_id, new_name, camera_id)
    return {
        "status": "renamed",
        "camera_id": camera_id,
        "person_id": person_id,
        "display_name": new_name,
    }


# ==============================================================================
# Track Quality Cache (per-camera)
# ==============================================================================

@router.get("/{camera_id}/track/{track_id}/quality_cache", response_model=QualityCacheResponse)
async def get_quality_cache(camera_id: str, track_id: int) -> QualityCacheResponse:
    """
    获取指定 track 的质量缓存 (face/body pool 的图片和元数据)。
    """
    import cv2

    orch = _get_camera_orchestrator(camera_id)
    state = orch.tracks.get(track_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    cache = state.quality_cache

    def _convert_pool(pool: list[CachedFrame]) -> list[CachedFrameInfo]:
        items = []
        for cf in pool:
            try:
                _, buf = cv2.imencode('.jpg', cf.entry.crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            except Exception:
                continue
            items.append(CachedFrameInfo(
                image_b64=img_b64,
                quality=round(cf.quality, 3),
                timestamp=cf.entry.timestamp,
                pose_bucket=cf.entry.pose_bucket.value,
                enrolled=cf.enrolled,
            ))
        return items

    return QualityCacheResponse(
        track_id=track_id,
        face_pool=_convert_pool(cache.face_pool),
        body_pool=_convert_pool(cache.body_pool),
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
    except ValueError as e:
        logger.warning(f"Bad request in confirm_identity: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Confirm identity failed")
        raise HTTPException(
            status_code=500,
            detail=f"Confirmation failed: {str(e)}",
        ) from e


@router.delete("/{camera_id}/gallery/person/{person_id}")
async def delete_person(camera_id: str, person_id: str) -> dict:
    """
    删除底库中的人物。

    同时从内存 gallery 和数据库中移除。
    """
    from src.gallery.persistence import get_gallery_persistence

    # 从活跃 orchestrator 内存中移除 (同步: 标记 + 清 gallery + reset tracks)
    orch = get_camera_orchestrator(camera_id)
    if orch is not None:
        orch.delete_person(person_id)
    else:
        from src.api.registry import camera_registry
        logger.warning(
            "DELETE person={} camera={}: orch=NOT_FOUND, registry_keys={}",
            person_id, camera_id, list(camera_registry.keys()),
        )

    # 从数据库中移除 (orch 在线时走 save_lock 防止与入库 commit 交叉)
    try:
        if orch is not None:
            await orch.delete_person_from_db(person_id)
        else:
            persistence = get_gallery_persistence()
            await persistence.delete_profile(person_id, camera_id)
    except Exception as e:
        logger.exception("Failed to delete person {} from DB", person_id)
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(e)}",
        ) from e

    logger.info("Deleted person {} (camera={})", person_id, camera_id)
    return {
        "status": "deleted",
        "camera_id": camera_id,
        "person_id": person_id,
    }
