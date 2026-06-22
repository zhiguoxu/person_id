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

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from loguru import logger
from src.api.registry import get_camera_orchestrator
from src.config import get_config as _get_config

from src.api.schemas import (
    BodyQualityTestResponse,
    BodySimilarityBodyInfo,
    BodySimilarityTestResponse,
    CachedFrameInfo,
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
    ConfirmIdentityRequest,
    FaceSimilarityFaceInfo,
    FaceSimilarityTestResponse,
    FeatureEntryInfo,
    OutfitInfo,
    PersonDetailResponse,
    PersonListResponse,
    PersonSummary,
    QualityCacheResponse,
    ReIDCompareResponse,
    RenamePersonRequest,
    TunableParam,
)
from src.pipeline.frame_buffer import CachedFrame
from src.tier1.attention import select_best_detection
from src.tier1.detection import get_fast_detector
from src.tier1.face_detector_light import get_face_detector_light
from src.tier2.features import get_face_extractor, get_ediffiqa
from src.pipeline.quality_utils import compute_quality_hint, compute_sharpness
import cv2
import numpy as np

router = APIRouter(prefix="/api", tags=["api"])


# ==============================================================================
# ISS Stream refresh (stop + start → FLV URL)
# ==============================================================================

@router.post("/refresh_stream")
async def refresh_stream() -> dict:
    """重启 ISS 直播流并返回 FLV URL。

    1. POST /iss/stop_stream
    2. POST /iss/start_stream
    3. 返回 data.Flv
    """
    import httpx

    cfg = _get_config().server
    base = cfg.iss_api_url.rstrip("/")
    headers = {"device-sn": cfg.iss_device_sn}

    async with httpx.AsyncClient(timeout=15.0) as client:
        # stop
        try:
            await client.post(f"{base}/iss/stop_stream", headers=headers)
        except Exception as e:
            logger.warning("ISS stop_stream failed (ignored): {}", e)

        # start
        resp = await client.post(f"{base}/iss/start_stream", headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ISS start_stream failed: {resp.status_code}")

        data = resp.json()
        if data.get("code") != 0:
            raise HTTPException(status_code=502, detail=f"ISS error: {data.get('msg', 'unknown')}")

        flv_url = data.get("data", {}).get("Flv", "")
        if not flv_url:
            raise HTTPException(status_code=502, detail="ISS returned no FLV URL")

    logger.info("ISS stream refreshed: {}", flv_url)
    return {"flv_url": flv_url}


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
    config = _get_config()
    tunable = config.get_tunable_params()
    params = {
        key: TunableParam(**info) for key, info in tunable.items()
    }
    flags = {
        "AGG_MIN_FACE_QUALITY": config.multiframe.agg_min_face_quality,
        "AGG_MIN_BODY_QUALITY": config.multiframe.agg_min_body_quality,
        "IMAGE_CORRECTION_ENABLED": config.server.image_correction_enabled,
    }
    return ConfigResponse(params=params, flags=flags)


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


# ==============================================================================
# Debug / Testing API
# ==============================================================================

@router.post("/test_body_quality", response_model=BodyQualityTestResponse)
async def test_body_quality(file: UploadFile = File(...)) -> BodyQualityTestResponse:
    """测试单张图片的 body quality 计算细节 (包含 hint 和 sharpness)。"""
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return BodyQualityTestResponse(has_person=False, error="Failed to decode image")

        detector = get_fast_detector()
        detections = detector.detect(frame)

        if not detections:
            return BodyQualityTestResponse(has_person=False)

        # 注意力选人: 1 人直接取, 多人按注意力评分选
        best_idx = select_best_detection(detections, frame.shape)
        det = detections[best_idx]
        if det.bbox is None:
            return BodyQualityTestResponse(has_person=False)

        h_f, w_f = frame.shape[:2]

        # 1. 计算 quality_hint
        q_hint = compute_quality_hint(det.bbox, det.keypoints, (h_f, w_f))

        # 2. 提取 crop
        x1 = max(0, int(det.bbox[0]))
        y1 = max(0, int(det.bbox[1]))
        x2 = min(w_f, int(det.bbox[2]))
        y2 = min(h_f, int(det.bbox[3]))
        crop = frame[y1:y2, x1:x2].copy()

        # 3. 计算 sharpness
        sharpness = compute_sharpness(crop) if crop.size > 0 else 0.0

        # 4. 计算最终 quality (和 tier2/batch_extractor.py 逻辑一致)
        final_quality = 0.75 * q_hint + 0.25 * sharpness

        return BodyQualityTestResponse(
            has_person=True,
            quality=final_quality,
            quality_hint=q_hint,
            sharpness=sharpness,
            bbox=det.bbox.tolist()
        )
    except Exception as e:
        logger.exception("Error testing body quality")
        return BodyQualityTestResponse(has_person=False, error=str(e))


@router.post("/test_face_similarity", response_model=FaceSimilarityTestResponse)
async def test_face_similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    undistort: str = Form("false"),
) -> FaceSimilarityTestResponse:
    """测试两张图片的人脸相似度 (使用与底库匹配相同的 ArcFace/AdaFace + cosine similarity)。"""
    do_undistort = undistort.lower() in ("true", "1", "yes")
    try:
        detector = get_fast_detector()
        face_det = get_face_detector_light()
        face_ext = get_face_extractor()
        ediffiqa = get_ediffiqa()

        def _process_image(image_bytes: bytes) -> tuple[FaceSimilarityFaceInfo, np.ndarray | None, str | None]:
            """处理单张图片: 检测人体 → 检测人脸 → 对齐 → 提取嵌入。"""
            corrected_b64 = None
            # 镜头畸变矫正
            if do_undistort:
                from src.utils.image_correction import correct_image_bytes
                try:
                    image_bytes = correct_image_bytes(image_bytes)
                except Exception:
                    logger.warning("Image undistortion failed, using original")

            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return FaceSimilarityFaceInfo(has_face=False), None, None

            # 矫正后的原图编码供前端预览
            if do_undistort:
                _, cb = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                corrected_b64 = base64.b64encode(cb.tobytes()).decode('ascii')

            h_f, w_f = frame.shape[:2]

            # Step 1: 人体检测 (YOLO)
            detections = detector.detect(frame)
            if not detections or detections[0].bbox is None:
                # 没检测到人体, 直接在全图检测人脸
                face_result = face_det.get_aligned_face(frame)
                if face_result is None:
                    return FaceSimilarityFaceInfo(has_face=False), None, corrected_b64
                aligned, face_bbox_raw, kps = face_result
                face_bbox = [float(face_bbox_raw[0]), float(face_bbox_raw[1]),
                             float(face_bbox_raw[2]), float(face_bbox_raw[3])]
                quality = ediffiqa.predict(aligned)
                _, buf = cv2.imencode('.jpg', aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                aligned_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                embedding = face_ext.extract_embedding(aligned)
                return FaceSimilarityFaceInfo(
                    has_face=True,
                    person_bbox=None,
                    face_bbox=face_bbox,
                    face_quality=round(quality, 4),
                    aligned_face_b64=aligned_b64,
                ), embedding, corrected_b64

            # 注意力选人: 1 人直接取, 多人按注意力评分选
            best_idx = select_best_detection(detections, frame.shape)
            det = detections[best_idx]
            x1 = max(0, int(det.bbox[0]))
            y1 = max(0, int(det.bbox[1]))
            x2 = min(w_f, int(det.bbox[2]))
            y2 = min(h_f, int(det.bbox[3]))
            person_bbox = [float(x1), float(y1), float(x2), float(y2)]
            crop = frame[y1:y2, x1:x2].copy()

            # Step 2: 在人体裁剪上检测人脸 (SCRFD)
            face_result = face_det.get_aligned_face(crop)
            if face_result is None:
                return FaceSimilarityFaceInfo(has_face=False, person_bbox=person_bbox), None, corrected_b64

            aligned, face_bbox_raw, kps = face_result
            # 人脸框从 crop 坐标转到原图坐标
            face_bbox = [
                float(face_bbox_raw[0]) + x1,
                float(face_bbox_raw[1]) + y1,
                float(face_bbox_raw[2]) + x1,
                float(face_bbox_raw[3]) + y1,
            ]

            # Step 3: 人脸质量 (eDifFIQA)
            quality = ediffiqa.predict(aligned)

            # 编码对齐人脸供前端显示
            _, buf = cv2.imencode('.jpg', aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            aligned_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            # Step 4: 人脸嵌入 (ArcFace/AdaFace)
            embedding = face_ext.extract_embedding(aligned)

            return FaceSimilarityFaceInfo(
                has_face=True,
                person_bbox=person_bbox,
                face_bbox=face_bbox,
                face_quality=round(quality, 4),
                aligned_face_b64=aligned_b64,
            ), embedding, corrected_b64

        # 处理两张图片
        bytes1 = await file1.read()
        bytes2 = await file2.read()

        info1, emb1, corr1 = _process_image(bytes1)
        info2, emb2, corr2 = _process_image(bytes2)

        # 计算相似度 (与 gallery matcher 一致的 cosine similarity)
        similarity = None
        if emb1 is not None and emb2 is not None:
            similarity = round(float(np.dot(emb1, emb2)), 4)

        return FaceSimilarityTestResponse(
            face1=info1,
            face2=info2,
            similarity=similarity,
            corrected_image1_b64=corr1,
            corrected_image2_b64=corr2,
        )

    except Exception as e:
        logger.exception("Error testing face similarity")
        return FaceSimilarityTestResponse(
            face1=FaceSimilarityFaceInfo(has_face=False),
            face2=FaceSimilarityFaceInfo(has_face=False),
            error=str(e),
        )


@router.post("/test_body_similarity", response_model=BodySimilarityTestResponse)
async def test_body_similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    undistort: str = Form("false"),
) -> BodySimilarityTestResponse:
    """测试两张图片的全身 ReID 相似度 (SOLIDER Swin-Small + cosine similarity)。"""
    do_undistort = undistort.lower() in ("true", "1", "yes")
    try:
        from src.tier2.features import get_body_extractor
        detector = get_fast_detector()
        body_ext = get_body_extractor()

        def _process_image(image_bytes: bytes) -> tuple[BodySimilarityBodyInfo, np.ndarray | None, str | None]:
            """处理单张图片: 检测人体 → 裁剪 → 提取 body embedding。"""
            corrected_b64 = None
            # 镜头畸变矫正
            if do_undistort:
                from src.utils.image_correction import correct_image_bytes
                try:
                    image_bytes = correct_image_bytes(image_bytes)
                except Exception:
                    logger.warning("Image undistortion failed, using original")

            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return BodySimilarityBodyInfo(has_body=False), None, None

            # 矫正后的原图编码供前端预览
            if do_undistort:
                _, cb = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                corrected_b64 = base64.b64encode(cb.tobytes()).decode('ascii')

            h_f, w_f = frame.shape[:2]

            # 人体检测 (YOLO)
            detections = detector.detect(frame)
            if not detections or detections[0].bbox is None:
                # 未检测到人体, 以全图作为 body crop
                crops = [frame]
                person_bbox = [0.0, 0.0, float(w_f), float(h_f)]
            else:
                best_idx = select_best_detection(detections, frame.shape)
                det = detections[best_idx]
                x1 = max(0, int(det.bbox[0]))
                y1 = max(0, int(det.bbox[1]))
                x2 = min(w_f, int(det.bbox[2]))
                y2 = min(h_f, int(det.bbox[3]))
                person_bbox = [float(x1), float(y1), float(x2), float(y2)]
                crops = [frame[y1:y2, x1:x2].copy()]

            # 提取 body embedding
            embeddings = body_ext.extract_batch(crops)
            if not embeddings:
                return BodySimilarityBodyInfo(has_body=False), None, corrected_b64

            # 编码裁剪图供前端显示
            crop_resized = cv2.resize(crops[0], (128, 384))
            _, buf = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            return BodySimilarityBodyInfo(
                has_body=True,
                person_bbox=person_bbox,
                body_crop_b64=crop_b64,
            ), embeddings[0], corrected_b64

        # 处理两张图片
        bytes1 = await file1.read()
        bytes2 = await file2.read()

        info1, emb1, corr1 = _process_image(bytes1)
        info2, emb2, corr2 = _process_image(bytes2)

        # 计算相似度 (cosine similarity, embeddings 已 L2 归一化)
        similarity = None
        if emb1 is not None and emb2 is not None:
            similarity = round(float(np.dot(emb1, emb2)), 4)

        return BodySimilarityTestResponse(
            body1=info1,
            body2=info2,
            similarity=similarity,
            embedding_dim=body_ext.EMBEDDING_DIM,
            corrected_image1_b64=corr1,
            corrected_image2_b64=corr2,
        )

    except Exception as e:
        logger.exception("Error testing body similarity")
        return BodySimilarityTestResponse(
            body1=BodySimilarityBodyInfo(has_body=False),
            body2=BodySimilarityBodyInfo(has_body=False),
            error=str(e),
        )


# ── ReID 模型对比 ─────────────────────────────────────────────
_osnet_extractor = None


def _get_osnet_extractor():
    """延迟加载 OSNet 提取器 (singleton)。"""
    global _osnet_extractor
    if _osnet_extractor is None:
        from src.tier2.features.osnet_extractor import OSNetExtractor
        hw_config = _get_config().hardware
        _osnet_extractor = OSNetExtractor(device=hw_config.device)
    return _osnet_extractor


@router.post("/test_reid_compare", response_model=ReIDCompareResponse)
async def test_reid_compare(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    undistort: str = Form("false"),
) -> ReIDCompareResponse:
    """对比两种 ReID 模型 (SOLIDER Swin-Small vs OSNet-AIN) 的相似度。"""
    do_undistort = undistort.lower() in ("true", "1", "yes")
    try:
        from src.tier2.features import get_body_extractor
        detector = get_fast_detector()
        solider_ext = get_body_extractor()
        osnet_ext = _get_osnet_extractor()

        def _process_image(image_bytes: bytes):
            corrected_b64 = None
            # 镜头畸变矫正
            if do_undistort:
                from src.utils.image_correction import correct_image_bytes
                try:
                    image_bytes = correct_image_bytes(image_bytes)
                except Exception:
                    logger.warning("Image undistortion failed, using original")

            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return BodySimilarityBodyInfo(has_body=False), None, None

            # 矫正后的原图编码供前端预览
            if do_undistort:
                _, cb = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                corrected_b64 = base64.b64encode(cb.tobytes()).decode('ascii')

            h_f, w_f = frame.shape[:2]
            detections = detector.detect(frame)
            if not detections or detections[0].bbox is None:
                crops = [frame]
                person_bbox = [0.0, 0.0, float(w_f), float(h_f)]
            else:
                best_idx = select_best_detection(detections, frame.shape)
                det = detections[best_idx]
                x1 = max(0, int(det.bbox[0]))
                y1 = max(0, int(det.bbox[1]))
                x2 = min(w_f, int(det.bbox[2]))
                y2 = min(h_f, int(det.bbox[3]))
                person_bbox = [float(x1), float(y1), float(x2), float(y2)]
                crops = [frame[y1:y2, x1:x2].copy()]

            crop_resized = cv2.resize(crops[0], (128, 384))
            _, buf = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_b64 = base64.b64encode(buf.tobytes()).decode('ascii')

            return BodySimilarityBodyInfo(
                has_body=True,
                person_bbox=person_bbox,
                body_crop_b64=crop_b64,
            ), crops[0], corrected_b64

        bytes1 = await file1.read()
        bytes2 = await file2.read()
        info1, crop1, corr1 = _process_image(bytes1)
        info2, crop2, corr2 = _process_image(bytes2)

        solider_sim = None
        osnet_sim = None

        if crop1 is not None and crop2 is not None:
            # SOLIDER
            s_embs = solider_ext.extract_batch([crop1, crop2])
            if len(s_embs) == 2:
                solider_sim = round(float(np.dot(s_embs[0], s_embs[1])), 4)

            # OSNet
            o_embs = osnet_ext.extract_batch([crop1, crop2])
            if len(o_embs) == 2:
                osnet_sim = round(float(np.dot(o_embs[0], o_embs[1])), 4)

        return ReIDCompareResponse(
            body1=info1,
            body2=info2,
            solider_similarity=solider_sim,
            solider_dim=solider_ext.EMBEDDING_DIM,
            osnet_similarity=osnet_sim,
            osnet_dim=osnet_ext.EMBEDDING_DIM,
            corrected_image1_b64=corr1,
            corrected_image2_b64=corr2,
        )

    except Exception as e:
        logger.exception("Error comparing ReID models")
        return ReIDCompareResponse(
            body1=BodySimilarityBodyInfo(has_body=False),
            body2=BodySimilarityBodyInfo(has_body=False),
            error=str(e),
        )


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
                overlay_bbox=entry.overlay_bbox,
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
                overlay_bbox=entry.overlay_bbox,
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
                pose_bucket=cf.entry.detection.pose_bucket.value,
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

@router.delete("/{camera_id}/track/{track_id}/quality_cache")
async def clear_quality_cache(camera_id: str, track_id: int) -> dict:
    """清空指定 track 的 quality cache, 使新数据能重新进入。"""
    orch = _get_camera_orchestrator(camera_id)
    state = orch.tracks.get(track_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")

    state.quality_cache.clear()
    state.force_probe = True
    logger.info("Quality cache cleared for track_id={} (camera={})", track_id, camera_id)
    return {"status": "cleared", "track_id": track_id}


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
