"""拉流录像查询 / 预览 / 下载 API

数据由 pipeline/video_recorder 在 consume 生命周期内写入:
- 元数据: session_store.video_recordings (本服务 db_url)
- 内容: COS videos/{device_sn}/{started_at}.mp4

查询按录像开始时间范围过滤; 预览/下载经 COS 临时链 307 跳转。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from voice_agent_common.utils.clock import parse_to_naive_cst, to_cst_iso

from session_store import VideoStore
from session_store.models import VideoRecordingORM
from src import deps

router = APIRouter(prefix="/api/videos", tags=["videos"])


class VideoItem(BaseModel):
    id: int
    device_sn: str
    stream_session_id: str
    started_at: str
    ended_at: str | None = None
    duration_ms: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    cos_key: str | None = None
    status: str
    error_message: str | None = None


class VideoListResponse(BaseModel):
    items: list[VideoItem]
    total_returned: int = Field(description="本页条数")


def _to_item(orm: VideoRecordingORM) -> VideoItem:
    return VideoItem(
        id=orm.id,
        device_sn=orm.device_sn,
        stream_session_id=orm.stream_session_id,
        started_at=to_cst_iso(orm.started_at) or "",
        ended_at=to_cst_iso(orm.ended_at),
        duration_ms=orm.duration_ms,
        width=orm.width,
        height=orm.height,
        fps=orm.fps,
        frame_count=orm.frame_count,
        cos_key=orm.cos_key,
        status=orm.status,
        error_message=orm.error_message,
    )


@router.get("/", response_model=VideoListResponse)
@router.get("", response_model=VideoListResponse, include_in_schema=False)
async def list_videos(
    start_from: str | None = Query(
        None, description="开始时间下界 (ISO, 含), 按录像 started_at 过滤"),
    start_to: str | None = Query(
        None, description="开始时间上界 (ISO, 含), 按录像 started_at 过滤"),
    device_sn: str | None = Query(None, description="设备号精确匹配"),
    status: str | None = Query(
        "ready",
        description="状态过滤: ready/recording/failed; 传空串不过滤",
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> VideoListResponse:
    """按录像开始时间范围查询视频信息列表(新→旧)。"""
    try:
        sf = parse_to_naive_cst(start_from) if start_from else None
        st = parse_to_naive_cst(start_to) if start_to else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"时间格式无效: {e}") from e

    status_filter: str | None
    if status is None or status == "":
        status_filter = None
    else:
        status_filter = status

    rows = await VideoStore.list_by_start_range(
        start_from=sf,
        start_to=st,
        device_sn=device_sn or None,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    items = [_to_item(r) for r in rows]
    return VideoListResponse(items=items, total_returned=len(items))


@router.get("/{video_id}", response_model=VideoItem)
async def get_video(video_id: int) -> VideoItem:
    orm = await VideoStore.get(video_id)
    if orm is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return _to_item(orm)


@router.get("/{video_id}/media")
async def get_video_media(
    video_id: int,
    download: bool = Query(False, description="是否作为附件下载"),
    filename: str | None = Query(None, description="下载时的文件名"),
):
    """获取录像临时访问链接并 307 跳转 (预览/下载)。"""
    orm = await VideoStore.get(video_id)
    if orm is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if not orm.cos_key or orm.status != "ready":
        raise HTTPException(status_code=404, detail="Video file not ready")
    if deps.cos_client is None:
        raise HTTPException(status_code=503, detail="COS client unavailable")
    try:
        name = filename or f"video_{orm.device_sn}_{orm.id}.mp4"
        url = await deps.cos_client.get_temporary_file_url(
            orm.cos_key, download=download, filename=name if download else None,
        )
        return RedirectResponse(url=url, status_code=307)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{video_id}")
async def delete_video(video_id: int) -> dict:
    """删除录像元数据及 COS 对象。"""
    snapshot = await VideoStore.delete(video_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Video not found")
    cos_key = snapshot.get("cos_key")
    if cos_key and deps.cos_client is not None:
        try:
            await deps.cos_client.delete_object(cos_key)
        except Exception as e:
            # 元数据已删; COS 残留只记日志, 不让用户重试卡死
            raise HTTPException(
                status_code=500,
                detail=f"元数据已删除, COS 删除失败: {e}",
            ) from e
    return {"success": True}
