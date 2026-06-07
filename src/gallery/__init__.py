from __future__ import annotations

from src.gallery.persistence import GalleryPersistence
from src.gallery.data_models import PersonProfile


async def load_gallery(
    db_path: str, camera_id: str,
) -> dict[str, PersonProfile]:
    """从 SQLite 加载指定摄像头的底库。"""
    persistence = GalleryPersistence(db_path, camera_id)
    await persistence.initialize()
    gallery = await persistence.load_all_profiles()
    await persistence.close()
    return gallery


async def save_gallery(
    db_path: str,
    gallery: dict[str, PersonProfile],
    camera_id: str,
) -> None:
    """保存底库到 SQLite（按 camera_id 隔离）。"""
    persistence = GalleryPersistence(db_path, camera_id)
    await persistence.initialize()
    for profile in gallery.values():
        await persistence.save_profile(profile)
    await persistence.close()


__all__ = [
    "GalleryPersistence",
    "load_gallery",
    "save_gallery",
]
