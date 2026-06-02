"""Gallery package — 底库管理、匹配、重排序与持久化。"""
from src.gallery.matcher import GalleryMatcher
from src.gallery.updater import GalleryUpdater
from src.gallery.reranker import KReciprocalReranker
from src.gallery.persistence import GalleryPersistence


def create_gallery_matcher(config):
    """创建底库匹配器。"""
    return GalleryMatcher(config)


def create_gallery_updater(config):
    """创建底库更新器。"""
    return GalleryUpdater(config)


async def load_gallery(db_path):
    """从 SQLite 加载底库。"""
    persistence = GalleryPersistence(db_path)
    await persistence.initialize()
    gallery = await persistence.load_all_profiles()
    await persistence.close()
    return gallery


async def save_gallery(db_path, gallery):
    """保存底库到 SQLite。"""
    persistence = GalleryPersistence(db_path)
    await persistence.initialize()
    for profile in gallery.values():
        await persistence.save_profile(profile)
    await persistence.close()


__all__ = [
    "GalleryMatcher",
    "GalleryUpdater",
    "KReciprocalReranker",
    "GalleryPersistence",
    "create_gallery_matcher",
    "create_gallery_updater",
    "load_gallery",
    "save_gallery",
]
