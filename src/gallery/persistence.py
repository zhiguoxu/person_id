"""
Gallery Persistence — SQLModel ORM 持久化

使用 SQLModel + SQLAlchemy AsyncSession 异步地将底库 PersonProfile 存储到 SQLite。
特征向量以 numpy 二进制 blob 存储 (tobytes / frombuffer), 保证精度无损。

GalleryPersistence 通过 @cache 实现单例, 引擎在 FastAPI lifespan 中异步初始化。
"""
from __future__ import annotations

from functools import cache

from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlmodel import SQLModel, select

from src.gallery.converters import (
    entry_to_feature_row,
    outfit_to_row,
    rows_to_profile,
    sync_outfit_fields,
    sync_person_fields,
)
from src.gallery.data_models import (
    FeatureEntry,
    OutfitRecord,
    PersonProfile,
)
from src.gallery.db_models import (
    BodyFeatureRow,
    FaceFeatureRow,
    PersonRow,
    WardrobeRow,
)


@cache
def get_gallery_persistence() -> GalleryPersistence:
    """获取 GalleryPersistence 单例。"""
    return GalleryPersistence()


class GalleryPersistence:
    """SQLModel ORM 底库持久化管理器 (单例)。

    使用方式::

        # lifespan startup
        persistence = get_gallery_persistence()
        await persistence.initialize(db_path)

        # 业务代码
        persistence = get_gallery_persistence()
        await persistence.load_all_profiles(camera_id)

        # lifespan shutdown
        await get_gallery_persistence().close()
    """

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None

    async def initialize(self, db_path: str) -> None:
        """创建引擎并初始化表结构。在 FastAPI lifespan startup 中调用。"""
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        logger.info("GalleryPersistence initialized: db={}", db_path)

    async def close(self) -> None:
        """关闭引擎。在 FastAPI lifespan shutdown 中调用。"""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            logger.info("GalleryPersistence closed")

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError(
                "GalleryPersistence not initialized. "
                "Call await initialize(db_path) first."
            )
        return self._engine

    # 加载
    # ------------------------------------------------------------------

    async def load_all_profiles(
            self, camera_id: str,
    ) -> dict[str, PersonProfile]:
        """从数据库加载指定摄像头的全部人物档案。"""
        profiles: dict[str, PersonProfile] = {}

        try:
            async with AsyncSession(self.engine) as session:
                stmt = select(PersonRow).where(
                    PersonRow.camera_id == camera_id
                )
                results = await session.exec(stmt)
                person_rows = results.all()

                for person_row in person_rows:
                    face_rows = await self._load_face_features(
                        session, person_row.person_id,
                    )
                    body_rows = await self._load_body_features(
                        session, person_row.person_id,
                    )
                    wardrobe_rows = await self._load_wardrobe(
                        session, person_row.person_id,
                    )

                    profile = rows_to_profile(
                        person_row, face_rows, body_rows, wardrobe_rows,
                    )
                    profiles[profile.person_id] = profile

            logger.info(
                "Loaded {} profiles (camera={})",
                len(profiles), camera_id,
            )

        except Exception:
            logger.exception("Failed to load profiles from database")
            raise

        return profiles

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    async def delete_profile(
            self, person_id: str, camera_id: str,
    ) -> None:
        """从数据库删除指定人物档案 (级联删除关联的特征和衣橱)。"""
        try:
            async with AsyncSession(self.engine) as session:
                existing = await self._get_person_row(
                    session, person_id, camera_id,
                )
                if existing:
                    await session.delete(existing)
                    await session.commit()
                    logger.info(
                        "Deleted profile: {} (camera={})",
                        person_id, camera_id,
                    )
                else:
                    logger.warning(
                        "Profile not found for deletion: {} (camera={})",
                        person_id, camera_id,
                    )
        except Exception:
            logger.exception("Failed to delete profile: {}", person_id)
            raise

    # ------------------------------------------------------------------
    # 增量更新 (细粒度)
    # ------------------------------------------------------------------

    async def upsert_person_row(
            self, profile: PersonProfile, camera_id: str,
    ) -> None:
        """仅更新 persons 表的元数据 (不触碰特征子表)。

        自动设置 last_updated 为当前时间, update_count 自增。
        """
        import time
        now = time.time()
        profile.last_updated = now
        profile.update_count += 1

        async with AsyncSession(self.engine) as session:
            existing = await self._get_person_row(
                session, profile.person_id, camera_id,
            )
            if existing:
                sync_person_fields(existing, profile)
            else:
                existing = PersonRow(
                    person_id=profile.person_id,
                    camera_id=camera_id,
                )
                sync_person_fields(existing, profile)
            session.add(existing)
            await session.commit()

    async def add_feature(
            self,
            person_id: str,
            entry: FeatureEntry,
            kind: str,
    ) -> None:
        """INSERT 单条特征行。

        Args:
            person_id: 人物 ID。
            entry: 新增的特征条目。
            kind: "face" 或 "body"。
        """
        row_cls = FaceFeatureRow if kind == "face" else BodyFeatureRow
        async with AsyncSession(self.engine) as session:
            session.add(entry_to_feature_row(person_id, entry, row_cls))
            await session.commit()

    async def replace_feature(
            self,
            person_id: str,
            new_entry: FeatureEntry,
            evicted: FeatureEntry,
            kind: str,
    ) -> None:
        """原子替换: DELETE 旧行 + INSERT 新行 (单事务)。

        Args:
            person_id: 人物 ID。
            new_entry: 新入库的特征条目。
            evicted: 被淘汰的旧条目。
            kind: "face" 或 "body"。
        """
        row_cls = FaceFeatureRow if kind == "face" else BodyFeatureRow
        async with AsyncSession(self.engine) as session:
            # DELETE 旧行
            stmt = select(row_cls).where(
                row_cls.person_id == person_id,
                row_cls.pose_bucket == evicted.pose_bucket.value,
                row_cls.timestamp == evicted.timestamp,
            )
            result = await session.exec(stmt)
            old_row = result.first()
            if old_row:
                await session.delete(old_row)

            # INSERT 新行
            session.add(entry_to_feature_row(person_id, new_entry, row_cls))
            await session.commit()

    async def add_outfit(
            self, person_id: str, outfit: OutfitRecord,
    ) -> None:
        """INSERT 单条衣橱记录 (衣橱未满时新增)。"""
        async with AsyncSession(self.engine) as session:
            session.add(outfit_to_row(person_id, outfit))
            await session.commit()

    async def update_outfit(
            self, person_id: str, old: OutfitRecord, updated: OutfitRecord,
    ) -> None:
        """UPDATE 已有衣橱记录 (EMA 更新后)。按 person_id + first_seen 定位。"""
        async with AsyncSession(self.engine) as session:
            stmt = select(WardrobeRow).where(
                WardrobeRow.person_id == person_id,
                WardrobeRow.first_seen == old.first_seen,
            )
            result = await session.exec(stmt)
            row = result.first()
            if row:
                sync_outfit_fields(row, updated)
                session.add(row)
                await session.commit()

    async def replace_outfit(
            self, person_id: str, new_outfit: OutfitRecord, evicted: OutfitRecord,
    ) -> None:
        """原子替换: DELETE 旧衣橱记录 + INSERT 新记录 (单事务)。"""
        async with AsyncSession(self.engine) as session:
            # DELETE 旧行
            stmt = select(WardrobeRow).where(
                WardrobeRow.person_id == person_id,
                WardrobeRow.first_seen == evicted.first_seen,
            )
            result = await session.exec(stmt)
            old_row = result.first()
            if old_row:
                await session.delete(old_row)

            # INSERT 新行
            session.add(outfit_to_row(person_id, new_outfit))
            await session.commit()

    # ------------------------------------------------------------------
    # 内部: 查询辅助
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_person_row(
            session: AsyncSession, person_id: str, camera_id: str,
    ) -> PersonRow | None:
        stmt = select(PersonRow).where(
            PersonRow.person_id == person_id,
            PersonRow.camera_id == camera_id,
        )
        result = await session.exec(stmt)
        return result.first()

    @staticmethod
    async def _load_face_features(
            session: AsyncSession, person_id: str,
    ) -> list[FaceFeatureRow]:
        stmt = select(FaceFeatureRow).where(
            FaceFeatureRow.person_id == person_id
        )
        result = await session.exec(stmt)
        return list(result.all())

    @staticmethod
    async def _load_body_features(
            session: AsyncSession, person_id: str,
    ) -> list[BodyFeatureRow]:
        stmt = select(BodyFeatureRow).where(
            BodyFeatureRow.person_id == person_id
        )
        result = await session.exec(stmt)
        return list(result.all())

    @staticmethod
    async def _load_wardrobe(
            session: AsyncSession, person_id: str,
    ) -> list[WardrobeRow]:
        stmt = select(WardrobeRow).where(
            WardrobeRow.person_id == person_id
        )
        result = await session.exec(stmt)
        return list(result.all())
