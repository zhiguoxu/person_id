"""
Gallery Persistence — SQLModel ORM 持久化

使用 SQLModel + SQLAlchemy AsyncSession 异步地将底库 PersonProfile 存储到 SQLite。
特征向量以 numpy 二进制 blob 存储 (tobytes / frombuffer), 保证精度无损。

GalleryPersistence 通过 @cache 实现单例, 引擎在 FastAPI lifespan 中异步初始化。
"""
from __future__ import annotations

from functools import cache

import numpy as np
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlmodel import SQLModel, select

from src.gallery.data_models import (
    BodyProportions,
    FeatureEntry,
    OutfitRecord,
    PersonProfile,
    PoseBucket,
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

    # ------------------------------------------------------------------
    # 保存 (单个)
    # ------------------------------------------------------------------

    async def save_profile(
        self, profile: PersonProfile, camera_id: str,
    ) -> None:
        """将 PersonProfile 完整写入数据库 (INSERT OR REPLACE)。"""
        try:
            async with AsyncSession(self.engine) as session:
                existing = await self._get_person_row(
                    session, profile.person_id, camera_id,
                )
                if existing:
                    await session.delete(existing)
                    await session.flush()

                row = self._profile_to_row(profile, camera_id)
                session.add(row)
                await session.commit()

            logger.debug("Saved profile: {}", profile.person_id)

        except Exception:
            logger.exception("Failed to save profile: {}", profile.person_id)
            raise

    # ------------------------------------------------------------------
    # 保存 (批量)
    # ------------------------------------------------------------------

    async def save_all_profiles(
        self, profiles: dict[str, PersonProfile], camera_id: str,
    ) -> None:
        """批量保存所有人物档案 (单事务)。"""
        try:
            async with AsyncSession(self.engine) as session:
                stmt = select(PersonRow).where(
                    PersonRow.camera_id == camera_id
                )
                result = await session.exec(stmt)
                for old_row in result.all():
                    await session.delete(old_row)
                await session.flush()

                for profile in profiles.values():
                    row = self._profile_to_row(profile, camera_id)
                    session.add(row)

                await session.commit()

            logger.info(
                "Batch saved {} profiles (camera={})",
                len(profiles), camera_id,
            )
        except Exception:
            logger.exception("Failed to batch save profiles")
            raise

    # ------------------------------------------------------------------
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

                    profile = self._row_to_profile(
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

    # ------------------------------------------------------------------
    # 内部: PersonProfile ↔ ORM Row 转换
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_to_row(
        profile: PersonProfile, camera_id: str,
    ) -> PersonRow:
        bp = profile.body_proportions

        row = PersonRow(
            person_id=profile.person_id,
            camera_id=camera_id,
            display_name=profile.display_name,
            created_at=profile.created_at,
            last_updated=profile.last_updated,
            update_count=profile.update_count,
            vlm_description=profile.vlm_description,
            bp_torso_leg=bp.torso_leg_ratio if bp else None,
            bp_shoulder_hip=bp.shoulder_hip_ratio if bp else None,
            bp_arm_torso=bp.arm_torso_ratio if bp else None,
            bp_head_body=bp.head_body_ratio if bp else None,
            bp_height_px=bp.relative_height_px if bp else None,
            bp_samples=profile.body_proportions_samples,
        )

        for bucket, entries in profile.face_features.items():
            for entry in entries:
                row.face_features.append(FaceFeatureRow(
                    person_id=profile.person_id,
                    pose_bucket=bucket.value,
                    embedding=entry.embedding.astype(np.float32).tobytes(),
                    quality_score=entry.quality_score,
                    timestamp=entry.timestamp,
                    source_image=entry.source_image,
                ))

        for bucket, entries in profile.body_features.items():
            for entry in entries:
                row.body_features.append(BodyFeatureRow(
                    person_id=profile.person_id,
                    pose_bucket=bucket.value,
                    embedding=entry.embedding.astype(np.float32).tobytes(),
                    quality_score=entry.quality_score,
                    timestamp=entry.timestamp,
                    source_image=entry.source_image,
                ))

        for outfit in profile.wardrobe:
            row.wardrobe_items.append(WardrobeRow(
                person_id=profile.person_id,
                body_embedding=outfit.body_embedding.astype(np.float32).tobytes(),
                quality_score=outfit.quality_score,
                first_seen=outfit.first_seen,
                last_seen=outfit.last_seen,
                seen_count=outfit.seen_count,
            ))

        return row

    @staticmethod
    def _row_to_profile(
        person_row: PersonRow,
        face_rows: list[FaceFeatureRow],
        body_rows: list[BodyFeatureRow],
        wardrobe_rows: list[WardrobeRow],
    ) -> PersonProfile:
        bp: BodyProportions | None = None
        if person_row.bp_torso_leg is not None:
            bp = BodyProportions(
                torso_leg_ratio=person_row.bp_torso_leg,
                shoulder_hip_ratio=person_row.bp_shoulder_hip,
                arm_torso_ratio=person_row.bp_arm_torso,
                head_body_ratio=person_row.bp_head_body,
                relative_height_px=person_row.bp_height_px,
            )

        profile = PersonProfile(
            person_id=person_row.person_id,
            display_name=person_row.display_name,
            created_at=person_row.created_at,
            last_updated=person_row.last_updated,
            update_count=person_row.update_count,
            vlm_description=person_row.vlm_description,
            body_proportions=bp,
            body_proportions_samples=person_row.bp_samples,
        )

        for face_row in face_rows:
            embedding = np.frombuffer(
                face_row.embedding, dtype=np.float32,
            ).copy()
            bucket = PoseBucket(face_row.pose_bucket)
            entry = FeatureEntry(
                embedding=embedding,
                pose_bucket=bucket,
                quality_score=face_row.quality_score,
                timestamp=face_row.timestamp,
                source_image=face_row.source_image,
            )
            profile.face_features.setdefault(bucket, []).append(entry)

        for body_row in body_rows:
            embedding = np.frombuffer(
                body_row.embedding, dtype=np.float32,
            ).copy()
            bucket = PoseBucket(body_row.pose_bucket)
            entry = FeatureEntry(
                embedding=embedding,
                pose_bucket=bucket,
                quality_score=body_row.quality_score,
                timestamp=body_row.timestamp,
                source_image=body_row.source_image,
            )
            profile.body_features.setdefault(bucket, []).append(entry)

        for wardrobe_row in wardrobe_rows:
            body_embedding = np.frombuffer(
                wardrobe_row.body_embedding, dtype=np.float32,
            ).copy()
            outfit = OutfitRecord(
                body_embedding=body_embedding,
                quality_score=wardrobe_row.quality_score,
                first_seen=wardrobe_row.first_seen,
                last_seen=wardrobe_row.last_seen,
                seen_count=wardrobe_row.seen_count,
            )
            profile.wardrobe.append(outfit)

        return profile
