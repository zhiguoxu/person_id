"""
Gallery Persistence — SQLite 持久化

使用 aiosqlite 异步地将底库 PersonProfile 存储到 SQLite 数据库。
特征向量以 numpy 二进制 blob 存储 (tobytes / frombuffer), 保证精度无损。

表结构:
    - persons: 人物基本信息 + 体型比例
    - face_features: 人脸特征条目 (一人多条, 按姿态桶)
    - wardrobe: 衣橱记录 (一人多条)
"""
from __future__ import annotations

import json
from typing import Optional

import aiosqlite
import numpy as np
from loguru import logger

from src.gallery.data_models import (
    BodyProportions,
    FeatureEntry,
    OutfitRecord,
    PersonProfile,
    PoseBucket,
)

# 默认特征维度 (InsightFace=512, ReID=2048), 通过 blob 自适应
_FACE_EMBED_DIM = 512
_BODY_EMBED_DIM = 2048


class GalleryPersistence:
    """SQLite 底库持久化管理器。

    所有公共方法均为异步 (async), 需在 asyncio 事件循环中调用。
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        logger.info("GalleryPersistence configured with db: {}", db_path)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """创建数据库连接和表结构。

        幂等操作: 已存在的表不会被覆盖。
        """
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                person_id       TEXT PRIMARY KEY,
                display_name    TEXT NOT NULL,
                created_at      REAL NOT NULL,
                last_seen       REAL NOT NULL,
                total_appearances INTEGER NOT NULL DEFAULT 0,
                vlm_description TEXT,
                -- 体型比例 (nullable)
                bp_torso_leg    REAL,
                bp_shoulder_hip REAL,
                bp_arm_torso    REAL,
                bp_head_body    REAL,
                bp_height_px    REAL,
                bp_samples      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS face_features (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id       TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                pose_bucket     TEXT NOT NULL,
                embedding       BLOB NOT NULL,
                quality_score   REAL NOT NULL,
                timestamp       REAL NOT NULL,
                source_image    BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_face_person
                ON face_features(person_id);

            CREATE TABLE IF NOT EXISTS wardrobe (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id       TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                body_embedding  BLOB NOT NULL,
                quality_score   REAL NOT NULL,
                first_seen      REAL NOT NULL,
                last_seen       REAL NOT NULL,
                seen_count      INTEGER NOT NULL DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_wardrobe_person
                ON wardrobe(person_id);
            """
        )
        await self._db.commit()
        logger.info("Database initialized at {}", self._db_path)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------

    async def save_profile(self, profile: PersonProfile) -> None:
        """将 PersonProfile 完整写入数据库 (INSERT OR REPLACE)。

        Args:
            profile: 待保存的人物档案。
        """
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        try:
            # 写入 persons 表
            bp = profile.body_proportions
            await self._db.execute(
                """
                INSERT OR REPLACE INTO persons
                    (person_id, display_name, created_at, last_seen,
                     total_appearances, vlm_description,
                     bp_torso_leg, bp_shoulder_hip, bp_arm_torso,
                     bp_head_body, bp_height_px, bp_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.person_id,
                    profile.display_name,
                    profile.created_at,
                    profile.last_seen,
                    profile.total_appearances,
                    profile.vlm_description,
                    bp.torso_leg_ratio if bp else None,
                    bp.shoulder_hip_ratio if bp else None,
                    bp.arm_torso_ratio if bp else None,
                    bp.head_body_ratio if bp else None,
                    bp.relative_height_px if bp else None,
                    profile.body_proportions_samples,
                ),
            )

            # 删除旧的人脸特征和衣橱 (重写策略)
            await self._db.execute(
                "DELETE FROM face_features WHERE person_id = ?",
                (profile.person_id,),
            )
            await self._db.execute(
                "DELETE FROM wardrobe WHERE person_id = ?",
                (profile.person_id,),
            )

            # 写入人脸特征
            for bucket, entries in profile.face_features.items():
                for entry in entries:
                    await self._db.execute(
                        """
                        INSERT INTO face_features
                            (person_id, pose_bucket, embedding,
                             quality_score, timestamp, source_image)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            profile.person_id,
                            bucket.value,
                            entry.embedding.astype(np.float32).tobytes(),
                            entry.quality_score,
                            entry.timestamp,
                            entry.source_image,
                        ),
                    )

            # 写入衣橱
            for outfit in profile.wardrobe:
                await self._db.execute(
                    """
                    INSERT INTO wardrobe
                        (person_id, body_embedding, quality_score,
                         first_seen, last_seen, seen_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.person_id,
                        outfit.body_embedding.astype(np.float32).tobytes(),
                        outfit.quality_score,
                        outfit.first_seen,
                        outfit.last_seen,
                        outfit.seen_count,
                    ),
                )

            await self._db.commit()
            logger.debug("Saved profile: {}", profile.person_id)

        except Exception:
            logger.exception("Failed to save profile: {}", profile.person_id)
            raise

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    async def load_all_profiles(self) -> dict[str, PersonProfile]:
        """从数据库加载全部人物档案。

        Returns:
            字典, key=person_id, value=PersonProfile。
        """
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        profiles: dict[str, PersonProfile] = {}

        try:
            # 读取 persons 表
            async with self._db.execute("SELECT * FROM persons") as cursor:
                rows = await cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

            for row in rows:
                data = dict(zip(col_names, row))
                pid = data["person_id"]

                # 重建体型比例
                bp: Optional[BodyProportions] = None
                if data["bp_torso_leg"] is not None:
                    bp = BodyProportions(
                        torso_leg_ratio=data["bp_torso_leg"],
                        shoulder_hip_ratio=data["bp_shoulder_hip"],
                        arm_torso_ratio=data["bp_arm_torso"],
                        head_body_ratio=data["bp_head_body"],
                        relative_height_px=data["bp_height_px"],
                    )

                profile = PersonProfile(
                    person_id=pid,
                    display_name=data["display_name"],
                    created_at=data["created_at"],
                    last_seen=data["last_seen"],
                    total_appearances=data["total_appearances"],
                    vlm_description=data["vlm_description"],
                    body_proportions=bp,
                    body_proportions_samples=data["bp_samples"],
                )
                profiles[pid] = profile

            # 读取人脸特征
            async with self._db.execute(
                "SELECT * FROM face_features ORDER BY person_id"
            ) as cursor:
                face_rows = await cursor.fetchall()
                face_cols = [desc[0] for desc in cursor.description]

            for row in face_rows:
                data = dict(zip(face_cols, row))
                pid = data["person_id"]
                if pid not in profiles:
                    continue

                embedding = np.frombuffer(
                    data["embedding"], dtype=np.float32
                ).copy()
                bucket = PoseBucket(data["pose_bucket"])

                entry = FeatureEntry(
                    embedding=embedding,
                    pose_bucket=bucket,
                    quality_score=data["quality_score"],
                    timestamp=data["timestamp"],
                    source_image=data["source_image"],
                )
                profiles[pid].face_features.setdefault(bucket, []).append(entry)

            # 读取衣橱
            async with self._db.execute(
                "SELECT * FROM wardrobe ORDER BY person_id"
            ) as cursor:
                wardrobe_rows = await cursor.fetchall()
                wardrobe_cols = [desc[0] for desc in cursor.description]

            for row in wardrobe_rows:
                data = dict(zip(wardrobe_cols, row))
                pid = data["person_id"]
                if pid not in profiles:
                    continue

                body_embedding = np.frombuffer(
                    data["body_embedding"], dtype=np.float32
                ).copy()

                outfit = OutfitRecord(
                    body_embedding=body_embedding,
                    quality_score=data["quality_score"],
                    first_seen=data["first_seen"],
                    last_seen=data["last_seen"],
                    seen_count=data["seen_count"],
                )
                profiles[pid].wardrobe.append(outfit)

            logger.info("Loaded {} profiles from database", len(profiles))

        except Exception:
            logger.exception("Failed to load profiles from database")
            raise

        return profiles

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    async def delete_profile(self, person_id: str) -> None:
        """从数据库删除指定人物档案 (级联删除关联的特征和衣橱)。

        Args:
            person_id: 要删除的人物 ID。
        """
        if self._db is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        try:
            await self._db.execute(
                "DELETE FROM persons WHERE person_id = ?", (person_id,)
            )
            await self._db.commit()
            logger.info("Deleted profile: {}", person_id)
        except Exception:
            logger.exception("Failed to delete profile: {}", person_id)
            raise

    # ------------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------------

    async def update_profile(self, profile: PersonProfile) -> None:
        """更新已存在的人物档案 (等同于 save_profile 的 INSERT OR REPLACE)。

        Args:
            profile: 待更新的人物档案。
        """
        await self.save_profile(profile)

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """关闭数据库连接。"""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("Database connection closed")
