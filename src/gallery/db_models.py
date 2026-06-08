"""
Gallery DB Models — SQLModel ORM 表定义

使用 SQLModel 定义与现有 SQLite 表结构一一对应的 ORM 模型。
这些模型仅负责数据库映射，业务逻辑仍在 data_models.py 的 Pydantic 模型中。

表结构:
    - persons: 人物基本信息 + 体型比例
    - face_features: 人脸特征条目 (一人多条, 按姿态桶)
    - body_features: 人体特征条目 (一人多条, 按姿态桶)
    - wardrobe: 衣橱记录 (一人多条)
"""
from __future__ import annotations

from sqlmodel import SQLModel, Field, Relationship


# ==============================================================================
# persons 表
# ==============================================================================

class PersonRow(SQLModel, table=True):
    """人物基本信息 + 体型比例。"""

    __tablename__ = "persons"

    person_id: str = Field(primary_key=True)
    camera_id: str = Field(index=True)
    display_name: str
    created_at: float
    last_updated: float
    update_count: int = Field(default=0)
    vlm_description: str | None = Field(default=None)

    # 体型比例 (nullable)
    bp_torso_leg: float | None = Field(default=None)
    bp_shoulder_hip: float | None = Field(default=None)
    bp_arm_torso: float | None = Field(default=None)
    bp_head_body: float | None = Field(default=None)
    bp_height_px: float | None = Field(default=None)
    bp_samples: int = Field(default=0)

    # Relationships
    face_features: list[FaceFeatureRow] = Relationship(
        back_populates="person",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    body_features: list[BodyFeatureRow] = Relationship(
        back_populates="person",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    wardrobe_items: list[WardrobeRow] = Relationship(
        back_populates="person",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


# ==============================================================================
# 特征条目基类 (不建表, 仅字段复用)
# ==============================================================================

class FeatureRowBase(SQLModel):
    """人脸/人体特征条目的公共字段。"""

    id: int | None = Field(default=None, primary_key=True)
    person_id: str = Field(foreign_key="persons.person_id")
    pose_bucket: str
    embedding: bytes  # numpy float32 tobytes
    quality_score: float
    timestamp: float
    source_image: bytes | None = Field(default=None)


# ==============================================================================
# face_features 表
# ==============================================================================

class FaceFeatureRow(FeatureRowBase, table=True):
    """人脸特征条目。"""

    __tablename__ = "face_features"

    person: PersonRow | None = Relationship(back_populates="face_features")


# ==============================================================================
# body_features 表
# ==============================================================================

class BodyFeatureRow(FeatureRowBase, table=True):
    """人体特征条目。"""

    __tablename__ = "body_features"

    person: PersonRow | None = Relationship(back_populates="body_features")


# ==============================================================================
# wardrobe 表
# ==============================================================================

class WardrobeRow(SQLModel, table=True):
    """衣橱记录。"""

    __tablename__ = "wardrobe"

    id: int | None = Field(default=None, primary_key=True)
    person_id: str = Field(foreign_key="persons.person_id")
    body_embedding: bytes  # numpy float32 tobytes
    quality_score: float
    first_seen: float
    last_seen: float
    seen_count: int = Field(default=1)

    person: PersonRow | None = Relationship(back_populates="wardrobe_items")
