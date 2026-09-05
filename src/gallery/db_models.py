"""
Gallery DB Models — SQLModel ORM 表定义

使用 SQLModel 定义底库表结构的 ORM 模型, 同一份定义在 SQLite / MySQL 上建表
(引擎由 gallery_db_url 决定, 见 persistence.py)。
这些模型仅负责数据库映射，业务逻辑仍在 data_models.py 的 Pydantic 模型中。

表结构:
    - persons: 人物基本信息 + 体型比例
    - face_features: 人脸特征条目 (一人多条, 按姿态桶)
    - body_features: 人体特征条目 (一人多条, 按姿态桶)
    - wardrobe: 衣橱记录 (一人多条)

列类型对 MySQL 的两处显式声明 (SQLite 动态类型, 声明什么都按原值存, 不受影响):
    - float 一律 Double: MySQL 的 FLOAT 是单精度, Unix 时间戳 (~1.7e9) 会被舍到
      128s 粒度, replace_feature_in / update_outfit_in 按 timestamp / first_seen
      等值定位旧行会静默落空, 造成特征重复堆积
    - source_image 用 LONGBLOB: 人体特征存的是全帧 PNG (数 MB), MySQL 默认 BLOB
      上限 64KB 直接写入失败
"""

from sqlalchemy import Double, LargeBinary, Text
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlmodel import SQLModel, Field, Relationship

# 图片列: SQLite 仍是普通 BLOB, MySQL 换成 LONGBLOB (4GB 上限)
_ImageBlob = LargeBinary().with_variant(LONGBLOB(), "mysql")


# ==============================================================================
# persons 表
# ==============================================================================

class PersonRow(SQLModel, table=True):
    """人物基本信息 + 体型比例。"""

    __tablename__ = "persons"

    person_id: str = Field(primary_key=True)
    camera_id: str = Field(index=True)
    display_name: str
    created_at: float = Field(sa_type=Double)
    last_updated: float = Field(sa_type=Double)
    update_count: int = Field(default=0)
    # VLM 生成的自由文本描述, 长度不可控, 不能落 MySQL 的 VARCHAR(255)
    vlm_description: str | None = Field(default=None, sa_type=Text)

    # 体型比例 (nullable)
    bp_torso_leg: float | None = Field(default=None, sa_type=Double)
    bp_shoulder_hip: float | None = Field(default=None, sa_type=Double)
    bp_arm_torso: float | None = Field(default=None, sa_type=Double)
    bp_head_body: float | None = Field(default=None, sa_type=Double)
    bp_height_px: float | None = Field(default=None, sa_type=Double)
    bp_samples: int = Field(default=0)

    # Relationships
    face_features: list["FaceFeatureRow"] = Relationship(
        back_populates="person",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    body_features: list["BodyFeatureRow"] = Relationship(
        back_populates="person",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    wardrobe_items: list["WardrobeRow"] = Relationship(
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
    embedding: bytes  # numpy float32 tobytes (几 KB, 普通 BLOB 够用)
    quality_score: float = Field(sa_type=Double)
    timestamp: float = Field(sa_type=Double)
    source_image: bytes | None = Field(default=None, sa_type=_ImageBlob)
    overlay_bbox: str | None = Field(default=None)  # JSON: [x1,y1,x2,y2] 人脸框或人体框


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
    quality_score: float = Field(sa_type=Double)
    first_seen: float = Field(sa_type=Double)
    last_seen: float = Field(sa_type=Double)
    seen_count: int = Field(default=1)

    person: PersonRow | None = Relationship(back_populates="wardrobe_items")
