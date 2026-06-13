"""
Gallery Converters — ORM ↔ 领域模型转换

将 PersonProfile / FeatureEntry / OutfitRecord 等业务模型
与 PersonRow / FaceFeatureRow / BodyFeatureRow / WardrobeRow 等 ORM 行
之间的转换逻辑集中在此模块, 供 persistence.py 调用。
"""
from __future__ import annotations

import json
from typing import TypeVar

import numpy as np

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
    FeatureRowBase,
    PersonRow,
    WardrobeRow,
)

_F = TypeVar("_F", bound=FeatureRowBase)


# ==============================================================================
# FeatureEntry ↔ FeatureRow (FaceFeatureRow / BodyFeatureRow 通用)
# ==============================================================================

def entry_to_feature_row(
    person_id: str, entry: FeatureEntry, row_cls: type[_F],
) -> _F:
    """FeatureEntry → FeatureRow 转换。"""
    return row_cls(
        person_id=person_id,
        pose_bucket=entry.pose_bucket.value,
        embedding=entry.embedding.astype(np.float32).tobytes(),
        quality_score=entry.quality_score,
        timestamp=entry.timestamp,
        source_image=entry.source_image,
        overlay_bbox=json.dumps(entry.overlay_bbox) if entry.overlay_bbox else None,
    )


def feature_row_to_entry(row: FeatureRowBase) -> FeatureEntry:
    """FeatureRow → FeatureEntry 转换。"""
    overlay_bbox = None
    if row.overlay_bbox:
        try:
            overlay_bbox = json.loads(row.overlay_bbox)
        except (json.JSONDecodeError, TypeError):
            pass
    return FeatureEntry(
        embedding=np.frombuffer(row.embedding, dtype=np.float32).copy(),
        pose_bucket=PoseBucket(row.pose_bucket),
        quality_score=row.quality_score,
        timestamp=row.timestamp,
        source_image=row.source_image,
        overlay_bbox=overlay_bbox,
    )


# ==============================================================================
# OutfitRecord ↔ WardrobeRow
# ==============================================================================

def outfit_to_row(person_id: str, outfit: OutfitRecord) -> WardrobeRow:
    """OutfitRecord → WardrobeRow 转换。"""
    return WardrobeRow(
        person_id=person_id,
        body_embedding=outfit.body_embedding.astype(np.float32).tobytes(),
        quality_score=outfit.quality_score,
        first_seen=outfit.first_seen,
        last_seen=outfit.last_seen,
        seen_count=outfit.seen_count,
    )


def sync_outfit_fields(row: WardrobeRow, outfit: OutfitRecord) -> None:
    """将 OutfitRecord 的字段同步到已有 WardrobeRow (UPDATE 场景)。"""
    row.body_embedding = outfit.body_embedding.astype(np.float32).tobytes()
    row.quality_score = outfit.quality_score
    row.last_seen = outfit.last_seen
    row.seen_count = outfit.seen_count


def wardrobe_row_to_outfit(row: WardrobeRow) -> OutfitRecord:
    """WardrobeRow → OutfitRecord 转换。"""
    return OutfitRecord(
        body_embedding=np.frombuffer(
            row.body_embedding, dtype=np.float32,
        ).copy(),
        quality_score=row.quality_score,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        seen_count=row.seen_count,
    )


# ==============================================================================
# PersonProfile ↔ PersonRow (含字段同步)
# ==============================================================================

def sync_person_fields(
    row: PersonRow, profile: PersonProfile,
) -> None:
    """将 PersonProfile 的字段同步到 PersonRow (新建 / 更新通用)。"""
    bp = profile.body_proportions
    row.display_name = profile.display_name
    row.created_at = profile.created_at
    row.last_updated = profile.last_updated
    row.update_count = profile.update_count
    row.vlm_description = profile.vlm_description
    row.bp_torso_leg = bp.torso_leg_ratio if bp else None
    row.bp_shoulder_hip = bp.shoulder_hip_ratio if bp else None
    row.bp_arm_torso = bp.arm_torso_ratio if bp else None
    row.bp_head_body = bp.head_body_ratio if bp else None
    row.bp_height_px = bp.relative_height_px if bp else None
    row.bp_samples = profile.body_proportions_samples


# ==============================================================================
# 整体转换: PersonProfile ↔ 全部 ORM Row
# ==============================================================================


def rows_to_profile(
    person_row: PersonRow,
    face_rows: list[FaceFeatureRow],
    body_rows: list[BodyFeatureRow],
    wardrobe_rows: list[WardrobeRow],
) -> PersonProfile:
    """PersonRow + 子表行 → PersonProfile。"""
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
        entry = feature_row_to_entry(face_row)
        profile.face_features.setdefault(entry.pose_bucket, []).append(entry)

    for body_row in body_rows:
        entry = feature_row_to_entry(body_row)
        profile.body_features.setdefault(entry.pose_bucket, []).append(entry)

    for wardrobe_row in wardrobe_rows:
        profile.wardrobe.append(wardrobe_row_to_outfit(wardrobe_row))

    return profile
