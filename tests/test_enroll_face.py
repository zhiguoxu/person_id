"""enroll_face: 同桶近重复保护 (cos_sim ≥ 阈值则只比质量)。"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# data_models 只需要 config 快照和 logger, 避开 gallery.__init__ → persistence / Redis
_SRC = Path(__file__).resolve().parents[1] / "src"
_gallery_cfg = SimpleNamespace(
    face_quality_enroll_threshold=0.55,
    face_enroll_near_dup_threshold=0.95,
    max_faces_per_bucket=5,
    face_enroll_half_life_days=100.0,
)


def _ensure_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        if path is not None:
            mod.__path__ = [str(path)]
        sys.modules[name] = mod
    return mod


_ensure_pkg("src", _SRC)
_ensure_pkg("src.gallery", _SRC / "gallery")
_ensure_pkg("src.configs", _SRC / "configs")

_cfg_mod = types.ModuleType("src.configs.config")
_cfg_mod.config = SimpleNamespace(gallery=_gallery_cfg)
sys.modules["src.configs.config"] = _cfg_mod

_override_mod = types.ModuleType("src.configs.override")
_override_mod.current_config = lambda: SimpleNamespace(gallery=_gallery_cfg)
sys.modules["src.configs.override"] = _override_mod

_ensure_pkg("voice_agent_common")
_ensure_pkg("voice_agent_common.utils")
_log_mod = types.ModuleType("voice_agent_common.utils.logger")
_log_mod.logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
sys.modules["voice_agent_common.utils.logger"] = _log_mod

from src.gallery.data_models import FeatureEntry, PersonProfile, PoseBucket  # noqa: E402


def _unit(*vals: float) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float32)
    return a / np.linalg.norm(a)


def _entry(
        *emb: float,
        quality: float = 0.80,
        pose: PoseBucket = PoseBucket.FRONTAL,
        ts: float = 1_700_000_000.0,
) -> FeatureEntry:
    return FeatureEntry(
        embedding=_unit(*emb),
        pose_bucket=pose,
        quality_score=quality,
        timestamp=ts,
    )


@pytest.fixture
def profile() -> PersonProfile:
    return PersonProfile.create_new("t")


def test_first_face_appends(profile):
    op = profile.enroll_face(_entry(1.0, 0.0, quality=0.80))
    assert op is not None and op.evicted is None
    assert len(profile.face_features[PoseBucket.FRONTAL]) == 1


def test_near_dup_higher_quality_replaces(profile):
    old = _entry(1.0, 0.0, quality=0.80, ts=1.0)
    assert profile.enroll_face(old) is not None

    # 与 old 同向, cos=1.0 ≥ 0.95; 质量更高 → 替换
    new = _entry(1.0, 0.0, quality=0.90, ts=2.0)
    op = profile.enroll_face(new)
    assert op is not None
    assert op.evicted is old
    bucket = profile.face_features[PoseBucket.FRONTAL]
    assert len(bucket) == 1
    assert bucket[0] is new


def test_near_dup_lower_or_equal_quality_discards(profile):
    old = _entry(1.0, 0.0, quality=0.90, ts=1.0)
    profile.enroll_face(old)

    assert profile.enroll_face(_entry(1.0, 0.0, quality=0.80, ts=2.0)) is None
    assert profile.enroll_face(_entry(1.0, 0.0, quality=0.90, ts=3.0)) is None
    bucket = profile.face_features[PoseBucket.FRONTAL]
    assert len(bucket) == 1
    assert bucket[0] is old


def test_below_threshold_appends_as_diverse(profile):
    # cos( [1,0], [0.94, sqrt(1-0.94^2)] ) = 0.94 < 0.95 → 视为不同快照
    profile.enroll_face(_entry(1.0, 0.0, quality=0.80))
    other = _entry(0.94, float(np.sqrt(1.0 - 0.94 ** 2)), quality=0.70)
    op = profile.enroll_face(other)
    assert op is not None and op.evicted is None
    assert len(profile.face_features[PoseBucket.FRONTAL]) == 2


def test_at_threshold_is_near_dup(profile):
    old = _entry(1.0, 0.0, quality=0.80)
    profile.enroll_face(old)
    # 0.951 保证 float32 点积仍 ≥ 0.95 (恰好 0.95 会被量化到 0.949999)
    borderline = _entry(0.951, float(np.sqrt(1.0 - 0.951 ** 2)), quality=0.70)
    assert float(np.dot(old.embedding, borderline.embedding)) >= 0.95
    assert profile.enroll_face(borderline) is None
    assert len(profile.face_features[PoseBucket.FRONTAL]) == 1


def test_replaces_closest_near_dup_not_worse_diverse(profile):
    close = _entry(1.0, 0.0, 0.0, quality=0.85, ts=1.0)
    diverse = _entry(0.0, 1.0, 0.0, quality=0.60, ts=2.0)
    profile.enroll_face(close)
    profile.enroll_face(diverse)

    # 更接近 close (cos=1), 质量更高 → 只换 close, diverse 留下
    better = _entry(1.0, 0.0, 0.0, quality=0.95, ts=3.0)
    op = profile.enroll_face(better)
    assert op is not None and op.evicted is close
    bucket = profile.face_features[PoseBucket.FRONTAL]
    assert len(bucket) == 2
    assert any(e is better for e in bucket) and any(e is diverse for e in bucket)


def test_cross_bucket_not_near_dup(profile):
    profile.enroll_face(_entry(1.0, 0.0, quality=0.80, pose=PoseBucket.FRONTAL))
    op = profile.enroll_face(_entry(1.0, 0.0, quality=0.70, pose=PoseBucket.LEFT))
    assert op is not None and op.evicted is None
    assert len(profile.face_features[PoseBucket.FRONTAL]) == 1
    assert len(profile.face_features[PoseBucket.LEFT]) == 1


def test_near_dup_when_bucket_full_does_not_evict_diverse(profile, monkeypatch):
    monkeypatch.setattr(_gallery_cfg, "max_faces_per_bucket", 2)
    close = _entry(1.0, 0.0, quality=0.90, ts=1.0)
    diverse = _entry(0.0, 1.0, quality=0.56, ts=2.0)
    profile.enroll_face(close)
    profile.enroll_face(diverse)
    assert len(profile.face_features[PoseBucket.FRONTAL]) == 2

    # 若走容量淘汰, 低质量 diverse 会被挤掉、再塞进一条 close 的近重复
    worse_dup = _entry(1.0, 0.0, quality=0.80, ts=3.0)
    assert profile.enroll_face(worse_dup) is None
    bucket = profile.face_features[PoseBucket.FRONTAL]
    assert len(bucket) == 2
    assert any(e is close for e in bucket) and any(e is diverse for e in bucket)
