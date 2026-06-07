"""Tier 3 — VLM 仲裁阶段。"""

from __future__ import annotations

from functools import cache

from .vlm_arbitrator import VLMArbitrator


@cache
def get_vlm_arbitrator() -> VLMArbitrator:
    """创建 VLM 仲裁器（单例缓存）。"""
    return VLMArbitrator()


__all__ = [
    "VLMArbitrator",
    "get_vlm_arbitrator",
]
