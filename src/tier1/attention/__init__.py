"""注意力模块 — 目标选择与注意力评分"""

from __future__ import annotations
from src.tier1.attention.engine import AttentionEngine, select_best_detection


__all__ = ["AttentionEngine", "select_best_detection"]
