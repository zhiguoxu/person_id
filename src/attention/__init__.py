"""注意力模块 — 目标选择与注意力评分"""
from src.attention.engine import AttentionEngine


def create_attention_engine(config):
    """创建注意力引擎。"""
    return AttentionEngine(config)


__all__ = ["AttentionEngine", "create_attention_engine"]
