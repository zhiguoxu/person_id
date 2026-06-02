"""Identity package — 歧义消解、VLM 仲裁与多模态融合。"""
from src.identity.resolver import AmbiguityResolver
from src.identity.vlm_arbitrator import VLMArbitrator
from src.identity.multi_modal_fusion import MultiModalFusion


def create_reranker(config):
    """创建重排序器 (K-Reciprocal)。"""
    from src.gallery import KReciprocalReranker
    return KReciprocalReranker()


def create_fusion(config):
    """创建多模态融合器。"""
    return MultiModalFusion(config.matching)


def create_resolver(config):
    """创建歧义消解器。"""
    return AmbiguityResolver(config)


__all__ = [
    "AmbiguityResolver",
    "VLMArbitrator",
    "MultiModalFusion",
    "create_reranker",
    "create_fusion",
    "create_resolver",
]
