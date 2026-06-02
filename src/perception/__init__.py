"""感知模块 — VLM 仲裁与主动感知建议"""
from src.perception.advisor import ActivePerceptionAdvisor, PerceptionAdvice, AdvicePriority


def create_vlm(config):
    """创建 VLM 仲裁器。"""
    from src.identity import VLMArbitrator
    return VLMArbitrator(config.vlm)


__all__ = [
    "ActivePerceptionAdvisor",
    "PerceptionAdvice",
    "AdvicePriority",
    "create_vlm",
]
