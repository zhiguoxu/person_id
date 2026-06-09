"""
Tier2/Tier3 调度逻辑

状态驱动的固定间隔调度:
- IDENTIFYING/SUSPECTED/CONFLICT → 1s 快速间隔
- CONFIDENT/STRANGER → 5s 慢速间隔
- DEFINITE → 20s 后台富化
- 非注意力目标: 所有间隔 × non_attention_factor
"""
from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING

from src.config import get_config
from src.pipeline.data_models import IdentityStatus

if TYPE_CHECKING:
    from src.pipeline.track_state import TrackState


class Tier2Action(str, Enum):
    """Tier2 调度动作"""
    SKIP = "skip"                   # 不触发
    TRIGGER_REID = "reid"           # 触发 Tier2 ReID
    TRIGGER_ENRICH = "enrich"       # 触发后台富化


def should_trigger_tier2(state: TrackState) -> Tier2Action:
    """判断是否应该触发 Tier2 处理

    前置条件: caller 已保证 RecentBuffer 中有数据
    """
    config = get_config().multiframe
    now = time.monotonic()
    status = state.identity_result.status

    # force_probe: 立即触发 (Gallery 更新或冲突后)
    if state.force_probe:
        return Tier2Action.TRIGGER_REID

    # DEFINITE: 唯一终态, 仅做后台富化
    if status == IdentityStatus.DEFINITE:
        elapsed = now - state.last_tier2_time
        scale = 1.0 if state.is_current_target else config.non_attention_factor
        if elapsed >= config.definite_enrich_interval * scale:
            return Tier2Action.TRIGGER_ENRICH
        return Tier2Action.SKIP

    # 计算基准间隔
    if status in (IdentityStatus.IDENTIFYING, IdentityStatus.SUSPECTED,
                  IdentityStatus.CONFLICT):
        base_interval = config.tier2_fast_interval  # 1s
    else:
        # CONFIDENT, STRANGER
        base_interval = config.tier2_slow_interval  # 5s

    # 非注意力目标: 间隔 × 2
    scale = 1.0 if state.is_current_target else config.non_attention_factor
    interval = base_interval * scale

    elapsed = now - state.last_tier2_time
    if elapsed >= interval:
        return Tier2Action.TRIGGER_REID

    return Tier2Action.SKIP


def should_trigger_vlm(state: TrackState) -> bool:
    """判断是否应该触发 Tier3 VLM 仲裁

    条件:
    1. VLM 已启用
    2. 当前状态为 SUSPECTED 或 CONFLICT
    3. VLM 冷却期已过
    """
    if not get_config().vlm.enabled:
        return False

    if state.identity_result.status not in (IdentityStatus.SUSPECTED, IdentityStatus.CONFLICT):
        return False

    config = get_config().multiframe
    now = time.monotonic()
    scale = 1.0 if state.is_current_target else config.non_attention_factor
    elapsed = now - state.last_vlm_time
    return elapsed >= config.vlm_cooldown * scale
