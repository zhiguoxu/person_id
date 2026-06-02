"""
主动感知建议器 — 引导机器人获取更好的识别信号

根据当前追踪人物的姿态、人脸质量和身份状态，
生成机器人动作建议（如移动到正面、靠近、询问姓名等）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from loguru import logger

from src.config import Config
from src.gallery.data_models import IdentityStatus, PoseBucket, TrackedPerson


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class AdvicePriority(IntEnum):
    """建议优先级 (数值越大越紧急)"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class PerceptionAdvice:
    """
    主动感知建议。

    Attributes:
        action: 建议动作标识符 (e.g. 'move_to_front', 'move_closer', 'ask_name', 'greet')。
        reason: 人类可读的建议理由。
        priority: 建议优先级。
    """
    action: str
    reason: str
    priority: AdvicePriority


# ---------------------------------------------------------------------------
# 感知建议器
# ---------------------------------------------------------------------------

class ActivePerceptionAdvisor:
    """
    主动感知建议器。

    根据追踪人物的当前状态生成动作建议。
    当前为占位实现，后续可集成机器人运动规划。
    """

    # 低于此阈值认为人脸质量不足
    _LOW_FACE_QUALITY: float = 0.4

    def __init__(self, config: Config) -> None:
        """
        初始化感知建议器。

        Args:
            config: 系统配置。
        """
        self._config = config
        logger.info("ActivePerceptionAdvisor initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        person: TrackedPerson,
        identity_status: IdentityStatus,
    ) -> Optional[PerceptionAdvice]:
        """
        为指定人物生成感知建议。

        按优先级从高到低检查条件，返回最紧急的建议。
        若当前状态不需要建议则返回 None。

        Args:
            person: 追踪中的人物信息。
            identity_status: 当前身份识别状态。

        Returns:
            PerceptionAdvice 建议，或 None。
        """
        det = person.detection

        # 1. 背对 → 建议移动到正面 (最高优先级)
        if det.pose_bucket == PoseBucket.BACK:
            return PerceptionAdvice(
                action="move_to_front",
                reason=(
                    f"Track {person.track_id} is back-facing — "
                    "move to the front to enable face recognition"
                ),
                priority=AdvicePriority.HIGH,
            )

        # 2. 人脸质量低 → 建议靠近
        if person.face_quality is not None and person.face_quality < self._LOW_FACE_QUALITY:
            return PerceptionAdvice(
                action="move_closer",
                reason=(
                    f"Track {person.track_id} face quality too low "
                    f"({person.face_quality:.2f}) — move closer for better capture"
                ),
                priority=AdvicePriority.MEDIUM,
            )

        # 3. 身份不确定 → 建议询问姓名
        if identity_status in (IdentityStatus.SUSPECTED, IdentityStatus.CONFLICT):
            return PerceptionAdvice(
                action="ask_name",
                reason=(
                    f"Track {person.track_id} identity is {identity_status.value} — "
                    "consider asking for their name to confirm"
                ),
                priority=AdvicePriority.MEDIUM,
            )

        # 4. 陌生人 → 建议打招呼
        if identity_status == IdentityStatus.STRANGER:
            return PerceptionAdvice(
                action="greet",
                reason=(
                    f"Track {person.track_id} is a stranger — "
                    "greet and introduce to register"
                ),
                priority=AdvicePriority.LOW,
            )

        return None
