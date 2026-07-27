"""自动重推流共享数据模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RestreamLogLine(BaseModel):
    """自动重推流过程中的一条日志。"""

    time: float
    level: str = Field(description="info | warning | error")
    message: str


class RestreamAttempt(BaseModel):
    """一次自动重推流尝试的完整记录。"""

    camera_id: str
    env: str
    started_at: float
    finished_at: float
    trigger_fail_count: int = Field(description="触发本次恢复的连续拉流失败次数")
    trigger_error: str = Field("", description="触发时拉流侧的最后错误")
    device_online: bool | None = Field(
        None, description="voice_server 设备在线检查结果 (null=检查失败)"
    )
    outcome: str = Field(
        description="restreamed | device_offline | iss_start_failed | error"
    )
    old_url: str = ""
    new_url: str = Field("", description="重推成功时的新 FLV 地址")
    logs: list[RestreamLogLine] = Field(default_factory=list)
