"""person_id 的 loguru sink 重建。

voice_agent_common.utils.logger 在 import 期会 logger.remove() 掉全部已有
sink，换成带 {extra[device_sn]}/{extra[trace_id]} 列的公共格式（voice/agent
的日志链路）。person_id 全工程用的是未打补丁的裸 `from loguru import logger`，
没有这些 extra 字段——公共 sink 对它的每条日志都会格式化失败刷错误。

person_id 引入 common 的配置覆盖模块（src.configs.override）后必然触发上述
副作用，因此服务入口在完成 import 之后必须调用本函数重建自己的 sink。
"""
from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """重置 loguru 为 person_id 自己的控制台输出（格式同 loguru 默认风格）。"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        backtrace=True,
        diagnose=True,
    )
