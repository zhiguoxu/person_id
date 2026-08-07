"""Module-level dependency container, initialized in src/main.py lifespan."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_agent_common.infra.redis_client import RedisClient

# 本进程最近一次 lifespan 启动时刻(naive 北京时间)，供 /api/config 展示
started_at: datetime | None = None

# 日志聚合专用 Redis 连接(独立 db), 在 server.py lifespan 启动时建立
log_redis_client: RedisClient | None = None
