"""Module-level dependency container, initialized in src/main.py lifespan."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_agent_common.infra.oss.cos import CosClient
    from voice_agent_common.infra.redis_client import RedisClient

# 本进程最近一次 lifespan 启动时刻(naive 北京时间)，供 /api/config 展示
started_at: datetime | None = None

# 日志聚合专用 Redis 连接(独立 db), 在 server.py lifespan 启动时建立
log_redis_client: RedisClient | None = None

# 拉流期望状态/租约连接(config.redis.db 主库, pipeline/stream_state.py 消费)。
# redis 是必配块, lifespan 启动后必然非 None
stream_state_redis_client: RedisClient | None = None

# voice_server 在线标记直读连接(voice_online_redis_db, pipeline/restream.py
# 重推流前的设备在线检查)。同上, lifespan 启动后必然非 None
voice_online_redis_client: RedisClient | None = None

# 拉流录像上传 COS (与 voice_server 同款 CosClient)
cos_client: CosClient | None = None
