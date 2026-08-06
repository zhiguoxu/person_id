"""懒建 Redis 连接池的公共封装。

项目里每个 Redis 用途各持一个连接池(db 可能不同, 如拉流期望状态在
config.redis.db, voice_server 在线标记在其主库), 但"未配置即降级 +
懒建 + 统一连接参数"的行为是一致的, 收敛在这里:

- host 未配置(redis.host 为空)时 get() 返回 None 并只警告一次,
  由调用方按各自语义降级, 不阻断业务
- 连接参数(protocol=2 兼容老版本服务端、超时口径)与
  voice_agent_common 保持一致, 只在这一处维护
"""
from __future__ import annotations

from typing import Callable

import redis.asyncio as redis
from voice_agent_common.utils.logger import logger

from src.configs.config import Config, config


class LazyRedis:
    """按需创建的 Redis 连接池(每个用途一个实例, db 由 db_of 从配置解析)。"""

    def __init__(self, db_of: Callable[[Config], int], unconfigured_hint: str):
        """
        Args:
            db_of: 从 Config 解析本用途 db 编号(连接懒建, 配置加载后才会调用)
            unconfigured_hint: Redis 未配置时的一次性警告文案(说明降级后果)
        """
        self._db_of = db_of
        self._unconfigured_hint = unconfigured_hint
        self._client: redis.Redis | None = None
        self._warned_unconfigured = False

    def get(self) -> redis.Redis | None:
        """取连接; 未配置(host 为空)返回 None 并只警告一次。"""
        cfg = config
        if not cfg.redis.host:
            if not self._warned_unconfigured:
                self._warned_unconfigured = True
                logger.warning(self._unconfigured_hint)
            return None
        if self._client is None:
            # protocol=2 兼容老版本 Redis 服务端(与 voice_agent_common 口径一致)
            self._client = redis.Redis(
                host=cfg.redis.host,
                port=cfg.redis.port,
                password=cfg.redis.password or None,
                db=self._db_of(cfg),
                decode_responses=True,
                protocol=2,
                socket_connect_timeout=cfg.redis.socket_connect_timeout,
                socket_timeout=cfg.redis.socket_timeout,
            )
        return self._client

    async def close(self) -> None:
        """关闭连接池(lifespan shutdown 调用; 未建连时为空操作)。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
