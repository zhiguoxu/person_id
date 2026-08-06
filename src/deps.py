"""Module-level dependency container, initialized in src/api/server.py lifespan."""
from __future__ import annotations

from datetime import datetime

# 本进程最近一次 lifespan 启动时刻(naive 北京时间)，供 /api/config 展示
started_at: datetime | None = None
