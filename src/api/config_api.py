"""
运行配置查询 API。

  GET /api/config   返回 person_id 当前生效的完整配置(已脱敏)
                    供 web 控制台「系统配置」页展示(前端经代理以 /vision/api/config 访问)
"""
import os

import session_store
import voice_agent_common
from fastapi import APIRouter

from src import deps
from src.configs.config import config
from voice_agent_common.utils.config_dump import sanitize_config_for_api
from voice_agent_common.utils.pkg_versions import package_versions

config_router = APIRouter(prefix="/api/config", tags=["config"])


@config_router.get("")
async def get_config():
    """脱敏规则见 voice_agent_common.utils.config_dump, 密钥/密码/token 一律替换为 ***。"""
    return {
        "service": "person_id",
        "version": config.version,
        "env": os.getenv("APP_ENV", "dev"),
        # naive 北京时间，与项目时钟约定一致；前端按字面展示
        "started_at": (
            deps.started_at.isoformat(sep=" ", timespec="seconds")
            if deps.started_at else None
        ),
        "packages": package_versions(
            common=voice_agent_common,
            session_store=session_store,
        ),
        "config": sanitize_config_for_api(config),
    }
