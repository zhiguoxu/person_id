"""配置 DB 覆盖层的服务侧接线（person_id）。

与 voice_server / agent_server 同一套机制（voice_agent_common.config_override）：
yaml + 环境变量加载不动，数据库只存"被改过的项"，启动时 load_and_apply 把
覆盖值套到 config 内存单例上；hot 字段的消费点本来就是调用时现读单例
（如 `cfg = config.matching`），改完立即生效。

存储与 voice/agent 完全同款：session_store.ConfigOverrideStore（同一张
config_overrides 表），连接由 config.db_url 决定——dev 先落本地 SQLite，
切共享 MySQL 时只改 yaml 里的 db_url 一行。

与那两个服务的唯一差异：没有设备级覆盖（person_id 不存在按 device_sn
解析配置的消费链路，editable_fields 里全部字段 device_editable=False），
因此也不提供 current_config() 入口——业务代码继续直读 config 单例即可。
"""

from src.configs.config import config
from src.configs.editable_fields import FIELD_ANNOTATIONS, LOCKED_PATHS
from session_store import ConfigOverrideStore
from voice_agent_common.config_override import ConfigOverrideManager

# 实例化很轻, 真正读库套值在 src/api/server.py 的 lifespan 里(load_and_apply)
config_override_manager = ConfigOverrideManager(
    config, "person_id", FIELD_ANNOTATIONS, ConfigOverrideStore(), LOCKED_PATHS)
