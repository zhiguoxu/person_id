"""配置 DB 覆盖层的服务侧接线（person_id）。

与 voice_server / agent_server 同一套机制（voice_agent_common.config_override）：
yaml + 环境变量加载不动，数据库只存"被改过的项"。存储同款 session_store
的 config_overrides 表，连接由 config.db_url 决定。

current_config() 是运行期生效配置的唯一读取入口：hot 字段（matching/
gallery/multiframe 阈值、拉流/录像运行参数等）的消费点一律经它现读——
返回"baseline + 全局覆盖"的配置快照。config 单例是启动期状态（启动时
load_and_apply 套过一次覆盖后冻结），模型加载/建连等启动期消费读它是
正确的。与 voice/agent 的唯一差异：没有设备级覆盖（editable_fields 里
全部字段 device_editable=False），current_config() 恒返回全局快照。

config_sync 是多实例同步通道（必选）：写覆盖后向 Redis Stream 追加通知，
其他实例读到即 reload 重读 DB（自己发的通知按实例标识跳过）。流 key 的
命名空间复用 voice_live_namespace（本服务所属环境套的命名空间，与
voice/agent 的 live_namespace 同源）。
"""

from typing import cast

from src.configs.config import Config, config
from src.configs.editable_fields import FIELD_ANNOTATIONS, LOCKED_PATHS
from session_store import ConfigOverrideStore
from voice_agent_common.config_override import ConfigOverrideManager
from voice_agent_common.config_sync import ConfigSyncChannel
from voice_agent_common.infra.redis_client import RedisClient

# 实例化很轻且不建连: Redis 连接在 src/main.py 的 lifespan 里 start(),
# 真正读库套值在 load_and_apply()
config_sync = ConfigSyncChannel(
    RedisClient(config.redis), namespace=config.voice_live_namespace, service="person_id")
config_override_manager = ConfigOverrideManager(
    config, "person_id", FIELD_ANNOTATIONS, ConfigOverrideStore(),
    config_sync, LOCKED_PATHS)


def current_config() -> Config:
    """当前生效配置：hot 字段的消费点必须读这里，不要直读 config 单例。

    本服务无设备级覆盖，恒返回全局快照；每帧/每次调用现读的成本是一次
    字典查缓存，可放在逐帧路径上。
    """
    return cast(Config, config_override_manager.config_for(""))
