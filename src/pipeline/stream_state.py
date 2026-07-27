"""期望拉流状态持久化(Redis) — 服务重启后自动恢复拉流

consumer_registry 是纯内存状态: person_id 进程重启(部署/崩溃)后所有拉流
消费器随之消失——设备其实还在向 ISS 推流, 识别却停了, 只能有人手工再开一次。
这里把「哪些摄像头应该在拉流」作为期望状态存入 Redis
(hash: person_id:stream_state, field=camera_id, value=JSON):

- consume/start 成功 → upsert 一条 {url, env, auto_restream}
- consume/stop → 删除该摄像头的条目(显式停止才是"不再期望拉流")
- 自动重推流拿到新地址 → 同步更新 url, 重启后直接连最新地址
- 服务启动 → 按 Redis 里的条目逐个重建 StreamConsumer; 地址已失效也无妨,
  拉流失败会走既有的 auto_restream 自愈(查设备在线 → ISS 重推 → 换新地址)

存 Redis 而非本地文件: data/ 目录只给 sqlite 用; Redis 是项目统一的运行态
存储(voice_server 同款实例), 换机部署/多实例时状态也不落在单机磁盘上。
连接参数在 config.redis(.env 注入), 未配置时本模块全部空操作(记警告),
Redis 故障也只影响"下次重启的恢复", 不阻断本次启停操作。

恢复时严格校验: url/env/auto_restream 缺失或非法(如 env 不是 test|prod)的
条目是脏数据——env 决定自动重推流打哪套 ISS, 用默认值猜错环境会把测试流
推到生产(或反之), 宁可报错放弃: 记 error 日志并从 Redis 删除该条目。

期望状态的唯一真实来源在本服务: 所有启停入口(web 控制台经 agent_server 代理、
person_id 自带前端、注册流程)最终都汇到 consume/start|stop。不放在
agent_server 管理——跨服务各存一份会脑裂, 且 agent_server 对 person_id 是
best-effort 可选依赖, 不应反向承担它的生命周期。

进程收尾(lifespan shutdown)停消费器不删条目: 关停≠用户想停止拉流,
正是重启恢复要保住的东西。
"""
from __future__ import annotations

import json
import time

import redis.asyncio as redis
from loguru import logger

from src.config import get_config

# Redis hash: field=camera_id, value=期望状态 JSON
STATE_KEY = "person_id:stream_state"

_VALID_ENVS = ("test", "prod")

_client: redis.Redis | None = None
_warned_unconfigured = False


def _get_client() -> redis.Redis | None:
    """懒建 Redis 连接池; 未配置(host 为空)返回 None 并只警告一次。"""
    global _client, _warned_unconfigured
    cfg = get_config().redis
    if not cfg.host:
        if not _warned_unconfigured:
            _warned_unconfigured = True
            logger.warning(
                "Redis 未配置(REDIS_HOST 为空), 拉流期望状态不持久化, "
                "服务重启后不会自动恢复拉流")
        return None
    if _client is None:
        # protocol=2 兼容老版本 Redis 服务端(与 voice_agent_common 口径一致)
        _client = redis.Redis(
            host=cfg.host,
            port=cfg.port,
            password=cfg.password or None,
            db=cfg.db,
            decode_responses=True,
            protocol=2,
            socket_connect_timeout=cfg.socket_connect_timeout,
            socket_timeout=cfg.socket_timeout,
        )
    return _client


async def close() -> None:
    """关闭连接池(lifespan shutdown 调用)。"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def record_desired(camera_id: str, url: str, env: str,
                         auto_restream: bool) -> None:
    """记录/更新一个摄像头的期望拉流状态(consume/start 成功后调用)。"""
    r = _get_client()
    if r is None:
        return
    entry = {
        "url": url,
        "env": env,
        "auto_restream": auto_restream,
        "updated_at": time.time(),
    }
    try:
        await r.hset(STATE_KEY, camera_id, json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        # 写失败只影响"下次重启的恢复", 不能反过来打断本次启停操作
        logger.warning("拉流期望状态写入 Redis 失败: camera={} ({})", camera_id, e)


async def remove_desired(camera_id: str) -> None:
    """移除期望拉流状态(consume/stop 时调用, 对不存在的条目幂等)。"""
    r = _get_client()
    if r is None:
        return
    try:
        await r.hdel(STATE_KEY, camera_id)
    except Exception as e:
        logger.warning("拉流期望状态删除失败: camera={} ({})", camera_id, e)


async def update_url(camera_id: str, url: str) -> None:
    """自动重推流换了直播地址后同步更新, 让重启恢复直接用新地址。

    条目不存在时不补建: 说明用户已显式停止(stop 与恢复协程存在竞争窗口),
    补建会让"已停止的流"在下次重启时复活。
    """
    r = _get_client()
    if r is None:
        return
    try:
        raw = await r.hget(STATE_KEY, camera_id)
        if raw is None:
            return
        entry = json.loads(raw)
        entry["url"] = url
        entry["updated_at"] = time.time()
        await r.hset(STATE_KEY, camera_id, json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        logger.warning("拉流期望状态更新 url 失败: camera={} ({})", camera_id, e)


def _parse_entry(camera_id: str, raw: str) -> tuple[str, str, bool] | None:
    """严格解析一条期望状态; 脏数据返回 None(由调用方删除)。

    不给任何默认值: env 决定自动重推流打哪套 ISS 环境, 猜错会把流推错环境;
    这类数据只由本服务自己写入, 出现缺失/非法即说明写入方有 bug 或被人手改过,
    宁可报错放弃, 不能带病恢复。
    """
    try:
        entry = json.loads(raw)
    except ValueError:
        logger.error("拉流期望状态脏数据(非 JSON), 放弃并删除: camera={} raw={!r}",
                     camera_id, raw)
        return None
    url = entry.get("url")
    env = entry.get("env")
    auto_restream = entry.get("auto_restream")
    if not url or not isinstance(url, str):
        logger.error("拉流期望状态脏数据(缺 url), 放弃并删除: camera={} entry={}",
                     camera_id, entry)
        return None
    if env not in _VALID_ENVS:
        logger.error("拉流期望状态脏数据(env 缺失或非法, 不使用默认值), "
                     "放弃并删除: camera={} env={!r}", camera_id, env)
        return None
    if not isinstance(auto_restream, bool):
        logger.error("拉流期望状态脏数据(缺 auto_restream), 放弃并删除: "
                     "camera={} entry={}", camera_id, entry)
        return None
    return url, env, auto_restream


async def restore_streams() -> None:
    """(启动期) 按 Redis 中的期望状态重建全部拉流消费器。

    单个摄像头恢复失败(如 orchestrator 初始化异常)只记日志不中断;
    脏数据条目报错并从 Redis 删除; Redis 不可达则本次放弃恢复(不影响启动)。
    """
    from src.api.registry import consumer_registry, get_or_create_orchestrator
    from src.pipeline.stream_consumer import StreamConsumer

    r = _get_client()
    if r is None:
        return
    try:
        state: dict[str, str] = await r.hgetall(STATE_KEY)
    except Exception as e:
        logger.error("拉流期望状态读取失败(Redis 不可达?), 本次不恢复拉流: {}", e)
        return
    if not state:
        return

    logger.info("按期望状态恢复拉流: {} 个摄像头", len(state))
    for camera_id, raw in state.items():
        parsed = _parse_entry(camera_id, raw)
        if parsed is None:
            try:
                await r.hdel(STATE_KEY, camera_id)
            except Exception as e:
                logger.warning("脏数据条目删除失败: camera={} ({})", camera_id, e)
            continue
        url, env, auto_restream = parsed
        try:
            orch = await get_or_create_orchestrator(camera_id)
            consumer = StreamConsumer(
                camera_id=camera_id,
                url=url,
                orchestrator=orch,
                env=env,
                auto_restream=auto_restream,
            )
            consumer.start()
            consumer_registry[camera_id] = consumer
            logger.info("拉流已恢复: camera={}, env={}, url={}", camera_id, env, url)
        except Exception:
            logger.exception("拉流恢复失败(跳过): camera={}", camera_id)
