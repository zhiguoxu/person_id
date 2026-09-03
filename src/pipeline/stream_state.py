"""期望拉流状态持久化(Redis) — 服务重启后自动恢复拉流

consumer_registry 是纯内存状态: person_id 进程重启(部署/崩溃)后所有拉流
消费器随之消失——设备其实还在向 ISS 推流, 识别却停了, 只能有人手工再开一次。
这里把「哪些摄像头应该在拉流」作为期望状态存入 Redis
(hash: person_id:stream_state, field=camera_id, value=JSON):

- consume/start 成功 → upsert 一条 {url, env, auto_restream, lease_deadline}
- consume/stop → 删除该摄像头的条目(显式停止才是"不再期望拉流")
- 自动重推流拿到新地址 → 同步更新 url, 重启后直接连最新地址
- 服务启动 → 按 Redis 里的条目逐个重建 StreamConsumer; 地址已失效也无妨,
  拉流失败会走既有的 auto_restream 自愈(查设备在线 → ISS 重推 → 换新地址)

租约(lease_deadline, 可选):
调用方(voice_server 唤醒态联动)开流时可带租约时长; 到期未续租则看门狗
自动停消费并停设备推流。这是"摄像头不失控常开"的安全网——续租方
(voice_server)崩溃后, 摄像头最迟在租约到期时被关掉, 而不是永久开着。
续租走轻量的 consume/renew_lease(→ 本模块 renew_lease): 只刷新
lease_deadline, 不碰 ISS 推流和消费器——重复走完整的 device_stream/start
+ consume/start 续租时, ISS 可能换 FLV 地址, 触发"URL 变更先停旧再起新"
造成视频中断, 所以续租绝不能走完整开流。重复 consume/start(带
lease_seconds)也仍会续租(幂等分支), 作为语义兜底。
lease_deadline 缺失/None = 永久(web 控制台手动启停, 行为与租约引入前
一致)。租约随期望状态存 Redis: person_id 自己重启也不丢, restore_streams
恢复时会拒绝已过期的条目(顺带停设备推流), 不会复活"本该关掉"的流。

存 Redis 而非本地文件: data/ 目录只给 sqlite 用; Redis 是项目统一的运行态
存储(voice_server 同款实例), 换机部署/多实例时状态也不落在单机磁盘上。
连接是 common 统一的 RedisClient(deps.stream_state_redis_client, main.py
lifespan 建连/关闭; redis 是必配块, 连接必然存在)。Redis 运行中故障只影响
"下次重启的恢复"和租约记账, 不阻断本次启停操作。

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

import asyncio
import json
import time
from typing import TYPE_CHECKING

from voice_agent_common.utils.logger import logger

from src import deps
from src.api.registry import (
    camera_operation,
    get_or_create_orchestrator_unlocked,
    get_stream_consumer,
    maybe_release_orchestrator_unlocked,
    register_stream_consumer,
)

if TYPE_CHECKING:
    from src.pipeline.stream_consumer import StreamConsumer

# Redis hash: field=camera_id, value=期望状态 JSON
STATE_KEY = "person_id:stream_state"

_VALID_ENVS = ("test", "prod")


def _get_client():
    """本模块各函数的取连接入口。

    连接由 main.py lifespan 建在 deps 上(common RedisClient, redis 必配);
    这里取原始 redis 实例(.rds)做 hash 操作。
    """
    return deps.stream_state_redis_client.rds


async def record_desired(camera_id: str, url: str, env: str,
                         auto_restream: bool,
                         lease_deadline: float | None = None) -> None:
    """记录/更新一个摄像头的期望拉流状态(consume/start 成功后调用)。

    lease_deadline: 租约到期时刻(epoch 秒), None=永久。重复 start 走到这里
    即续租; 带租约与不带租约互相覆盖(最后一次 start 的意图生效)。
    """
    entry = {
        "url": url,
        "env": env,
        "auto_restream": auto_restream,
        "lease_deadline": lease_deadline,
        "updated_at": time.time(),
    }
    try:
        await _get_client().hset(
            STATE_KEY, camera_id, json.dumps(entry, ensure_ascii=False))
    except Exception as e:
        # 写失败只影响"下次重启的恢复", 不能反过来打断本次启停操作
        logger.warning("拉流期望状态写入 Redis 失败: camera={} ({})", camera_id, e)


async def renew_lease(camera_id: str, lease_seconds: float) -> bool:
    """续租: 仅刷新期望状态里的 lease_deadline, 不碰 ISS 推流和消费器。

    返回 False 表示没有可续的状态(条目不存在 = 已显式停止或从未记录,
    或 Redis 读写失败/条目损坏), 调用方应退回完整开流把状态补齐。
    对永久条目(控制台开的)续租会给它挂上租约——与 consume/start 一致的
    "最后一次意图生效"语义。
    """
    r = _get_client()
    try:
        raw = await r.hget(STATE_KEY, camera_id)
        if raw is None:
            return False
        entry = json.loads(raw)
        entry["lease_deadline"] = time.time() + lease_seconds
        entry["updated_at"] = time.time()
        await r.hset(STATE_KEY, camera_id, json.dumps(entry, ensure_ascii=False))
        return True
    except Exception as e:
        logger.warning("拉流租约续期失败: camera={} ({})", camera_id, e)
        return False


async def stop_consumption(camera_id: str, source: str) -> StreamConsumer | None:
    """停止拉流消费的公共步骤: 删期望状态 → 停消费器 → 回收 orchestrator。

    consume/stop(显式停止)与租约到期清理(_expire_lease)共用。期望状态
    无论消费器是否存在都先移除: 崩溃后条目可能残留而消费器已不在。
    返回被停掉的消费器(不存在时 None), 供路由拼状态响应。
    source: 触发停止的入口, 透传给 consumer.stop() 进日志。
    """
    async with camera_operation(camera_id):
        try:
            await _get_client().hdel(STATE_KEY, camera_id)
        except Exception as e:
            logger.warning("拉流期望状态删除失败: camera={} ({})", camera_id, e)

        consumer = get_stream_consumer(camera_id)
        if consumer is not None:
            await consumer.stop(source=source)

        # 必须在同一锁内调用 unlocked 版本, 否则会自锁死。
        await maybe_release_orchestrator_unlocked(camera_id)
        return consumer


async def update_url(camera_id: str, url: str) -> None:
    """自动重推流换了直播地址后同步更新, 让重启恢复直接用新地址。

    条目不存在时不补建: 说明用户已显式停止(stop 与恢复协程存在竞争窗口),
    补建会让"已停止的流"在下次重启时复活。
    """
    r = _get_client()
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


def _parse_entry(camera_id: str, raw: str) -> tuple[str, str, bool, float | None] | None:
    """严格解析一条期望状态; 脏数据返回 None(由调用方删除)。

    不给任何默认值: env 决定自动重推流打哪套 ISS 环境, 猜错会把流推错环境;
    这类数据只由本服务自己写入, 出现缺失/非法即说明写入方有 bug 或被人手改过,
    宁可报错放弃, 不能带病恢复。
    例外: lease_deadline 是后加字段, 缺失按 None(永久)处理——租约引入前
    写入的旧条目都没有它, 不能因此判为脏数据。
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
    lease_deadline = entry.get("lease_deadline")
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
    if lease_deadline is not None and not isinstance(lease_deadline, (int, float)):
        logger.error("拉流期望状态脏数据(lease_deadline 非数值), 放弃并删除: "
                     "camera={} entry={}", camera_id, entry)
        return None
    return url, env, auto_restream, lease_deadline


async def restore_streams() -> None:
    """(启动期) 按 Redis 中的期望状态重建全部拉流消费器。

    单个摄像头恢复失败(如 orchestrator 初始化异常)只记日志不中断;
    脏数据条目报错并从 Redis 删除; Redis 不可达则本次放弃恢复(不影响启动)。
    """
    from src.pipeline.stream_consumer import StreamConsumer

    r = _get_client()
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
        url, env, auto_restream, lease_deadline = parsed
        if lease_deadline is not None and lease_deadline <= time.time():
            # 停机期间租约已到期: 这条流"本该被关掉", 不能复活。设备可能还在
            # 推流(停机期间没人执行到期关流), 走完整的到期清理把推流也停掉
            logger.info("拉流租约在停机期间已过期, 不恢复并停设备推流: camera={}",
                        camera_id)
            await _expire_lease(camera_id, env, source="启动恢复(租约在停机期间已过期)")
            continue
        try:
            async with camera_operation(camera_id):
                existing = get_stream_consumer(camera_id)
                if existing is not None:
                    if existing.running and existing.url == url:
                        logger.info("拉流已存在, 跳过重复恢复: camera={}", camera_id)
                        continue
                    await existing.stop(source="启动恢复(URL 变更, 停旧起新)")

                orch = await get_or_create_orchestrator_unlocked(camera_id)
                consumer = StreamConsumer(
                    camera_id=camera_id,
                    url=url,
                    orchestrator=orch,
                    env=env,
                    auto_restream=auto_restream,
                )
                consumer.start(source="启动恢复(Redis 期望状态)")
                register_stream_consumer(camera_id, consumer)
                logger.info("拉流已恢复: camera={}, env={}, url={}", camera_id, env, url)
        except Exception:
            logger.exception("拉流恢复失败(跳过): camera={}", camera_id)


# ==============================================================================
# 租约看门狗: 周期扫描期望状态, 到期未续租的流自动停消费 + 停设备推流
# ==============================================================================

LEASE_CHECK_INTERVAL = 15.0  # 扫描周期(秒)。租约是安全网, 到期精度不敏感

_watchdog_task: "asyncio.Task | None" = None


async def _expire_lease(camera_id: str, env: str, source: str) -> None:
    """租约到期的完整清理: 停消费(同 consume/stop 的公共步骤) → 停设备推流。

    先停消费再停推流, 顺序不能反, 否则消费器对着死流反复重连甚至触发
    自动重推流复活。各步独立容错: 停推流失败只多耗带宽, 消费已停,
    安全网的主目的(识别管线不再消费)已达成。
    source: 触发清理的入口(看门狗/启动恢复), 透传给停消费与停推流的日志。
    """
    from src.api.iss_client import iss_stop_stream

    try:
        await stop_consumption(camera_id, source=source)
    except Exception:
        logger.exception("租约到期停消费失败: camera={}", camera_id)

    result = await iss_stop_stream(camera_id, env, source=source)
    if not result.ok:
        logger.warning("租约到期停设备推流失败: camera={} ({})", camera_id, result.error)


async def _enforce_leases_once() -> None:
    """扫描一轮期望状态, 清理到期未续租的流。"""
    r = _get_client()
    try:
        state: dict[str, str] = await r.hgetall(STATE_KEY)
    except Exception as e:
        logger.warning("租约扫描读取期望状态失败(Redis 不可达?): {}", e)
        return
    now = time.time()
    for camera_id, raw in state.items():
        parsed = _parse_entry(camera_id, raw)
        if parsed is None:
            # 脏数据交给下次重启的 restore_streams 删, 这里只管租约
            continue
        _url, env, _auto_restream, lease_deadline = parsed
        if lease_deadline is None or lease_deadline > now:
            continue
        # 复核一次再动手: hgetall 与此刻之间可能刚好有一次续租(hset), 直接
        # 按快照清理会把刚续上的流关掉。复核后仍有极窄的竞争窗口(复核到
        # 停流之间的续租), 但续租方(voice_server keeper)的周期性幂等重发
        # start 会在半个空闲窗口内把流拉起来, 不会长期失流
        try:
            raw2 = await r.hget(STATE_KEY, camera_id)
        except Exception:
            continue
        if raw2 is None:
            continue  # 期间已被显式停止
        parsed2 = _parse_entry(camera_id, raw2)
        if parsed2 is None or parsed2[3] is None or parsed2[3] > now:
            continue
        logger.info("拉流租约到期未续租, 自动关流: camera={} (deadline 已过 {:.0f}s)",
                    camera_id, now - parsed2[3])
        await _expire_lease(camera_id, parsed2[1], source="租约看门狗(到期未续租)")


async def _watchdog_loop() -> None:
    while True:
        await asyncio.sleep(LEASE_CHECK_INTERVAL)
        try:
            await _enforce_leases_once()
        except Exception:
            # 看门狗是安全网, 单轮异常(Redis 抖动等)不能杀死循环
            logger.exception("租约看门狗单轮执行异常(继续运行)")


def start_lease_watchdog() -> None:
    """启动租约看门狗后台任务(lifespan startup 调用)。"""
    global _watchdog_task
    _watchdog_task = asyncio.create_task(_watchdog_loop())


async def stop_lease_watchdog() -> None:
    """停止租约看门狗(lifespan shutdown 调用)。"""
    global _watchdog_task
    if _watchdog_task is not None:
        _watchdog_task.cancel()
        try:
            await _watchdog_task
        except asyncio.CancelledError:
            pass
        _watchdog_task = None
