import asyncio

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.configs.config import config
from src.configs.override import config_override_manager
from voice_agent_common.config_override_api import build_config_override_router
from voice_agent_common.utils.logger import setup_service_logging
from voice_agent_common.utils.log_stream import log_broadcaster, make_stream_forwarder
from voice_agent_common.infra.redis_client import RedisClient
from session_store.database import init_db, close_db
from session_store import VideoStore
from src import deps

# logger patch 会给每条日志注入 {extra[device_sn]}(= camera_id)/{extra[trace_id]}，
# 由 WS/REST/拉流入口写入 ContextVar（websocket.py / routes.py / stream_consumer.py）。
logger = setup_service_logging(
    log_dir="logs/person",
    level=config.log_level,
    source="person",
    service="person_id",
    port=config.port,
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup

    # 绑定事件循环，使日志广播器可转发本进程日志
    log_broadcaster.bind_loop(asyncio.get_running_loop())

    from voice_agent_common.utils.clock import now_cst
    deps.started_at = now_cst()

    import voice_agent_common as _common
    import session_store as _session_store
    from voice_agent_common.utils.config_dump import format_config_for_log
    logger.info(
        "👁️ person_id 启动，版本 v{}（common v{}，session_store v{}），监听 {}:{}",
        config.version, _common.__version__, _session_store.__version__,
        config.host, config.port,
    )

    # 本进程日志 XADD 到 Redis Stream, 由 console_server 统一消费入库/推前端。
    # 日志走独立连接(独立 db, 与 voice/agent/console 配置一致); set_forwarder 会
    # 补发 forwarder 就绪之前的启动期日志(uvicorn 生命周期等)。
    deps.log_redis_client = RedisClient(
        config.redis.model_copy(update={"db": config.log_stream.db}))
    await deps.log_redis_client.connect()
    log_broadcaster.set_forwarder(make_stream_forwarder(
        deps.log_redis_client, config.log_stream.stream_key, config.log_stream.maxlen))

    # 配置覆盖层的 DB 引擎 (与 voice/agent 同款 session_store, 连接由 db_url 决定)
    await init_db(config.db_url)

    # 进程崩溃/重启后残留的 recording 行本地文件已丢, 标 failed 避免永久挂起
    orphaned = await VideoStore.mark_orphaned_recordings_failed()
    if orphaned:
        logger.info("启动清理: {} 条未完成录像标为 failed", orphaned)

    # 拉流录像上传 COS (密钥在环境 yaml; 与 voice_server 同桶)
    from voice_agent_common.infra.oss.cos import CosClient
    deps.cos_client = CosClient(config.cos)

    # 把 web 控制台在线改过的配置(DB 覆盖层)套到内存配置单例上。
    # 需在底库初始化/模型预热/拉流恢复等后续组件消费 config 之前执行。
    await config_override_manager.load_and_apply()
    # 配置转储放在覆盖套用之后, 日志里看到的才是真正生效的值
    logger.info("person_id 配置:\n{}", format_config_for_log(config))

    # 业务 Redis 连接(统一走 common RedisClient, 与日志连接同款; redis 是
    # 必配块, 无降级路径):
    #   - 拉流期望状态/租约在主库(config.redis.db);
    #   - voice_server 在线标记在 voice_online_redis_db(重推流前的在线检查)。
    # 放在覆盖套用之后建连, 在线改过的 redis 配置才生效
    deps.stream_state_redis_client = RedisClient(config.redis)
    await deps.stream_state_redis_client.connect()
    deps.voice_online_redis_client = RedisClient(
        config.redis.model_copy(update={"db": config.voice_online_redis_db}))
    await deps.voice_online_redis_client.connect()

    from src.gallery.persistence import get_gallery_persistence
    persistence = get_gallery_persistence()
    await persistence.initialize(config.gallery_db_path)

    # enroll 质量评估模型(默认 large)只有注册路径用到, 懒加载会让进程启动后的
    # 首次注册当场付 ONNX + CUDA EP 初始化(实测 ~1.2s), 几乎顶到对话端 1.5s
    # 超时; 启动期加载并跑一次 dummy 推理, 把冷启动成本从用户请求里挪走。
    import time
    from src.tier2.features.ediffiqa import get_ediffiqa_enroll

    t0 = time.perf_counter()
    get_ediffiqa_enroll().warmup()
    logger.info("enroll 质量评估模型预热完成 ({:.0f}ms)",
                (time.perf_counter() - t0) * 1000)

    # 声纹模型(第五识别模态): 启动期加载 + 预热, 同 ediffiqa 的理由——
    # 首次请求付 CUDA EP 初始化会顶到对话端 1.5s 超时
    from src.voice.embedder import get_voice_embed_extractor
    get_voice_embed_extractor().warmup()

    # 恢复重启前的拉流: 期望状态(哪些摄像头该在拉流)在 consume/start|stop 时
    # 写入 Redis, 这里按它重建消费器——部署/崩溃重启后拉流自动续上, 不用手工
    # 再开。存的地址若已失效, 拉流失败会走 auto_restream 自愈换新地址。
    from src.pipeline.stream_state import restore_streams, start_lease_watchdog
    await restore_streams()

    # 租约看门狗: 带租约开启的拉流(voice_server 唤醒态联动)到期未续租时
    # 自动停消费并停设备推流——续租方崩溃后摄像头不会失控常开
    start_lease_watchdog()

    logger.info("应用已就绪 (摄像头将在首次连接时初始化)")

    yield

    # Shutdown
    from src.api.registry import camera_registry, consumer_registry
    from src.pipeline import stream_state

    logger.info(
        "应用正在关闭 ({} 个摄像头, {} 个拉流消费器) ...",
        len(camera_registry), len(consumer_registry),
    )
    # 看门狗先停: 避免它在下面停消费器/清注册表的过程中并发执行到期清理
    await stream_state.stop_lease_watchdog()
    for cam_id, consumer in list(consumer_registry.items()):
        logger.info("正在停止拉流消费器: {}", cam_id)
        await consumer.stop()
    consumer_registry.clear()

    for cam_id, orch in camera_registry.items():
        logger.info("正在关闭摄像头: {}", cam_id)
        await orch.shutdown()
    camera_registry.clear()

    # 注意: 不删 Redis 里的拉流期望状态——关停≠用户想停止拉流,
    # 正是下次启动 restore_streams 恢复的依据(带租约的条目由重启后的
    # restore_streams/看门狗按到期时刻裁决)
    await deps.stream_state_redis_client.disconnect()
    await deps.voice_online_redis_client.disconnect()

    await get_gallery_persistence().close()
    await close_db()
    logger.info("Person ID server 已停止")
    # 日志连接最后关: 之前的关闭日志还要经它转发到聚合 Stream
    await deps.log_redis_client.disconnect()


app = FastAPI(title="Person ID — Robot Vision System", lifespan=lifespan)

# 未处理异常打详细日志(方法/路径/参数/traceback 进 loguru 管道)并回带根因的
# 500 JSON——否则只有 uvicorn 的 "Exception in ASGI application", 排障靠猜
from voice_agent_common.utils.unhandled_errors import install_unhandled_exception_logger

install_unhandled_exception_logger(app)

# CORS（前端开发服务器）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
from src.api.routes import router as api_router
from src.api.websocket import ws_router
from src.api.video_api import router as video_router

app.include_router(api_router)
app.include_router(ws_router)
app.include_router(video_router)

# 声纹 API (第五识别模态, voice_server 每轮调用)
from src.api.voice import router as voice_router
# 运行配置查询: GET /api/config (web 控制台「系统配置」页展示, 已脱敏)
from src.api.config_api import config_router

app.include_router(voice_router)
app.include_router(config_router)
# 配置在线编辑端点: GET/PUT/DELETE /api/config/editable[/{key_path}]
# (web 控制台经 /person_id 代理以 /person_id/api/config/editable 访问, 编辑需口令;
# person_id 无设备级覆盖, 不传 device_name_resolver)
app.include_router(build_config_override_router(
    config_override_manager, prefix="/api/config"))

if __name__ == "__main__":
    # 手动构建 Server，避免 uvicorn.run() 内部的 asyncio_run 与 PyCharm 调试器冲突
    uvi_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_config=None,
        loop="asyncio",
        # 浏览器抓帧上传的单条 WS 消息可达数 MB, 默认 1MB 上限会掐断连接
        ws_max_size=config.ws_max_frame_size,
        # current_identity 在对话首字延迟的关键路径上, 闲置连接保得久一点,
        # 配合 agent_server 侧 60s 保活 ping, 避免每轮对话重付 TCP 握手。
        # (uvicorn 默认 5s, 而对话轮距几乎总超 5s。)
        timeout_keep_alive=300,
    )
    server = uvicorn.Server(uvi_config)
    asyncio.run(server.serve())
