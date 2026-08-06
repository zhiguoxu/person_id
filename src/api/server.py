"""
FastAPI 应用入口 — 服务器初始化与生命周期管理

多摄像头架构:
- 每个 WebSocket 连接对应一个独立的 VisionOrchestrator
- GPU 模型通过 cache 全局共享，不重复加载
- Gallery 按 camera_id 隔离存储 (同一 SQLite, 不同 camera_id)
- REST API 通过 camera_id 路径参数访问指定摄像头
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src import deps
from src.api.registry import camera_registry
from src.api.routes import router as api_router
from src.api.websocket import handle_ws_connection
from src.configs.config import config
from src.configs.override import config_override_manager
from src.utils.log_setup import setup_logging

# configs.override 的 import 链会触发 voice_agent_common 的公共日志 sink
# (格式依赖 person_id 没有的 extra 字段), 必须在 import 完成后重建自己的 sink
setup_logging(config.server.log_level)


# ==============================================================================
# Lifespan
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期。"""
    # --- startup ---
    from voice_agent_common.utils.clock import now_cst
    deps.started_at = now_cst()

    import voice_agent_common as _common
    from voice_agent_common.utils.config_dump import format_config_for_log
    logger.info("👁️ person_id 启动，版本 v{}（common v{}）",
                config.version, _common.__version__)

    # 把 web 控制台在线改过的配置(DB 覆盖层)套到内存配置单例上。
    # 需在底库初始化/模型预热/拉流恢复等后续组件消费 config 之前执行。
    # 存储与 voice/agent 同款 session_store, 连接由 config.db_url 决定
    from session_store.database import init_db
    await init_db(config.db_url)
    await config_override_manager.load_and_apply()
    # 配置转储放在覆盖套用之后, 日志里看到的才是真正生效的值
    logger.info("person_id 配置:\n{}", format_config_for_log(config))

    from src.gallery.persistence import get_gallery_persistence

    persistence = get_gallery_persistence()
    await persistence.initialize(config.server.gallery_db_path)

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
    from src.pipeline.stream_state import restore_streams
    await restore_streams()

    logger.info("应用已就绪 (摄像头将在首次连接时初始化)")

    yield  # ← 应用运行中

    # --- shutdown ---
    from src.api.registry import consumer_registry

    logger.info(
        "应用正在关闭 ({} 个摄像头, {} 个拉流消费器) ...",
        len(camera_registry), len(consumer_registry),
    )
    for cam_id, consumer in list(consumer_registry.items()):
        logger.info("正在停止拉流消费器: {}", cam_id)
        await consumer.stop()
    consumer_registry.clear()

    for cam_id, orch in camera_registry.items():
        logger.info("正在关闭摄像头: {}", cam_id)
        await orch.shutdown()
    camera_registry.clear()

    # 注意: 不删 Redis 里的拉流期望状态——关停≠用户想停止拉流,
    # 正是下次启动 restore_streams 恢复的依据
    from src.pipeline import restream, stream_state
    await stream_state.close()
    await restream.close()

    await get_gallery_persistence().close()

    # 配置覆盖层的 DB 引擎 (session_store, 启动时 init_db 创建) 最后关
    from session_store.database import close_db
    await close_db()
    logger.info("应用关闭完成")


# ==============================================================================
# App factory
# ==============================================================================

def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="Person ID — Robot Vision System",
        description="实时多摄像头人物识别与追踪系统 API",
        version=config.version,
        lifespan=lifespan,
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 全局兜底异常处理 ---
    # 未捕获异常默认返回纯文本 "Internal Server Error", 前端 toast 拿不到原因;
    # 这里统一转成 JSON detail, 让所有错误都能展示在前端。
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
        logger.exception("未捕获异常: {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"服务器内部错误: {type(exc).__name__}: {exc}"},
            # 该响应在 CORSMiddleware 外层生成, 需自带 CORS 头
            headers={"Access-Control-Allow-Origin": "*"},
        )

    # --- REST routes ---
    app.include_router(api_router)

    # 声纹 API (第五识别模态, voice_server 每轮调用)
    from src.api.voice import router as voice_router
    app.include_router(voice_router)

    # 全量脱敏配置 dump: GET /api/config (控制台「系统配置」页展示)
    from src.api.config_api import config_router
    app.include_router(config_router)

    # 配置在线编辑端点: GET/PUT/DELETE /api/config/editable[/{key_path}]
    # (web 控制台经 /vision 代理以 /vision/api/config/editable 访问, 编辑需口令)
    from voice_agent_common.config_override_api import build_config_override_router
    app.include_router(build_config_override_router(
        config_override_manager, prefix="/api/config"))

    # --- WebSocket endpoint ---
    @app.websocket("/ws/vision")
    async def ws_vision(
        websocket: WebSocket,
        camera_id: str = Query(),
    ) -> None:
        """WebSocket 端点: 每个连接绑定一个摄像头。

        连接方式: ws://host:port/ws/vision?camera_id=cam_01
        """
        await handle_ws_connection(websocket, camera_id)

    # 前端统一在 web 控制台的 vision 页 (web/src/vision, 经 /vision 代理访问本服务);
    # 旧的自带静态前端已下线 (备份见 frontend_backup_20260806.tar.gz), 不再挂载静态目录
    return app


def main() -> None:
    """直接运行时的入口点。在远程 CUDA 服务器上运行。"""
    import asyncio

    logger.info(
        "服务器启动于 {}:{}",
        config.server.host, config.server.port,
    )

    app = create_app()

    # 直接调用 asyncio.run(server.serve())，
    # 绕过 uvicorn.Server.run() 中传递 loop_factory 的逻辑，
    # 避免 PyCharm 调试器 patch asyncio.run() 导致的不兼容
    uv_config = uvicorn.Config(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        ws_max_size=config.server.ws_max_frame_size,
        # current_identity 在对话首字延迟的关键路径上, 闲置连接保得久一点,
        # 配合 agent_server 侧 60s 保活 ping, 避免每轮对话重付 TCP 握手。
        # (uvicorn 默认 5s, 而对话轮距几乎总超 5s。)
        timeout_keep_alive=300,
    )
    server = uvicorn.Server(uv_config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
