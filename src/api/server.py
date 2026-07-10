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
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.registry import camera_registry
from src.api.routes import router as api_router
from src.api.websocket import handle_ws_connection
from src.config import FRONTEND_DIR, load_config


# ==============================================================================
# Lifespan
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期。"""
    # --- startup ---
    from src.config import get_config
    from src.gallery.persistence import get_gallery_persistence

    persistence = get_gallery_persistence()
    await persistence.initialize(get_config().server.gallery_db_path)

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

    await get_gallery_persistence().close()
    logger.info("应用关闭完成")


# ==============================================================================
# App factory
# ==============================================================================

def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="Person ID — Robot Vision System",
        description="实时多摄像头人物识别与追踪系统 API",
        version="0.2.0",
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

    # --- Static files (frontend) ---
    frontend_path = Path(FRONTEND_DIR)
    if frontend_path.exists() and frontend_path.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )
        logger.info("已从 {} 挂载前端", frontend_path)
    else:
        logger.warning(
            "未找到前端目录: {}", frontend_path
        )

    return app


def main() -> None:
    """直接运行时的入口点。在远程 CUDA 服务器上运行。"""
    import asyncio

    config = load_config()

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
