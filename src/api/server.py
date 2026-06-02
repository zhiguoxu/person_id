"""
FastAPI 应用入口 — 服务器初始化与生命周期管理

启动 FastAPI 应用:
- CORS 中间件
- 静态文件 (frontend/)
- REST 路由
- WebSocket 端点
- 应用启动/关闭事件 (模型加载、底库加载/保存)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.routes import router as api_router
from src.api.routes import set_orchestrator
from src.api.websocket import VisionWebSocket
from src.config import FRONTEND_DIR, Config, load_config
from src.pipeline.orchestrator import VisionOrchestrator

# Module-level references (set during lifespan)
_orchestrator: VisionOrchestrator | None = None
_ws_handler: VisionWebSocket | None = None


# ==============================================================================
# Lifespan
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期: 启动时初始化, 关闭时清理。"""
    global _orchestrator, _ws_handler

    config = app.state.config
    logger.info("Application starting ...")

    # 初始化编排器
    _orchestrator = VisionOrchestrator(config)
    await _orchestrator.initialize()

    # 注入到路由模块
    set_orchestrator(_orchestrator)

    # 初始化 WebSocket 处理器
    _ws_handler = VisionWebSocket(_orchestrator, config)
    app.state.ws_handler = _ws_handler

    logger.info("Application ready")

    yield  # ← 应用运行中

    # 关闭
    logger.info("Application shutting down ...")
    if _orchestrator:
        await _orchestrator.shutdown()
    logger.info("Application shutdown complete")


# ==============================================================================
# App factory
# ==============================================================================

def create_app(config: Config | None = None) -> FastAPI:
    """
    创建并配置 FastAPI 应用。

    Args:
        config: 全局配置。如果为 None 则从环境变量加载。

    Returns:
        配置完毕的 FastAPI 应用实例。
    """
    if config is None:
        config = load_config()

    app = FastAPI(
        title="Person ID — Robot Vision System",
        description="实时人物识别与追踪系统 API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 保存配置到 app state
    app.state.config = config

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- REST routes ---
    app.include_router(api_router)

    # --- WebSocket endpoint ---
    @app.websocket("/ws/vision")
    async def ws_vision(websocket: WebSocket) -> None:
        """主 WebSocket 端点: 接收视频帧, 返回识别结果。"""
        if _ws_handler is None:
            await websocket.close(code=1013, reason="Server not ready")
            return
        await _ws_handler.handle_connection(websocket)

    # --- Static files (frontend) ---
    frontend_path = Path(FRONTEND_DIR)
    if frontend_path.exists() and frontend_path.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(frontend_path), html=True),
            name="frontend",
        )
        logger.info("Frontend mounted from {}", frontend_path)
    else:
        logger.warning(
            "Frontend directory not found: {}", frontend_path
        )

    return app


def main() -> None:
    """直接运行时的入口点。在远程 CUDA 服务器上运行。"""
    config = load_config()

    logger.info(
        "Starting server on {}:{}",
        config.server.host, config.server.port,
    )
    logger.info(
        "Frontend runs locally, connects via WebSocket to this server"
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        ws_max_size=config.server.ws_max_frame_size,
    )


if __name__ == "__main__":
    main()
