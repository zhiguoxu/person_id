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
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.routes import router as api_router
from src.api.websocket import handle_ws_connection
from src.config import FRONTEND_DIR, load_config
from src.pipeline.orchestrator import VisionOrchestrator

# 全局摄像头注册表: camera_id → VisionOrchestrator
camera_registry: dict[str, VisionOrchestrator] = {}


def get_camera_orchestrator(camera_id: str) -> VisionOrchestrator | None:
    """获取指定摄像头的编排器（供 REST routes 使用）。"""
    return camera_registry.get(camera_id)


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

    logger.info("Application ready (cameras will initialize on first connection)")

    yield  # ← 应用运行中

    # --- shutdown ---
    logger.info("Application shutting down ({} cameras) ...", len(camera_registry))
    for cam_id, orch in camera_registry.items():
        logger.info("Shutting down camera: {}", cam_id)
        await orch.shutdown()
    camera_registry.clear()

    await get_gallery_persistence().close()
    logger.info("Application shutdown complete")


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
        await handle_ws_connection(websocket, camera_id, camera_registry)

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

    app = create_app()

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        ws_max_size=config.server.ws_max_frame_size,
    )


if __name__ == "__main__":
    main()
