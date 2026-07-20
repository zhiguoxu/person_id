"""声音 embedding REST API — voice_server 每轮对话调用(内网, best-effort)

- POST /api/voice/embed — 本轮净语音 → 512 维说话人向量 + 净语音秒数
- GET /api/voice/status — 服务可用性(调试/巡检)

person_id 在声纹链路中只做"信号→向量"(感知), 比对/注册等身份决策在 agent_server。

音频格式: 请求体为裸 int16 PCM(16kHz 单声道), Content-Type: application/octet-stream。
选裸字节而非 base64/multipart: 每轮 100-200KB, 内网一跳, 零编解码开销。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from src.voice.embedder import get_voice_embed_extractor

router = APIRouter(prefix="/api/voice", tags=["voice"])

# 单轮语音上限: 60s @ 16kHz int16 ≈ 1.9MB, 超过按异常拒收(防误传整段录音)
_MAX_PCM_BYTES = 60 * 16000 * 2


@router.post("/embed")
async def voice_embed(request: Request) -> dict:
    """本轮音频 → 说话人向量。推理在线程池执行, 不占事件循环。"""
    pcm = await request.body()
    if not pcm:
        raise HTTPException(status_code=400, detail="请求体为空(需 int16 PCM 16kHz)")
    if len(pcm) > _MAX_PCM_BYTES:
        raise HTTPException(status_code=413, detail=f"音频过长: {len(pcm)} bytes")
    svc = get_voice_embed_extractor()
    if not svc.available:
        raise HTTPException(status_code=503, detail="声音 embedding 服务未就绪")
    result = await asyncio.to_thread(svc.embed, pcm)
    if result is None:
        raise HTTPException(status_code=500, detail="embedding 提取失败")
    embedding, net_speech_sec = result
    return {"embedding": embedding, "net_speech_sec": net_speech_sec, "dim": len(embedding)}


@router.get("/status")
async def voice_status() -> dict:
    svc = get_voice_embed_extractor()
    return {"available": svc.available, "dim": svc.dim}
