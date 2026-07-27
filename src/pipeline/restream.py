"""
拉流自动恢复 — 连续拉流失败后自动重新开启设备推流

背景: ISS 设备推流偶发损坏后, 服务端拉流(StreamConsumer)会一直重连失败,
无法自愈, 只能人工"重新推流 + 重新拉流"。本模块把该过程自动化:

1. StreamConsumer 连续拉流失败达到阈值 (server.stream_restream_fail_threshold,
   web 控制台可调) 时调用 run_restream_recovery()
2. 先经 voice_server 查设备是否在线 (设备关机后无法推流, 不在线则跳过重推)
3. 在线则调 ISS stop_stream + start_stream 重新推流, 拿到新 FLV 地址
4. StreamConsumer 切换到新地址继续拉流; 新地址经 consume/status 同步到前端

每次恢复尝试的完整过程(触发原因、在线检查、ISS 调用结果、每一步的错误)
都追加写入 data/restream_log/{camera_id}.jsonl, 前端可通过
GET /api/{camera_id}/device_stream/restream_log 查询。
"""
from __future__ import annotations

import time

import httpx
from loguru import logger
from pydantic import ValidationError

from src.api.iss_client import ISSEnv, iss_start_stream, iss_stop_stream
from src.config import DATA_DIR, get_config
from src.pipeline.restream_models import RestreamAttempt, RestreamLogLine

RESTREAM_LOG_DIR = DATA_DIR / "restream_log"


async def check_device_online(device_sn: str) -> tuple[bool | None, str]:
    """经 voice_server 查设备是否在线。

    设备与 voice_server 之间保持 WebSocket 长连接, 有活跃会话即设备开机在网。
    返回 (在线状态, 描述); voice_server 不可达/响应异常时状态为 None,
    由调用方决定如何降级。
    """
    base = get_config().server.voice_server_api_url.rstrip("/")
    url = f"{base}/api/device/{device_sn}/online"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return None, f"voice_server 在线检查请求失败: {url} ({e})"
    if resp.status_code != 200:
        return None, f"voice_server 在线检查异常: HTTP {resp.status_code}, body={resp.text[:200]}"
    try:
        data = resp.json()
        online = bool(data["online"])
    except Exception as e:
        return None, f"voice_server 在线检查响应无法解析: {resp.text[:200]} ({e})"
    return online, "设备在线" if online else "设备不在线 (无活跃 WebSocket 会话, 可能已关机/断网)"


async def run_restream_recovery(
    camera_id: str,
    env: ISSEnv,
    current_url: str,
    fail_count: int,
    last_error: str,
) -> RestreamAttempt:
    """执行一次自动重推流恢复, 返回完整的尝试记录 (已落盘)。

    outcome 语义:
    - restreamed:      重推成功, new_url 为新 FLV 地址, 消费器应切换
    - device_offline:  设备不在线, 按约定不重推 (等设备开机后下个周期再试)
    - iss_start_failed: ISS 重新开启推流失败
    - error:           其他意外错误
    """
    started_at = time.time()
    logs: list[RestreamLogLine] = []

    def add_log(level: str, message: str) -> None:
        logs.append(RestreamLogLine(time=time.time(), level=level, message=message))
        log_fn = {"info": logger.info, "warning": logger.warning}.get(level, logger.error)
        log_fn("[拉流自动恢复] camera={}: {}", camera_id, message)

    add_log("warning",
            f"连续拉流失败 {fail_count} 次 (最后错误: {last_error or '无'}), "
            f"触发自动重推流 (env={env}, 当前地址: {current_url})")

    device_online: bool | None = None
    outcome = "error"
    new_url = ""
    try:
        # 1. 设备在线检查 (设备关机后推流指令送不到, 不在线就不重推)
        device_online, online_msg = await check_device_online(camera_id)
        if device_online is None:
            # 在线状态查不到 (voice_server 不可达等): 继续尝试重推 —— 宁可多试
            # 一次 ISS, 也不要因为旁路检查故障卡死恢复流程
            add_log("warning", f"{online_msg}; 无法确认设备状态, 仍继续尝试重推流")
        elif not device_online:
            add_log("error", f"{online_msg}; 跳过重推流, 等设备上线后再试")
            outcome = "device_offline"
            return _finish(camera_id, env, started_at, fail_count, last_error,
                           device_online, outcome, current_url, new_url, logs)
        else:
            add_log("info", online_msg)

        # 2. 先停旧推流 (清掉 ISS 侧可能损坏的流会话; 失败不阻断, 只记录)
        stop_result = await iss_stop_stream(camera_id, env)
        if stop_result.ok:
            add_log("info", "ISS stop_stream 成功 (旧推流已清理)")
        else:
            add_log("warning", f"ISS stop_stream 失败 (不阻断重推): {stop_result.error}")

        # 3. 重新开启推流
        start_result = await iss_start_stream(camera_id, env)
        if not start_result.ok:
            add_log("error", f"ISS start_stream 失败: {start_result.error}")
            outcome = "iss_start_failed"
            return _finish(camera_id, env, started_at, fail_count, last_error,
                           device_online, outcome, current_url, new_url, logs)

        new_url = start_result.flv_url
        outcome = "restreamed"
        if new_url == current_url:
            add_log("info", f"重推流成功, ISS 返回的地址与之前相同: {new_url}")
        else:
            add_log("info", f"重推流成功, 新直播地址: {new_url}")
        return _finish(camera_id, env, started_at, fail_count, last_error,
                       device_online, outcome, current_url, new_url, logs)
    except Exception as e:
        logger.exception("[拉流自动恢复] camera={} 意外失败", camera_id)
        add_log("error", f"恢复流程意外失败: {e}")
        return _finish(camera_id, env, started_at, fail_count, last_error,
                       device_online, "error", current_url, new_url, logs)


def _finish(
    camera_id: str,
    env: str,
    started_at: float,
    fail_count: int,
    last_error: str,
    device_online: bool | None,
    outcome: str,
    old_url: str,
    new_url: str,
    logs: list[RestreamLogLine],
) -> RestreamAttempt:
    """组装尝试记录并落盘。"""
    attempt = RestreamAttempt(
        camera_id=camera_id,
        env=env,
        started_at=started_at,
        finished_at=time.time(),
        trigger_fail_count=fail_count,
        trigger_error=last_error,
        device_online=device_online,
        outcome=outcome,
        old_url=old_url,
        new_url=new_url,
        logs=logs,
    )
    try:
        RESTREAM_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = RESTREAM_LOG_DIR / f"{camera_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(attempt.model_dump_json() + "\n")
    except Exception:
        # 落盘失败不影响恢复流程本身 (记录尽力而为)
        logger.exception("重推日志落盘失败: camera={}", camera_id)
    return attempt


def load_restream_attempts(camera_id: str, limit: int) -> list[RestreamAttempt]:
    """读取该设备最近的重推尝试记录, 新的在前。

    文件按次追加、事件低频, 直接全量读; 单行损坏跳过不影响其余记录。
    """
    path = RESTREAM_LOG_DIR / f"{camera_id}.jsonl"
    if not path.exists():
        return []
    attempts: list[RestreamAttempt] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                attempts.append(RestreamAttempt.model_validate_json(line))
            except ValidationError:
                continue
    attempts.reverse()
    return attempts[:limit]
