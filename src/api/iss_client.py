"""
ISS 推流服务客户端

封装 ISS start_stream / stop_stream 两个接口的调用, 供两处复用:
- REST 路由 (routes.py): 前端手动开启/停止设备推流
- 拉流自动恢复 (pipeline/restream.py): 拉流连续失败后自动重推流

失败用结果类型表达 (ok=False + 可读 error), 不抛异常:
路由侧把 error 转成 HTTPException 透传给前端, 恢复侧把 error 记进重推日志。
"""
from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

import httpx
from voice_agent_common.utils.logger import logger

from src.configs.config import config

ISSEnv = Literal["test", "prod"]


@dataclass
class ISSCallResult:
    """一次 ISS 接口调用的结果。

    ok=False 时 error 为人类可读的失败描述(含 ISS 原始响应摘要),
    http_status 为建议回给前端的 HTTP 状态码。
    """
    ok: bool
    flv_url: str = ""       # start_stream 成功时的 FLV 直播地址
    error: str = ""
    http_status: int = 502


def _is_dns_failure(exc: BaseException) -> bool:
    """判断连接错误的根因是否为域名解析失败。

    httpx.ConnectError 同时覆盖 DNS 解析失败与连接被拒等情况,
    沿异常链找 socket.gaierror 才能区分出前者。
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, socket.gaierror):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _fail_detail(prefix: str, resp: httpx.Response) -> str:
    """拼上 ISS 原始响应, 让调用方能直接看到失败原因。"""
    body = (resp.text or "").strip()
    if len(body) > 300:
        body = body[:300] + "..."
    return f"{prefix} — ISS 响应 (HTTP {resp.status_code}): {body or '(空)'}"


async def _iss_post(action: str, camera_id: str, env: ISSEnv) -> httpx.Response | ISSCallResult:
    """调用 ISS 接口; 网络层失败时直接返回带可读 error 的 ISSCallResult。"""
    cfg = config
    base = cfg.iss_api_url(env).rstrip("/")
    headers = {"device-sn": camera_id}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            return await client.post(f"{base}/iss/{action}", headers=headers)
    except httpx.ConnectError as e:
        if _is_dns_failure(e):
            host = urlparse(base).hostname or base
            logger.error("ISS 域名解析失败: {} ({})", base, e)
            return ISSCallResult(
                ok=False,
                error=f"无法解析推流服务域名: {host} (ISS {env} 环境)，"
                      f"该域名可能尚未配置 DNS 或地址写错，请联系服务负责人",
            )
        logger.error("ISS 服务连接失败: {} ({})", base, e)
        return ISSCallResult(
            ok=False,
            error=f"无法连接推流服务 (ISS {env}: {base})，服务可能未启动或网络不通，请联系服务负责人",
        )
    except httpx.TimeoutException:
        logger.error("ISS 服务请求超时: {}", base)
        return ISSCallResult(
            ok=False,
            error=f"推流服务 (ISS {env}: {base}) 响应超时，请稍后重试",
            http_status=504,
        )
    except httpx.HTTPError as e:
        logger.error("ISS 请求异常: {} ({})", base, e)
        return ISSCallResult(ok=False, error=f"推流服务 (ISS) 请求异常: {e}")


async def iss_start_stream(camera_id: str, env: ISSEnv, source: str) -> ISSCallResult:
    """开启设备推流, 成功时返回 FLV 直播地址。

    camera_id 即设备 device-sn。ISS 失败时 (含 HTTP 500) body 里通常带有 msg,
    透传给调用方便于定位 (最常见: 设备号不存在 → "无法获取设备 DB ID")。
    source: 触发本次调用的入口(接口/自动恢复等), 只进日志, 用于追踪谁在开推流。
    """
    resp = await _iss_post("start_stream", camera_id, env)
    if isinstance(resp, ISSCallResult):
        return resp
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code != 200 or data.get("code") != 0:
        logger.warning(
            "ISS start_stream 失败: env={}, device-sn={}, source={}, HTTP {}, body={}",
            env, camera_id, source, resp.status_code, resp.text,
        )
        return ISSCallResult(
            ok=False,
            error=_fail_detail(
                f"设备推流开启失败 (device-sn={camera_id})，请确认设备号正确", resp,
            ),
        )

    flv_url = (data.get("data") or {}).get("Flv", "")
    if not flv_url:
        logger.warning(
            "ISS 未返回 FLV 地址: device-sn={}, body={}", camera_id, resp.text,
        )
        return ISSCallResult(
            ok=False,
            error=_fail_detail(f"ISS 未返回 FLV 地址 (device-sn={camera_id})", resp),
        )

    logger.info(
        "ISS 设备推流已启动: env={}, device-sn={}, source={}, flv={}",
        env, camera_id, source, flv_url,
    )
    return ISSCallResult(ok=True, flv_url=flv_url)


async def iss_stop_stream(camera_id: str, env: ISSEnv, source: str) -> ISSCallResult:
    """停止设备推流。

    source: 触发本次调用的入口(接口/租约到期/自动恢复等), 只进日志, 用于追踪谁在停推流。
    """
    resp = await _iss_post("stop_stream", camera_id, env)
    if isinstance(resp, ISSCallResult):
        return resp
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code != 200 or data.get("code") != 0:
        logger.warning(
            "ISS stop_stream 失败: env={}, device-sn={}, source={}, HTTP {}, body={}",
            env, camera_id, source, resp.status_code, resp.text,
        )
        return ISSCallResult(
            ok=False,
            error=_fail_detail(f"停止设备推流失败 (device-sn={camera_id})", resp),
        )

    logger.info(
        "ISS 设备推流已停止: env={}, device-sn={}, source={}", env, camera_id, source,
    )
    return ISSCallResult(ok=True)
