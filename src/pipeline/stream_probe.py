"""拉流探测(ffprobe) — 开流前探分辨率, 失败时给出能直接定位的原因

ffprobe 对各类故障的表现不同, 原始报错要翻译一下才好排查:
- 一直挂到超时: 直播服务器已连上但不吐数据。FLV 直播源在没有推流端时就是
  这样(服务器 hold 住 HTTP 连接等推流), 所以超时几乎等价于"设备未推流 /
  推流刚被停掉"; 少数情况是 TCP 层被防火墙丢包
- 404 / 403 / 5XX / 连接拒绝 / DNS 失败 / 返回非媒体内容: 秒级返回, 原因在
  stderr 里(DNS 失败等原因在倒数第二行, 最后一行只是笼统的 Input/output
  error, 所以匹配整段 stderr 而不是只看最后一行)

reason 字符串 = 中文提示 + ffprobe 原文, 同时进日志和 consume/status 的
last_error, 前端也能看到。
"""
from __future__ import annotations

import subprocess
import time

# ffprobe 等待上限。直播源不吐数据时 ffprobe 不会自己退出, 只能靠这个
# 超时兜底; 正常可达的流探测在 1-4s 内返回
DEFAULT_PROBE_TIMEOUT = 10.0

# stderr 只展示尾部这么多行: ffprobe -v error 下报错通常 1-2 行
_STDERR_TAIL_LINES = 3

# (stderr 里的特征串, 中文解释), 按顺序首个命中生效。特征串按 ffmpeg 8.x
# 实测输出写, 老版本措辞相同。HTTP 状态码必须带 "Server returned" 前缀:
# stderr 会回显 URL, 裸 "404"/"403" 会误命中 token/UUID 里的数字
_STDERR_HINTS: tuple[tuple[str, str], ...] = (
    ("Server returned 404", "直播服务器上没有这路流(设备未推流或推流已停)"),
    ("Server returned 403", "直播服务器拒绝鉴权(token 无效或已过期, 需重新 ISS start 换地址)"),
    ("Server returned 401", "直播服务器拒绝鉴权(token 无效或已过期, 需重新 ISS start 换地址)"),
    ("Server returned 5XX", "直播服务器内部错误"),
    ("Server returned 4", "直播服务器拒绝请求(4XX)"),
    ("Failed to resolve hostname", "直播服务器域名解析失败"),
    ("Connection refused", "直播服务器拒绝连接(端口未监听或地址错)"),
    ("Connection timed out", "连不上直播服务器(网络不通)"),
    ("Invalid data found", "连上了但返回的不是可识别的媒体流(可能是错误页或空响应)"),
    ("End of file", "直播服务器连上后立刻断开(流刚结束或尚未开始)"),
)


class StreamProbeError(Exception):
    """探测失败。str(e) 即可直接进日志的原因描述。"""


def probe_resolution(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT) -> tuple[int, int]:
    """ffprobe 探测首路视频流的分辨率。

    流不可达/无效抛 StreamProbeError(带翻译后的原因); ffprobe 本身不存在等
    本机环境问题原样抛 OSError, 由调用方决定是否退回其他解码器。
    """
    t0 = time.monotonic()
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                url,
            ],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise StreamProbeError(
            f"ffprobe {timeout:g}s 内无响应: 多为直播服务器已连上但没有推流端"
            f"(设备未推流或推流已被停掉), 少数为网络不通"
        ) from None
    elapsed = time.monotonic() - t0
    text = out.stdout.decode().strip()
    if out.returncode != 0 or not text:
        raise StreamProbeError(_explain_failure(out, elapsed))
    try:
        w, h = text.split("\n")[0].split(",")
        return int(w), int(h)
    except ValueError:
        # 流头里没有尺寸信息时 ffprobe 输出 "N/A,N/A" 之类
        raise StreamProbeError(f"ffprobe 未探到分辨率, 输出: {text!r}") from None


def _explain_failure(out: subprocess.CompletedProcess, elapsed: float) -> str:
    stderr = out.stderr.decode("utf-8", "replace").strip()
    lines = stderr.splitlines()
    tail = " | ".join(lines[-_STDERR_TAIL_LINES:]) or "(stderr 无输出)"
    detail = f"ffprobe exit={out.returncode} 耗时 {elapsed:.1f}s: {tail}"
    if out.returncode == 0:
        return f"能连上但没有视频流(v:0), 该地址可能只有音频或尚无媒体数据; {detail}"
    hint = next((h for needle, h in _STDERR_HINTS if needle in stderr), None)
    return f"{hint}; {detail}" if hint else detail


def diagnose_open_failure(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT) -> str:
    """cv2.VideoCapture 打不开时补一次 ffprobe, 拿到 cv2 后端不暴露的失败原因。

    只在失败路径调用, 代价是把这次失败到重连的间隔最多拉长一个 timeout。
    """
    try:
        w, h = probe_resolution(url, timeout)
    except StreamProbeError as e:
        return str(e)
    except OSError as e:
        return f"cv2 后端不暴露原因, ffprobe 也不可用无法诊断 ({e})"
    return (f"ffprobe 能探到 {w}x{h} 的流但 cv2 打不开"
            f"(多为 cv2 的 FFMPEG 后端不支持该编码/协议, 或流刚恢复)")
