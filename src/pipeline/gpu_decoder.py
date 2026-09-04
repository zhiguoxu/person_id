"""NVDEC GPU 拉流解码 — ffmpeg 子进程, 接口对齐 cv2.VideoCapture

背景: cv2.VideoCapture(CAP_FFMPEG) 是纯 CPU 软解, 720p H.264 单路约占
10-20% 核, 几百路规模下解码就吃光整机 CPU。NVDEC 是显卡上独立的解码
ASIC(与 NVENC 同理, 不占 CUDA 核心), 把解码搬上去后 CPU 只剩下
NV12→BGR 像素格式转换(swscale, 单路 720p@15fps 约 2-3% 核)。

实现: ffmpeg -hwaccel cuda 拉流解码, rawvideo BGR 经 stdout 管道逐帧读回。
接口对齐 cv2.VideoCapture(isOpened/read/release/set), StreamConsumer 的
读流循环两种解码器无差别使用; 打开失败由调用方按次退回 CPU 解码。

分辨率处理: rawvideo 管道无帧头, 必须先 ffprobe 探测分辨率(stream_probe)
并用 scale 滤镜锁死输出尺寸——流中途变分辨率时输出仍按锁定尺寸缩放, 不会
错位解析(重连后重新探测即跟上新分辨率)。

失败原因统一放 last_error, 由 StreamConsumer 打进日志/状态接口:
- 打开失败: 探测阶段的原因(stream_probe 已翻译成可读描述)或 ffmpeg 启动失败
- read() 失败: 只意味着 ffmpeg 子进程退出了, 服务端 EOF(设备没推数据)、
  找不到 codec 参数、NVDEC 不支持该 profile 在管道这头长得一模一样。所以
  stderr 由后台线程持续读走并保留尾部几行, 退出时连同退出码一起写入
"""
from __future__ import annotations

import collections
import subprocess
import threading

import numpy as np
from voice_agent_common.utils.logger import logger

from src.pipeline.stream_probe import (
    DEFAULT_PROBE_TIMEOUT,
    StreamProbeError,
    probe_resolution,
)

# 本机 ffmpeg 是否支持 cuda hwaccel(进程级缓存, 一次探测)
_nvdec_supported: bool | None = None

# ffmpeg stderr 只留尾部这么多行: 退出原因("Output file is empty" /
# "Could not find codec parameters" / hwaccel 报错)都在最后几行
_STDERR_TAIL_LINES = 8


def nvdec_supported() -> bool:
    """本机 ffmpeg 是否带 cuda hwaccel。不支持的机器缓存 False,
    之后每次开流直接走 CPU, 不重复付探测/失败开销。"""
    global _nvdec_supported
    if _nvdec_supported is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True, timeout=10,
            )
            _nvdec_supported = out.returncode == 0 and b"cuda" in out.stdout
        except Exception:
            _nvdec_supported = False
        logger.info("NVDEC 解码探测: {}",
                    "可用" if _nvdec_supported else "不可用(拉流走 CPU 解码)")
    return _nvdec_supported


def _gpu_index(device: str) -> int:
    """config.hardware.device('cuda:1'/'cpu') → hwaccel_device 序号。"""
    if device.startswith("cuda") and ":" in device:
        try:
            return int(device.split(":")[-1])
        except ValueError:
            pass
    return 0


class NvdecCapture:
    """GPU 解码拉流读取器(与 cv2.VideoCapture 同接口)。"""

    def __init__(self, url: str, device: str = "cuda:0",
                 probe_timeout: float = DEFAULT_PROBE_TIMEOUT) -> None:
        self._proc: subprocess.Popen | None = None
        self._width = 0
        self._height = 0
        self._frame_bytes = 0
        # True = 流本身不可达(未推流/超时), 与 NVDEC 无关——CPU 解码同样
        # 打不开, 调用方不必做无谓的 CPU 退回, 交给外层重连即可
        self.stream_unreachable = False
        # 打开失败 / read() 失败的原因, 供调用方打日志与状态接口
        self.last_error: str | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(
            maxlen=_STDERR_TAIL_LINES,
        )
        self._stderr_thread: threading.Thread | None = None

        try:
            self._width, self._height = probe_resolution(url, probe_timeout)
        except OSError as e:
            # ffprobe 不存在等本机环境问题 → 值得退回 CPU 解码
            self.last_error = f"ffprobe 不可用: {e}"
            logger.warning("NVDEC 探测环境异常: {} ({})", url, e)
            return
        except StreamProbeError as e:
            self.stream_unreachable = True
            self.last_error = str(e)
            logger.warning("拉流探测失败: {}, 原因: {}", url, e)
            return
        if self._width <= 0 or self._height <= 0:
            self.last_error = f"ffprobe 返回非法分辨率 {self._width}x{self._height}"
            return
        self._frame_bytes = self._width * self._height * 3

        try:
            self._proc = subprocess.Popen(
                [
                    # warning 级才有 "Output file is empty, nothing was encoded"
                    # (服务端 EOF、一帧没出) 和 "Could not find codec parameters",
                    # error 级下这两种最常见的退出都是静默的
                    "ffmpeg", "-loglevel", "warning",
                    "-hwaccel", "cuda", "-hwaccel_device", str(_gpu_index(device)),
                    "-fflags", "nobuffer", "-flags", "low_delay",
                    "-i", url,
                    "-an",
                    # 锁死输出尺寸: 流中途变分辨率也不会错位解析
                    "-vf", f"scale={self._width}:{self._height}",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            self.last_error = f"ffmpeg 启动失败: {e}"
            logger.warning("NVDEC ffmpeg 启动失败: {}", e)
            self._proc = None
            return
        # stderr 必须有人持续读: 管道 64KB 写满后 ffmpeg 会卡在日志写入上,
        # 表现为拉流无故停帧
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stderr,),
            name="nvdec-stderr", daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, pipe) -> None:
        try:
            for raw in pipe:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    self._stderr_tail.append(line)
        except (OSError, ValueError):
            pass  # release() 关管道时的收尾异常

    def _describe_exit(self, got: int) -> str:
        """stdout 短读后汇总退出原因。短读即 ffmpeg 已关 stdout, 进程随即退出,
        等一小会拿退出码, 再等 stderr 线程把尾部读完。"""
        rc: int | None = None
        proc = self._proc
        if proc is not None:
            try:
                rc = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        # 负数退出码 = 被信号杀(-9 多为 OOM killer)
        rc_text = f"exit={rc}" if rc is not None else "进程未退出"
        tail = " | ".join(self._stderr_tail) or "(stderr 无输出)"
        return f"ffmpeg {rc_text}, 短读 {got}/{self._frame_bytes} 字节, stderr: {tail}"

    # ── cv2.VideoCapture 接口 ──

    def isOpened(self) -> bool:
        return self._proc is not None

    def set(self, *_args) -> bool:
        """兼容 cv2 的属性设置调用(如 CAP_PROP_BUFFERSIZE), 无操作。"""
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._proc is None:
            return False, None
        # BufferedReader.read(n) 阻塞至读满 n 字节, 只有 EOF(流断/进程退出)
        # 才会短读
        data = self._proc.stdout.read(self._frame_bytes)
        if data is None or len(data) < self._frame_bytes:
            self.last_error = self._describe_exit(len(data) if data else 0)
            return False, None
        frame = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape(self._height, self._width, 3)
            .copy()  # frombuffer 是只读视图, 下游(畸变矫正等)需要可写数组
        )
        return True, frame

    def release(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
