"""声音 embedding 提取 — PCM → 说话人向量(纯函数, 无状态)

person_id 在声纹链路中的唯一职责: 把一段语音变成 512 维说话人向量(GPU 6.8ms)。
比对/阈值/注册等身份决策与声纹库都在 agent_server 侧(与花名册同生命周期);
voice_server 只做透传。架构定稿见 voice_agent/test/speaker_id/recipe_findings.md。

配方(评测锁定): 能量裁剪 → 3s 窗/1.5s 步子段 embedding 取均值 → L2 归一化。
全程 best-effort: 模型缺失/推理异常降级为"无结果", 不抛异常。
"""

from __future__ import annotations

import os
import threading
import time
from functools import cache

import numpy as np
from loguru import logger

from src.config import get_config

SR = 16000


def _l2norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """int16 PCM 字节流 → float32 [-1, 1)。voice_server VAD 的输出格式。"""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _energy_trim(samples: np.ndarray, thr_ratio: float = 0.1) -> np.ndarray:
    """按 20ms 帧 RMS 裁掉首尾静音(轮次音频含 pre-buffer 与 VAD 尾静音)。"""
    frame = SR // 50
    n = len(samples) // frame
    if n == 0:
        return samples
    rms = np.sqrt(np.mean(samples[:n * frame].reshape(n, frame) ** 2, axis=1))
    thr = thr_ratio * np.percentile(rms, 95)
    active = np.where(rms > thr)[0]
    if len(active) == 0:
        return samples
    return samples[active[0] * frame:(active[-1] + 1) * frame]


# 模型文件不入 git(37.8MB): 缺失时启动自动下载。镜像优先(境内), 官方兜底
_MODEL_URLS = [
    "https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/{name}",
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/{name}",
]


def _download_model(model_path: str, timeout_sec: float = 120) -> bool:
    """按文件名从模型仓下载到 model_path(原子写入)。全部源失败返回 False。"""
    import urllib.request
    name = os.path.basename(model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp = model_path + ".downloading"
    for url in (u.format(name=name) for u in _MODEL_URLS):
        try:
            logger.info(f"声纹模型缺失, 下载中: {url}")
            with urllib.request.urlopen(url, timeout=timeout_sec) as resp, \
                    open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            os.replace(tmp, model_path)
            logger.info(f"声纹模型已下载: {model_path} "
                        f"({os.path.getsize(model_path) >> 20}MB)")
            return True
        except Exception as e:
            logger.warning(f"声纹模型下载失败({url}): {e}")
    return False


class _Extractor:
    """sherpa-onnx 说话人 embedding 封装, 实现子段平均配方。线程安全由外层锁保证。"""

    def __init__(self, model_path: str, num_threads: int, provider: str,
                 seg_window_sec: float, seg_hop_sec: float):
        import sherpa_onnx  # 延迟导入: 未启用/缺依赖时不拖累进程启动
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=num_threads, provider=provider)
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dim = self._ex.dim
        self._win = int(seg_window_sec * SR)
        self._hop = int(seg_hop_sec * SR)

    def _raw_embed(self, samples: np.ndarray) -> np.ndarray:
        st = self._ex.create_stream()
        st.accept_waveform(SR, samples)
        st.input_finished()
        return _l2norm(np.array(self._ex.compute(st), dtype=np.float32))

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """净语音 → 子段平均 embedding(短于一个窗则整段)。"""
        if len(samples) <= self._win:
            return self._raw_embed(samples)
        embs = [self._raw_embed(samples[i:i + self._win])
                for i in range(0, len(samples) - self._win + 1, self._hop)]
        return _l2norm(np.mean(embs, axis=0))


class VoiceEmbedExtractor:
    """embedding 提取门面(声纹模态的感知端)。命名与 voice_server 区分:
    本类持模型真正提取向量, voice_server 侧的 VoiceEmbedder 只是取向量的客户端。
    sherpa stream 非线程安全, 一把锁串行(GPU 单次 ~7ms)。"""

    def __init__(self) -> None:
        cfg = get_config().voice_embed
        self._lock = threading.Lock()
        self._extractor: _Extractor | None = None
        if not cfg.enabled:
            logger.info("声音 embedding 未启用 (voice.enabled=false)")
            return
        try:
            if not os.path.exists(cfg.model_path) and not _download_model(cfg.model_path):
                raise FileNotFoundError(f"模型不存在且下载失败: {cfg.model_path}")
            self._extractor = _Extractor(
                cfg.model_path, cfg.num_threads, cfg.provider,
                cfg.seg_window_sec, cfg.seg_hop_sec)
            logger.info(f"声音 embedding 已就绪: model={os.path.basename(cfg.model_path)} "
                        f"provider={cfg.provider} dim={self._extractor.dim}")
        except Exception as e:
            logger.warning(f"声音 embedding 初始化失败(本次运行禁用): {e}")

    @property
    def available(self) -> bool:
        return self._extractor is not None

    @property
    def dim(self) -> int:
        return self._extractor.dim if self._extractor else 0

    def warmup(self) -> None:
        """dummy 推理一次, 把 CUDA EP 初始化成本从首个请求挪到启动期。"""
        if self.available:
            t0 = time.perf_counter()
            with self._lock:
                self._extractor.embed(np.zeros(SR, dtype=np.float32))
            logger.info("声音 embedding 模型预热完成 ({:.0f}ms)",
                        (time.perf_counter() - t0) * 1000)

    def embed(self, pcm_bytes: bytes) -> tuple[list[float], float] | None:
        """一段音频 → (embedding, 净语音秒数)。异常/不可用返回 None。"""
        if not self.available or not pcm_bytes:
            return None
        try:
            samples = _energy_trim(_pcm_to_float32(pcm_bytes))
            sec = len(samples) / SR
            with self._lock:
                emb = self._extractor.embed(samples)
            return emb.tolist(), round(sec, 2)
        except Exception as e:
            logger.warning(f"声音 embedding 提取失败: {e}")
            return None


@cache
def get_voice_embed_extractor() -> VoiceEmbedExtractor:
    return VoiceEmbedExtractor()
