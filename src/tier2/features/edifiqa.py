"""
eDifFIQA(T) 人脸质量评估

基于 eDifFIQA Tiny 模型 (MobileFaceNet, ~1.7M params) 的 ONNX 推理封装。
输入 112×112 对齐人脸, 输出 [0, 1] 质量分 (越高越适合人脸识别)。

参考:
- 论文: https://arxiv.org/abs/2310.09537
- 模型: https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa
"""
from __future__ import annotations

from functools import cache

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from src.config import get_config, MODELS_DIR


class EDifFIQA:
    """eDifFIQA(T) 人脸质量评估 — ONNX 推理。

    预处理: BGR → RGB, [0,255] → [-1,1], HWC → NCHW
    后处理: sigmoid(logit) → [0, 1]
    """

    def __init__(self) -> None:
        model_path = MODELS_DIR / "edifiqa_tiny.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"eDifFIQA model not found: {model_path}. "
                "Download from: https://huggingface.co/opencv/"
                "face_image_quality_assessment_ediffiqa"
            )

        ctx_id = get_config().hardware.insightface_ctx_id
        providers = (
            [("CUDAExecutionProvider", {"device_id": ctx_id}), "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        self._session = ort.InferenceSession(
            str(model_path), providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("EDifFIQA loaded: model=edifiqa_tiny, ctx_id={}", ctx_id)

    def predict(self, aligned_face_112: np.ndarray) -> float:
        """评估对齐人脸的质量。

        Args:
            aligned_face_112: 112×112 BGR 对齐人脸 (来自 norm_crop)。

        Returns:
            质量分 [0, 1], 越高越好。
        """
        # 预处理: BGR → RGB, normalize to [-1, 1], HWC → NCHW
        rgb = cv2.cvtColor(aligned_face_112, cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) / 255.0) - 0.5) / 0.5
        blob = np.moveaxis(blob[None, ...], -1, 1)  # (1, 3, 112, 112)

        # ONNX 推理
        raw_output = self._session.run(None, {self._input_name: blob})[0]
        score = float(raw_output.flatten()[0])

        # 后处理: 如果输出是 raw logit (可能 <0 或 >1), 做 sigmoid
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + np.exp(-score))

        return score


@cache
def get_edifiqa() -> EDifFIQA:
    """获取 eDifFIQA 质量评估器 (单例缓存)。"""
    return EDifFIQA()
