"""
eDifFIQA 人脸质量评估

基于 eDifFIQA 模型的 ONNX 推理封装，支持 Tiny/Small/Medium/Large 四种变体。
输入 112×112 对齐人脸, 输出 [0, 1] 质量分 (越高越适合人脸识别)。

变体说明:
  - tiny:   MobileFaceNet backbone (~1.7M params)  — 最快
  - small:  IResNet-18 backbone    (~11M params)   — 平衡
  - medium: IResNet-50 backbone    (~44M params)   — 较高精度
  - large:  IResNet-100 backbone   (~65M params)   — 最高精度, 跨模型泛化最好

参考:
- 论文: https://arxiv.org/abs/2310.09537
- 模型 (Tiny): https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa
- 模型 (S/M/L): https://github.com/yakhyo/face-image-quality-assessment
"""
from __future__ import annotations

from functools import cache

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from src.config import get_config, MODELS_DIR

# 变体名 → ONNX 文件名映射
_VARIANT_TO_FILE = {
    "tiny": "ediffiqa_tiny.onnx",
    "small": "ediffiqa_s.onnx",
    "medium": "ediffiqa_m.onnx",
    "large": "ediffiqa_l.onnx",
}


class EDifFIQA:
    """eDifFIQA 人脸质量评估 — ONNX 推理。

    支持 tiny/small/medium/large 四种变体, 通过 config.face.ediffiqa_variant 配置。

    预处理: BGR → RGB, [0,255] → [-1,1], HWC → NCHW
    后处理: sigmoid(logit) → [0, 1]
    """

    def __init__(self) -> None:
        cfg = get_config()
        variant = cfg.face.ediffiqa_variant.lower()

        if variant not in _VARIANT_TO_FILE:
            raise ValueError(
                f"Unknown eDifFIQA variant: '{variant}'. "
                f"Valid options: {list(_VARIANT_TO_FILE.keys())}"
            )

        model_filename = _VARIANT_TO_FILE[variant]
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"eDifFIQA model not found: {model_path}. "
                f"Run download_models.sh or manually download the '{variant}' variant.\n"
                f"  Tiny:        https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa\n"
                f"  Small/Med/L: https://github.com/yakhyo/face-image-quality-assessment/releases"
            )

        ctx_id = cfg.hardware.insightface_ctx_id
        providers = (
            [("CUDAExecutionProvider", {"device_id": ctx_id}), "CPUExecutionProvider"]
            if ctx_id >= 0
            else ["CPUExecutionProvider"]
        )

        self._session = ort.InferenceSession(
            str(model_path), providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._variant = variant
        logger.info("EDifFIQA loaded: variant={}, model={}, ctx_id={}", variant, model_filename, ctx_id)

    @property
    def variant(self) -> str:
        """当前使用的变体名。"""
        return self._variant

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
def get_ediffiqa() -> EDifFIQA:
    """获取 eDifFIQA 质量评估器 (单例缓存)。"""
    return EDifFIQA()
