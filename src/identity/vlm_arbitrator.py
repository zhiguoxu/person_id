"""
VLM Arbitrator — Qwen-VL-Max 视觉语言模型仲裁

当 ReID 管线输出 CONFLICT 或 SUSPECTED 状态时, 调用 VLM 进行视觉比对:
    1. 将查询图片和候选图片发送到 Qwen-VL-Max
    2. 提示模型比较是否为同一人
    3. 解析结构化 JSON 响应

使用 OpenAI 兼容 API (AsyncOpenAI), 支持超时和重试。
"""
from __future__ import annotations

import base64
import json
from typing import Any, Optional

from loguru import logger

from src.config import VLMConfig

# 延迟导入 openai 以避免模块加载失败
_openai_client = None


def _get_openai():
    """延迟导入 openai 模块。"""
    import openai
    return openai


_VLM_SYSTEM_PROMPT = """You are a person re-identification expert. You will be shown images of people and asked to determine if they are the same person.

Analyze carefully:
- Facial features (if visible)
- Body build and proportions
- Clothing and accessories
- Hair style and color
- Any distinguishing features

Respond ONLY with a JSON object in this exact format:
{
    "is_same_person": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "distinguishing_features": ["feature1", "feature2"]
}"""


class VLMArbitrator:
    """VLM 仲裁器 — 通过 Qwen-VL-Max 视觉对比确认身份。

    作为 ReID 管线的二级仲裁手段, 在传统特征匹配不确定时提供
    基于视觉理解的辅助判断。
    """

    def __init__(self, config: VLMConfig) -> None:
        self._config = config
        self._client: Optional[Any] = None
        logger.info(
            "VLMArbitrator initialized (model={}, base_url={})",
            config.model,
            config.base_url,
        )

    def _ensure_client(self) -> Any:
        """确保 AsyncOpenAI 客户端已创建。"""
        if self._client is None:
            openai = _get_openai()
            self._client = openai.AsyncOpenAI(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_sec,
                max_retries=self._config.max_retries,
            )
        return self._client

    async def arbitrate(
        self,
        query_image: bytes,
        candidate_images: list[tuple[str, bytes]],
    ) -> dict:
        """调用 VLM 对比查询图片与候选图片。

        Args:
            query_image: 查询人物的 JPEG 图片字节。
            candidate_images: 候选人列表, 每项为 (person_id, jpeg_bytes)。

        Returns:
            VLM 解析结果字典:
                - is_same_person: bool
                - confidence: float [0, 1]
                - reasoning: str
                - distinguishing_features: list[str]
            出错时返回默认不匹配结果。
        """
        default_response = {
            "is_same_person": False,
            "confidence": 0.0,
            "reasoning": "VLM arbitration failed or unavailable",
            "distinguishing_features": [],
        }

        if not query_image or not candidate_images:
            logger.warning("VLM arbitrate called with empty images")
            return default_response

        try:
            client = self._ensure_client()

            # 构建消息内容
            content = self._build_message_content(query_image, candidate_images)

            response = await client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            # 解析响应
            raw_text = response.choices[0].message.content
            result = self._parse_response(raw_text)

            logger.info(
                "VLM arbitration result: same={}, conf={:.3f}, reason={}",
                result["is_same_person"],
                result["confidence"],
                result["reasoning"][:80],
            )
            return result

        except Exception as e:
            logger.error("VLM arbitration failed: {}", str(e))
            return default_response

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_message_content(
        query_image: bytes,
        candidate_images: list[tuple[str, bytes]],
    ) -> list[dict]:
        """构建 VLM API 的消息内容 (含图片)。

        Args:
            query_image: 查询图片 JPEG 字节。
            candidate_images: 候选图片列表。

        Returns:
            OpenAI 兼容的 content 列表。
        """
        content: list[dict] = []

        # 查询图片
        query_b64 = base64.b64encode(query_image).decode("utf-8")
        content.append({
            "type": "text",
            "text": "This is the query person to identify:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{query_b64}",
            },
        })

        # 候选图片
        for i, (person_id, img_bytes) in enumerate(candidate_images):
            candidate_b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "text",
                "text": f"Candidate {i + 1} (ID: {person_id}):",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{candidate_b64}",
                },
            })

        content.append({
            "type": "text",
            "text": (
                "Compare the query person with the candidate(s). "
                "Are they the same person? Focus on the most likely candidate. "
                "Respond with the JSON format specified."
            ),
        })

        return content

    @staticmethod
    def _parse_response(raw_text: str) -> dict:
        """解析 VLM 返回的文本为结构化 JSON。

        尝试直接解析, 失败则尝试从 markdown 代码块中提取。

        Args:
            raw_text: VLM 返回的原始文本。

        Returns:
            解析后的字典, 包含标准化字段。
        """
        default = {
            "is_same_person": False,
            "confidence": 0.0,
            "reasoning": "Failed to parse VLM response",
            "distinguishing_features": [],
        }

        if not raw_text:
            return default

        # 尝试直接 JSON 解析
        try:
            result = json.loads(raw_text.strip())
            return _normalize_vlm_result(result)
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1).strip())
                return _normalize_vlm_result(result)
            except json.JSONDecodeError:
                pass

        # 尝试找到任何 JSON 对象
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                result = json.loads(brace_match.group(0))
                return _normalize_vlm_result(result)
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse VLM response: {}", raw_text[:200])
        default["reasoning"] = f"Unparseable response: {raw_text[:100]}"
        return default


def _normalize_vlm_result(result: dict) -> dict:
    """标准化 VLM 结果字段。

    Args:
        result: 原始解析的字典。

    Returns:
        包含标准化字段的字典。
    """
    return {
        "is_same_person": bool(result.get("is_same_person", False)),
        "confidence": float(min(max(result.get("confidence", 0.0), 0.0), 1.0)),
        "reasoning": str(result.get("reasoning", "")),
        "distinguishing_features": list(result.get("distinguishing_features", [])),
    }
