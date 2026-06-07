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
import re

import openai
from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_config


class VLMResponse(BaseModel):
    """VLM 仲裁的结构化响应。"""
    matched_candidate_id: str | None = None
    grade: str = "STRANGER"         # "DEFINITE" | "CONFIDENT" | "SUSPECTED" | "STRANGER"
    reasoning: str = ""
    distinguishing_features: list[str] = Field(default_factory=list)


def _get_vlm_system_prompt() -> str:
    return """You are a person re-identification expert. You will be shown a query image of a person and a list of candidate images. Your task is to determine which candidate (if any) is the SAME person as the query.

Analyze carefully:
- Facial features (if visible)
- Body build and proportions
- Clothing and accessories
- Hair style and color
- Any distinguishing features

You MUST assign one of these grades to your best-matching candidate:
- "DEFINITE": You are absolutely certain they are the same person. Multiple strong features match (face, build, clothing, accessories).
- "CONFIDENT": You are highly confident they match. Most features align, though minor details may be occluded or unclear.
- "SUSPECTED": They look similar, but you lack sufficient clear evidence to be confident. Only a few features match.
- "STRANGER": None of the candidates match the query person. Set matched_candidate_id to null.

Respond ONLY with a JSON object in this exact format:
{
    "matched_candidate_id": "ID of the most likely matching candidate, or null if grade is STRANGER",
    "grade": "DEFINITE or CONFIDENT or SUSPECTED or STRANGER",
    "reasoning": "brief explanation of why you chose this grade",
    "distinguishing_features": ["feature1", "feature2"]
}"""


class VLMArbitrator:
    """VLM 仲裁器 — 通过 Qwen-VL-Max 视觉对比确认身份。

    作为 ReID 管线的二级仲裁手段, 在传统特征匹配不确定时提供
    基于视觉理解的辅助判断。
    """

    def __init__(self) -> None:
        self._config = get_config().vlm
        self._client = openai.AsyncOpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout_sec,
            max_retries=self._config.max_retries,
        )

        logger.info(
            "VLMArbitrator initialized (model={}, base_url={})",
            self._config.model,
            self._config.base_url,
        )

    async def arbitrate(
            self,
            query_image: bytes,
            candidate_images: list[tuple[str, bytes]],
    ) -> VLMResponse:
        """调用 VLM 对比查询图片与候选图片。

        Args:
            query_image: 查询人物的 JPEG 图片字节。
            candidate_images: 候选人列表, 每项为 (person_id, jpeg_bytes)。

        Returns:
            VLMResponse: 解析后的结构化结果。
            出错时返回默认不匹配结果。
        """
        if not query_image or not candidate_images:
            logger.warning("VLM arbitrate called with empty images")
            return VLMResponse(reasoning="VLM arbitration failed or unavailable")

        try:
            # 构建消息内容
            content = _build_message_content(query_image, candidate_images)

            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": _get_vlm_system_prompt()},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512,
            )

            # 解析响应
            raw_text = response.choices[0].message.content
            result = _parse_response(raw_text)

            logger.info(
                "VLM arbitration result: matched={}, grade={}, reason={}",
                result.matched_candidate_id,
                result.grade,
                result.reasoning[:80],
            )
            return result

        except Exception as e:
            logger.error("VLM arbitration failed: {}", str(e))
            return VLMResponse(reasoning="VLM arbitration failed or unavailable")


# ------------------------------------------------------------------
# 内部函数
# ------------------------------------------------------------------

def _build_message_content(
        query_image: bytes,
        candidate_images: list[tuple[str, bytes]],
) -> list[dict]:
    """构建 VLM API 的消息内容 (含图片)。"""
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


def _parse_response(raw_text: str) -> VLMResponse:
    """解析 VLM 返回的文本为 VLMResponse。

    尝试直接解析, 失败则尝试从 markdown 代码块中提取。
    """
    if not raw_text:
        return VLMResponse(reasoning="Empty VLM response")

    # 尝试直接 JSON 解析
    try:
        data = json.loads(raw_text.strip())
        return _normalize(data)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块提取
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1).strip())
            return _normalize(data)
        except json.JSONDecodeError:
            pass

    # 尝试找到任何 JSON 对象
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            return _normalize(data)
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse VLM response: {}", raw_text[:200])
    return VLMResponse(reasoning=f"Unparseable response: {raw_text[:100]}")


def _normalize(data: dict) -> VLMResponse:
    """将原始 dict 标准化为 VLMResponse。"""
    matched_id = data.get("matched_candidate_id")
    if matched_id == "null":
        matched_id = None
    elif isinstance(matched_id, str):
        matched_id = matched_id.strip()

    grade = str(data.get("grade", "STRANGER")).upper().strip()
    if grade not in ("DEFINITE", "CONFIDENT", "SUSPECTED", "STRANGER"):
        grade = "STRANGER"

    return VLMResponse(
        matched_candidate_id=matched_id,
        grade=grade,
        reasoning=str(data.get("reasoning", "")),
        distinguishing_features=list(data.get("distinguishing_features", [])),
    )
