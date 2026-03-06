"""
Image analyzer — determines if an image is a bill/receipt and extracts its contents.

If the image is a bill, returns structured text that the chat agent can process
via its add_bill tool. If not, returns a user-facing explanation of what was detected.

Supported providers and their vision models:
  openai  → gpt-4o (default vision model)
  bedrock → anthropic.claude-3-5-sonnet-20241022-v2:0 (best vision model on Bedrock;
            handles receipt OCR, table layouts, and handwritten text well)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from agents.llm_factory import create_vision_llm
from config import ImageAnalyzerConfig

_ANALYSIS_PROMPT = """You are analyzing an image to determine if it is a restaurant or food bill/receipt.

Examine the image carefully and respond in the following JSON format:

{
  "is_bill": true or false,
  "description": "brief description of what the image shows",
  "message": "user-friendly message about what you found",
  "bill_text": "if is_bill is true: reproduce the full bill as clean text with each item on its own line including prices, tax, tip, and total. If is_bill is false: empty string"
}

Guidelines:
- is_bill should be true ONLY for restaurant receipts, food bills, or expense invoices
- If the image is blurry, unreadable, or not a bill, set is_bill to false with a clear message
- For bill_text, preserve all item names and prices exactly as shown
- Separate tax and tip clearly from food items in bill_text
- Do not add items that are not visible in the image
- Respond with valid JSON only, no extra text
"""


@dataclass
class ImageAnalysisResult:
    is_bill: bool
    description: str   # what the image shows
    message: str       # user-friendly response to show in chat
    bill_text: str     # extracted bill content (empty if not a bill)


class ImageAnalyzer:
    """Analyzes images to detect bills and extract their contents."""

    def __init__(self, config: ImageAnalyzerConfig):
        self._llm = create_vision_llm(config)
        self._provider = config.provider

    def analyze(self, image_bytes: bytes, media_type: str = "image/jpeg") -> ImageAnalysisResult:
        """
        Analyze an image and return structured results.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, WEBP, or GIF).
            media_type:  MIME type of the image, e.g. "image/jpeg".

        Returns:
            ImageAnalysisResult with is_bill, description, message, and bill_text.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        message = self._build_message(b64, media_type)

        try:
            response = self._llm.invoke([message])
            return self._parse_response(response.content)
        except Exception as e:
            return ImageAnalysisResult(
                is_bill=False,
                description="Analysis failed",
                message=f"I couldn't analyze the image: {e}",
                bill_text="",
            )

    def _build_message(self, b64: str, media_type: str) -> HumanMessage:
        """Build a provider-appropriate multimodal message."""
        if self._provider == "openai":
            return HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": _ANALYSIS_PROMPT},
            ])
        else:
            # Bedrock (Claude via Converse API)
            return HumanMessage(content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": _ANALYSIS_PROMPT},
            ])

    def _parse_response(self, content: str) -> ImageAnalysisResult:
        """Parse the LLM's JSON response into an ImageAnalysisResult."""
        import json
        import re

        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()

        try:
            data = json.loads(clean)
            return ImageAnalysisResult(
                is_bill=bool(data.get("is_bill", False)),
                description=data.get("description", ""),
                message=data.get("message", ""),
                bill_text=data.get("bill_text", ""),
            )
        except json.JSONDecodeError:
            # LLM returned something unparseable — treat as non-bill
            return ImageAnalysisResult(
                is_bill=False,
                description="Parse error",
                message="I analyzed the image but couldn't structure the result. Please paste the bill as text instead.",
                bill_text="",
            )
