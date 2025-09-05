from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional

import requests
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from config import OpenRouterConfig, load_openrouter_config


@dataclass(frozen=True)
class TextAnswer:
    text: str
    model: str


@dataclass(frozen=True)
class VisionAnswer:
    description: str
    model: str


class LLMClient:
    def __init__(self, cfg: Optional[OpenRouterConfig] = None):
        self.cfg = cfg or load_openrouter_config()

    # noinspection PyArgumentList
    def _text_model(self) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_name=self.cfg.text_model,
            base_url=self.cfg.base_url,
            provider=OpenRouterProvider(api_key=self.cfg.api_key),
        )

    async def ask_text(self, prompt: str, system: Optional[str] = None) -> TextAnswer:
        model = self._text_model()
        agent = Agent(model, system_prompt=system)
        result = await agent.run(prompt)
        # Prefer .data if available; fall back to string representation
        text = getattr(result, "data", None)
        if text is None:
            text = str(result)
        return TextAnswer(text=str(text), model=self.cfg.text_model)

    async def describe_image(
        self,
        image_b64: str,
        mime_type: str,
        prompt: Optional[str] = None,
        max_tokens: int = 512,
    ) -> VisionAnswer:
        """Describe an image using an OpenRouter vision-capable model via Chat Completions API.

        Uses a data URL to pass the image in the request.
        """
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            }
        )
        payload = {
            "model": self.cfg.vision_model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
        }

        # Run blocking HTTP in a thread to keep compatibility with async callers
        def _call():
            resp = requests.post(
                url, headers=headers, data=json.dumps(payload), timeout=60
            )
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_call)
        try:
            description = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(
                f"Unexpected response from OpenRouter vision model: {data}"
            ) from e

        return VisionAnswer(description=description, model=self.cfg.vision_model)
