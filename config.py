import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # dotenv is optional; ignore if not installed
    pass


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    text_model: str
    vision_model: str
    base_url: str = OPENROUTER_BASE_URL


def _get_env(
    name: str, required: bool = True, default: Optional[str] = None
) -> Optional[str]:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. Please set it in your environment."
        )
    return val


def load_openrouter_config() -> OpenRouterConfig:
    """Load OpenRouter configuration from environment variables.

    Required:
      - OPENROUTER_API_KEY
    Optional:
      - OPENROUTER_TEXT_MODEL (defaults to z-ai/glm-4.5)
      - OPENROUTER_VISION_MODEL (defaults to z-ai/glm-4.5v)
      - OPENROUTER_BASE_URL (defaults to https://openrouter.ai/api/v1)
    """
    api_key = _get_env("OPENROUTER_API_KEY")
    text_model = _get_env("OPENROUTER_TEXT_MODEL", required=False, default="z-ai/glm-4.5")
    vision_model = _get_env("OPENROUTER_VISION_MODEL", required=False, default="z-ai/glm-4.5v")
    base_url = _get_env(
        "OPENROUTER_BASE_URL", required=False, default=OPENROUTER_BASE_URL
    )

    return OpenRouterConfig(
        api_key=api_key,  # type: ignore[arg-type]
        text_model=text_model,  # type: ignore[arg-type]
        vision_model=vision_model,  # type: ignore[arg-type]
        base_url=base_url or OPENROUTER_BASE_URL,
    )
