from __future__ import annotations

import asyncio
from typing import Optional

from fastmcp import FastMCP

from llm import LLMClient
from utils_image import file_to_base64 as file_to_b64
from utils_image import url_to_base64 as url_to_b64

# Lazy client creation to avoid env var requirements at import-time
_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


mcp = FastMCP(
    name="AssistanceMCP",
    instructions=(
        """
        AssistanceMCP augments your reasoning with two capabilities:
        1) Co‑Reasoning: consult co_reason() to get an advisory solution, plan, or critique from a secondary LLM.
        2) Image Processing: if you cannot process images, use image_to_base64() then vision_describe() to ask a vision model what an image contains.
        Use this MCP for writing, coding help, and academic tasks: analysis, outlining, proofs, explanations, and code reasoning.
        Treat co_reason() outputs as considerations or guidelines; you remain the primary decision-maker.
        """
    ),
)


@mcp.tool(
    name="co_reason",
    description=(
        "Query a configured text model to get advisory considerations, plans, and critiques for a problem. "
        "Intended as guidance for the main model to consult, especially for coding and academic tasks."
    ),
    tags={"reasoning", "advice", "coding", "academics"},
    meta={"version": "1.0", "author": "Netrve"},
)
async def co_reason(
    question: str,
    context: Optional[str] = None,
    audience: Optional[str] = None,
    style: Optional[str] = None,
) -> dict:
    """Return advisory considerations from a secondary text model.

    Parameters:
      - question: The main question or task to analyze.
      - context: Optional additional context, snippets, constraints, or references.
      - audience: Optional audience (e.g., beginner, expert).
      - style: Optional style (e.g., concise bullets, step-by-step, outline).
    """
    system = (
        "You are a specialist co-reasoner. Provide structured, step-by-step considerations, "
        "identify pitfalls, and propose a clear plan. Be accurate, cite assumptions, and include "
        "concise justifications. Keep output suitable as guidance for another model."
    )

    prompt_parts = [
        "Task:",
        question,
    ]
    if context:
        prompt_parts += ["\nContext:", context]
    if audience:
        prompt_parts += ["\nAudience:", audience]
    if style:
        prompt_parts += ["\nPreferred style:", style]
    prompt_parts += [
        "\nRequirements:",
        "- Provide a short summary first.",
        "- Then a structured plan or analysis in bullets or numbered steps.",
        "- Call out key risks, edge cases, and assumptions.",
        "- If code-related, include language-appropriate pseudocode or snippets when helpful.",
    ]
    prompt = "\n".join(prompt_parts)

    answer = await get_client().ask_text(prompt, system=system)
    return {
        "type": "considerations",
        "model": answer.model,
        "content": answer.text,
    }


@mcp.tool(
    name="vision_describe",
    description=(
        "Describe an image using a vision-capable model. Provide base64-encoded image data and mime_type. "
        "Optionally include a custom prompt to steer the description."
    ),
    tags={"vision", "image", "describe"},
    meta={"version": "1.0", "author": "assistance-mcp"},
)
async def vision_describe(
    image_base64: str,
    mime_type: Optional[str] = None,
    prompt: Optional[str] = None,
    max_tokens: int = 512,
) -> dict:
    """Describe an image from base64 data using the configured vision model.

    If mime_type is omitted, defaults to image/png.
    """
    mime = mime_type or "image/png"
    result = await get_client().describe_image(
        image_base64, mime, prompt=prompt, max_tokens=max_tokens
    )
    return {
        "type": "vision_description",
        "model": result.model,
        "description": result.description,
    }


@mcp.tool(
    name="image_to_base64",
    description=(
        "Convert an image from a local file path or URL to base64 and a data URL. "
        "Exactly one of file_path or url must be provided."
    ),
    tags={"image", "base64", "utility"},
    meta={"version": "1.0", "author": "assistance-mcp"},
)
def image_to_base64(
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    data_url_or_base64: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> dict:
    # Exactly one of file_path, url, or data_url_or_base64 must be provided
    provided = [p is not None for p in (file_path, url, data_url_or_base64)]
    if sum(provided) != 1:
        raise ValueError(
            "Provide exactly one of file_path, url, or data_url_or_base64."
        )

    if file_path is not None:
        res = file_to_b64(file_path, mime_type=mime_type)
        return {
            "base64": res.base64,
            "mime_type": res.mime_type,
            "data_url": res.data_url,
            "length_bytes": res.length_bytes,
            "source": res.source,
        }

    if url is not None:
        res = url_to_b64(url, mime_type=mime_type)  # type: ignore[arg-type]
        return {
            "base64": res.base64,
            "mime_type": res.mime_type,
            "data_url": res.data_url,
            "length_bytes": res.length_bytes,
            "source": res.source,
        }

    # Handle data_url_or_base64
    data_str = data_url_or_base64 or ""
    if data_str.startswith("data:"):
        # Format: data:[<mime>][;base64],<data>
        try:
            header, b64 = data_str.split(",", 1)
        except ValueError:
            raise ValueError(
                "Invalid data URL: missing comma separating header and data."
            )
        header = header[len("data:") :]
        parts = header.split(";") if header else []
        mime = (
            parts[0]
            if parts and "/" in parts[0]
            else (mime_type or "application/octet-stream")
        )
        return {
            "base64": b64,
            "mime_type": mime,
            "data_url": data_str,
            "length_bytes": len(b64) * 3 // 4,  # approximate
            "source": "data-url",
        }
    else:
        # Assume plain base64 payload
        b64 = data_str
        mime = mime_type or "application/octet-stream"
        return {
            "base64": b64,
            "mime_type": mime,
            "data_url": f"data:{mime};base64,{b64}",
            "length_bytes": len(b64) * 3 // 4,  # approximate
            "source": "raw-base64",
        }


if __name__ == "__main__":
    # Run the MCP server. FastMCP typically serves over stdio for MCP clients.
    # This call will block and handle the session until terminated by the client.
    mcp.run()
