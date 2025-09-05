# AssistanceMCP

A minimal Model Context Protocol (MCP) server built with FastMCP. It augments a primary LLM with:

- Co-Reasoning: Query a configured text-only model via OpenRouter to get advisory solutions/considerations.
- Co-Writing: Ask a secondary LLM for story considerations, complexities, and continuity guidance for an ongoing story (advisory; not prose).
- Image Processing: Use a vision-enabled model via OpenRouter to describe images. Includes a utility to convert images (file path or URL) to base64.

This project uses PydanticAI to talk to OpenRouter for text reasoning, and OpenAI-compatible Chat Completions for vision.

## Requirements
- Python 3.10+
- An OpenRouter API key: https://openrouter.ai/

Install dependencies:

```
pip install .

# Or using uv (recommended):
uv sync
```

## Configuration
Set the following environment variables (PowerShell examples on Windows):

- `OPENROUTER_API_KEY` – Your OpenRouter API key (starts with `sk-or-...`).
- `OPENROUTER_TEXT_MODEL` – The model ID for the text-only co-reasoning model.
- `OPENROUTER_VISION_MODEL` – The model ID for the vision-capable model.
- `OPENROUTER_BASE_URL` – Optional. Defaults to `https://openrouter.ai/api/v1`.

PowerShell (for current session):
```
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:OPENROUTER_TEXT_MODEL = "anthropic/claude-3.5-sonnet"
$env:OPENROUTER_VISION_MODEL = "openai/gpt-4o-mini"
```

Persistently (PowerShell):
```
setx OPENROUTER_API_KEY "sk-or-..."
setx OPENROUTER_TEXT_MODEL "anthropic/claude-3.5-sonnet"
setx OPENROUTER_VISION_MODEL "openai/gpt-4o-mini"
```

You may also place them in a `.env` file; `python-dotenv` will be used if available.

## Running the server
Run the MCP server over stdio:

```
python server.py
```

Your MCP-compatible client (e.g., an IDE or agent runtime that speaks MCP) should connect to this process over stdio.

## Provided Tools

1) co_reason (async)
- Purpose: Co-reasoning guidance from a secondary text model.
- Parameters:
  - `question: str` (required)
  - `context: str | None = None`
  - `audience: str | None = None`
  - `style: str | None = None`
- Returns:
  - `{ "type": "considerations", "model": <model_id>, "content": <text> }`
- Notes: Treat output as advisory considerations/guidelines for the main model (especially for coding and academic tasks).

2) co_write_considerations (async)
- Purpose: Co-writing guidance for an ongoing story; provides considerations/complexities and continuity checks. Produces guidance only (no prose).
- Parameters:
  - `story_so_far: str` (required)
  - `request: str | None = None`
  - `constraints: str | None = None`
  - `audience: str | None = None`
  - `tone: str | None = None`
  - `pov: str | None = None`
  - `themes: str | None = None`
  - `characters: str | None = None`
  - `notes: str | None = None`
- Returns:
  - `{ "type": "co_writing_considerations", "model": <model_id>, "content": <text> }`
- Notes: The main LLM should use this as suggestions/guidance when writing the next beats/scenes.

3) image_to_base64 (sync)
- Purpose: Convert an image to base64 and data URL.
- Parameters (exactly one required):
  - `file_path: str | None = None`
  - `url: str | None = None`
  - `mime_type: str | None = None` (optional override)
- Returns:
  - `{ base64, mime_type, data_url, length_bytes, source }`

3) vision_describe (async)
- Purpose: Describe an image using a vision-capable model.
- Parameters:
  - `image_base64: str` (base64 data only; do not include `data:` prefix)
  - `mime_type: str | None = None` (defaults to `image/png`)
  - `prompt: str | None = None`
  - `max_tokens: int = 512`
- Returns:
  - `{ "type": "vision_description", "model": <model_id>, "description": <text> }`

Typical flow if your main LLM lacks vision support:
- Use `image_to_base64(file_path=..., or url=...)` to get base64.
- Pass `base64` (or `data_url` if your client adapts it) and `mime_type` to `vision_describe` to obtain a description.

## Internals
- `config.py` loads OpenRouter config from env vars.
- `llm.py` uses PydanticAI's `OpenAIModel` backend to call the configured text model for co-reasoning (`Agent.run`). For vision, it uses the OpenAI-compatible Chat Completions endpoint at `https://openrouter.ai/api/v1/chat/completions` with an image data URL.
- `server.py` defines the FastMCP server and tools. LLM client initialization is lazy to avoid import-time failures if env vars are missing.
- `utils_image.py` provides file/URL to base64 helpers.

## Example: Direct PydanticAI usage (text)
```
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel(
    "anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)
agent = Agent(model)
result = await agent.run("What is the meaning of life?")
print(result)
```

## Notes
- Ensure your chosen models on OpenRouter support the required capabilities (text vs. vision). Model IDs and availability can change.
- This server communicates over stdio; integrate it into your MCP-aware client accordingly.
- Security: Treat API keys as secrets. Avoid logging sensitive data.
