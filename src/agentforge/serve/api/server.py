"""FastAPI inference server with OpenAI-compatible /v1/chat/completions endpoint.

Backed by MLX-LM for Apple Silicon hardware-accelerated inference.
Compatible with any OpenAI client (LangChain, Continue.dev, Open WebUI, curl).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agentforge.config import ServerConfig


# ---------------------------------------------------------------------------
# Request / response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

_model_state: dict = {}


def _load_model(model_path: str, adapter_path: Optional[str], cfg: ServerConfig) -> None:
    """Load model + tokenizer into global state using MLX-LM."""
    from mlx_lm import load  # type: ignore[import]

    load_kwargs: dict = {"model_path": model_path}
    if adapter_path:
        load_kwargs["adapter_path"] = adapter_path

    model, tokenizer = load(**load_kwargs)
    _model_state["model"] = model
    _model_state["tokenizer"] = tokenizer
    _model_state["model_path"] = model_path
    _model_state["cfg"] = cfg


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(model_path: str, adapter_path: Optional[str], cfg: ServerConfig) -> FastAPI:
    """Create a FastAPI app with the model pre-loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load model on startup
        print(f"Loading model: {model_path}")
        if adapter_path:
            print(f"Adapter: {adapter_path}")
        _load_model(model_path, adapter_path, cfg)
        print("Model ready.")
        yield
        # Cleanup on shutdown
        _model_state.clear()

    app = FastAPI(title="AgentForge API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": _model_state.get("model_path")}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": _model_state.get("model_path", "local"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agentforge",
            }],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if "model" not in _model_state:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model = _model_state["model"]
        tokenizer = _model_state["tokenizer"]
        server_cfg: ServerConfig = _model_state["cfg"]

        max_tokens = request.max_tokens or server_cfg.max_tokens
        temp = request.temperature if request.temperature is not None else server_cfg.temp
        top_p = request.top_p if request.top_p is not None else server_cfg.top_p

        # Build prompt string from messages
        prompt = _messages_to_prompt(tokenizer, [m.model_dump() for m in request.messages])

        if request.stream:
            return StreamingResponse(
                _stream_response(model, tokenizer, prompt, max_tokens, temp, top_p, request.model),
                media_type="text/event-stream",
            )

        # Non-streaming
        from mlx_lm import generate  # type: ignore[import]

        response_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            verbose=False,
        )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response_text))

        return ChatCompletionResponse(
            id=completion_id,
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_prompt(tokenizer, messages: list[dict]) -> str:
    """Apply the tokenizer's chat template if available, else naive join."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    # Naive fallback
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


async def _stream_response(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temp: float,
    top_p: float,
    model_name: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks in OpenAI stream format."""
    import json
    from mlx_lm.utils import generate_step  # type: ignore[import]
    import mlx.core as mx

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    prompt_tokens = tokenizer.encode(prompt)
    prompt_mx = mx.array(prompt_tokens)

    tokens_generated = 0
    for token, _ in generate_step(prompt_mx, model, temp=temp, top_p=top_p):
        if tokens_generated >= max_tokens:
            break
        tokens_generated += 1

        token_text = tokenizer.decode([token.item()])
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": token_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
