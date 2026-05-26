"""
FastAPI inference server wrapping the GPT chatbot for REST API access.

This server exposes an OpenAI-compatible /v1/chat/completions endpoint,
allowing clients such as Open WebUI to connect to the self-hosted model
as if it were the OpenAI API.

Features
--------
* OpenAI-compatible REST endpoint  (POST /v1/chat/completions)
* Bearer-token authentication with a dedicated /auth endpoint consumed
  by Nginx's auth_request directive
* HTTP access-log middleware recording method, path, status, and latency
* Persistent CSV log of every inference request in logs/requests.csv
* Prometheus metrics auto-exposed on /metrics via
  prometheus-fastapi-instrumentator
* Configurable system prompt via the SYSTEM_PROMPT environment variable
* asyncio.Lock that serialises inference calls to protect the shared model

Usage
-----
    # From scripts/llm_from_scratch/ with the virtual environment active:
    source venv/bin/activate
    uvicorn api_server:app --host 0.0.0.0 --port 8000

    # Behind Nginx (trust forwarded headers from 127.0.0.1):
    uvicorn api_server:app --host 127.0.0.1 --port 8000 \\
        --proxy-headers --forwarded-allow-ips=127.0.0.1

Environment Variables
---------------------
    CHECKPOINT_PATH   Path to model checkpoint (default: checkpoints/best_model.pt)
    API_KEY           Bearer token clients must supply  (default: changeme)
    SYSTEM_PROMPT     Override the default system prompt text
    LOG_CSV_PATH      Output path for the request CSV log (default: logs/requests.csv)
    TRUSTED_IPS       Comma-separated proxy IPs whose X-Forwarded-For to trust
                      (default: 127.0.0.1)
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Local modules — run from scripts/llm_from_scratch/
from chatbot import (
    ASSISTANT_PREFIX,
    HUMAN_PREFIX,
    generate_response,
    load_model,
)
from model import GPTModel
from tokenizer import CharTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH: str = os.environ.get(
    "CHECKPOINT_PATH", os.path.join("checkpoints", "best_model.pt")
)
API_KEY: str = os.environ.get("API_KEY", "changeme")
SYSTEM_PROMPT: str = os.environ.get(
    "SYSTEM_PROMPT", "You are a helpful AI assistant."
)
LOG_CSV_PATH: str = os.environ.get(
    "LOG_CSV_PATH", os.path.join("logs", "requests.csv")
)
TRUSTED_IPS: list[str] = [
    ip.strip()
    for ip in os.environ.get("TRUSTED_IPS", "127.0.0.1").split(",")
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ensure the logs directory exists at import time
Path(LOG_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

_CSV_FIELDS = [
    "timestamp",
    "request_id",
    "client_ip",
    "model",
    "messages_json",
    "response",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "status_code",
]


def _append_csv(row: dict) -> None:
    """Append one request record to the CSV log (CPython GIL makes this safe)."""
    path = Path(LOG_CSV_PATH)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Global model state and inference lock
# ---------------------------------------------------------------------------

_model: GPTModel | None = None
_tokenizer: CharTokenizer | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_inference_lock: asyncio.Lock | None = None  # created inside lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release resources on shutdown."""
    global _model, _tokenizer, _inference_lock

    _inference_lock = asyncio.Lock()
    logger.info(
        "Loading model from '%s' on device '%s' ...", CHECKPOINT_PATH, _device
    )
    try:
        _model, _tokenizer = load_model(CHECKPOINT_PATH, _device)
    except SystemExit:
        logger.error(
            "Checkpoint not found at '%s'. Run train.py first.", CHECKPOINT_PATH
        )
        sys.exit(1)

    logger.info("Model ready — vocabulary size: %d", _tokenizer.vocab_size)
    yield
    logger.info("Inference server shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Inference Server",
    description=(
        "OpenAI-compatible REST API for a custom GPT-style language model. "
        "Connect Open WebUI or any OpenAI-compatible client to this server."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus — auto-instruments all routes and exposes /metrics
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# HTTP middleware: access log with latency
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _access_log(request: Request, call_next):
    """Log every HTTP request with method, path, status code, and latency."""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "%-6s %-40s  status=%d  %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        ms,
    )
    return response


# ---------------------------------------------------------------------------
# Schemas — OpenAI-compatible
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-local"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=200, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(default=40, ge=1)


class _Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[_Choice]
    usage: _Usage


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _bearer_token(request: Request) -> str | None:
    """Extract the Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    return auth[len("Bearer "):] if auth.startswith("Bearer ") else None


def _authorized(request: Request) -> bool:
    """Return True only when the supplied token equals the configured API_KEY."""
    return _bearer_token(request) == API_KEY


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Utility"])
async def health() -> dict:
    """Liveness probe — returns 200 OK while the server is running."""
    return {"status": "ok", "device": _device}


@app.get("/auth", tags=["Auth"])
async def auth_check(request: Request) -> Response:
    """
    Internal authentication sub-request endpoint for Nginx auth_request.

    Returns HTTP 200 if the Authorization: Bearer <token> header matches the
    configured API_KEY, or HTTP 401 to block the upstream request.

    Nginx marks this location as ``internal`` so it cannot be called directly
    by external clients.
    """
    if _authorized(request):
        return Response(status_code=status.HTTP_200_OK)
    return Response(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers={"WWW-Authenticate": 'Bearer realm="LLM Inference Server"'},
    )


@app.get("/v1/models", tags=["Models"])
async def list_models(request: Request) -> dict:
    """List available models (required by some OpenAI-compatible clients)."""
    if not _authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-local",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Inference"],
)
async def chat_completions(
    request: Request, body: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completion endpoint.

    Accepts a ``messages`` list with roles ``system``, ``user``, and
    ``assistant``.  Builds a structured prompt using the same
    ``### Human: / ### Assistant:`` template as chatbot.py, runs
    autoregressive inference, and returns the generated text in standard
    OpenAI response format.

    Concurrent inference calls are serialised with an asyncio.Lock to
    protect the shared model instance.
    """
    if not _authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _model is None or _tokenizer is None or _inference_lock is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Prefer the real client IP forwarded by Nginx
    client_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    t_start = time.perf_counter()

    # --- Parse messages into conversation history + final user turn ---
    system_content = SYSTEM_PROMPT
    history: list[tuple[str, str]] = []
    pending_user: str | None = None

    for msg in body.messages:
        if msg.role == "system":
            system_content = msg.content
        elif msg.role == "user":
            if pending_user is not None:
                history.append((pending_user, ""))
            pending_user = msg.content
        elif msg.role == "assistant" and pending_user is not None:
            history.append((pending_user, msg.content))
            pending_user = None

    if pending_user is None:
        raise HTTPException(
            status_code=422,
            detail="The last message must have role 'user'.",
        )

    # --- Assemble prompt using the same template as chatbot.py ---
    prompt = f"### System: {system_content}\n\n"
    for user_msg, assistant_msg in history:
        prompt += HUMAN_PREFIX + user_msg + ASSISTANT_PREFIX + assistant_msg + "\n\n"
    prompt += HUMAN_PREFIX + pending_user + ASSISTANT_PREFIX

    # --- Serialised inference ---
    async with _inference_lock:
        response_text = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_response(
                _model,
                _tokenizer,
                prompt,
                _device,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
            ),
        )

    latency_ms = (time.perf_counter() - t_start) * 1000
    prompt_tokens = len(_tokenizer.encode(prompt))
    completion_tokens = len(_tokenizer.encode(response_text)) if response_text else 0

    # --- CSV logging ---
    _append_csv(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "client_ip": client_ip,
            "model": body.model,
            "messages_json": json.dumps([m.model_dump() for m in body.messages]),
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "status_code": 200,
        }
    )

    logger.info(
        "id=%s  prompt_tokens=%d  completion_tokens=%d  latency=%.1f ms",
        request_id,
        prompt_tokens,
        completion_tokens,
        latency_ms,
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=body.model,
        choices=[
            _Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
            )
        ],
        usage=_Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
