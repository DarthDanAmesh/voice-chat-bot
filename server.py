"""Voice chat backend: serves the UI and proxies chat to a local Ollama instance."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm:135m")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5.0"))
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "12"))
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "6000"))
MAX_REPLY_TOKENS = int(os.getenv("MAX_REPLY_TOKENS", "220"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("voice_chat")

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies are read aloud, so answer in one "
    "to three short sentences of plain spoken prose. Never use markdown, bullet lists, "
    "headings, code blocks, emoji, or URLs. Use contractions and natural phrasing. "
    "Answer in the same language the user writes in. If you do not know something, say "
    "so in one sentence instead of guessing."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(OLLAMA_TIMEOUT, connect=CONNECT_TIMEOUT),
        trust_env=False,  # local Ollama must never go through a system proxy
    )
    logger.info("Chat model %s, Ollama at %s", OLLAMA_MODEL, OLLAMA_HOST)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="Voice Chat", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=60)
    language: str | None = Field(default=None, max_length=16)


def _ndjson(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode()


def _history(request: ChatRequest) -> list[dict]:
    """Trim history to a recent window and a character budget, oldest first.

    The newest message is always kept, even if it alone exceeds the budget.
    """
    recent = request.messages[-CONTEXT_MESSAGES:]
    kept: list[dict] = []
    budget = CONTEXT_CHAR_BUDGET
    for position, message in enumerate(reversed(recent)):
        if position and len(message.content) > budget:
            break
        budget -= len(message.content)
        kept.append({"role": message.role, "content": message.content})
    return list(reversed(kept))


def _ollama_payload(request: ChatRequest) -> dict:
    system = SYSTEM_PROMPT
    if request.language:
        system += f" The user is speaking {request.language}."
    return {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, *_history(request)],
        "stream": True,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_predict": MAX_REPLY_TOKENS,
        },
    }


def normalise(text: str) -> str:
    """Cleanup for the consolidated reply only; per-chunk text must keep its spacing."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[`*#]+", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    if body.messages[-1].role != "user":
        return JSONResponse(
            status_code=422,
            content={"detail": "The last message must be from the user."},
        )

    client: httpx.AsyncClient = request.app.state.http
    upstream_request = client.build_request(
        "POST", f"{OLLAMA_HOST}/api/chat", json=_ollama_payload(body)
    )

    # Connect before returning so failures get a real status code, not a broken stream.
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        logger.error("Ollama unreachable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"Ollama is not reachable at {OLLAMA_HOST}.",
                "hint": "Start it with `ollama serve`, then check `ollama list` for the model.",
            },
        )

    if upstream.status_code != 200:
        detail = (await upstream.aread()).decode("utf-8", "replace").strip()
        await upstream.aclose()
        logger.error("Ollama returned %s: %s", upstream.status_code, detail[:300])
        hint = None
        if upstream.status_code == 404:
            hint = f"Pull the model with `ollama pull {OLLAMA_MODEL}`."
        return JSONResponse(
            status_code=502,
            content={
                "detail": detail or f"Ollama returned HTTP {upstream.status_code}.",
                **({"hint": hint} if hint else {}),
            },
        )

    started = time.perf_counter()

    async def stream():
        buffer = ""
        sent = 0
        try:
            async for line in upstream.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                error = chunk.get("error")
                if error:
                    yield _ndjson({"type": "error", "message": str(error)})
                    return
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    buffer += piece
                    sent += len(piece)
                    yield _ndjson({"type": "delta", "text": piece})
                if chunk.get("done"):
                    break
            yield _ndjson({"type": "done", "text": normalise(buffer)})
            logger.info("Reply %d chars in %.2fs", sent, time.perf_counter() - started)
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health(request: Request):
    client: httpx.AsyncClient = request.app.state.http
    try:
        response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        models = [m.get("name", "") for m in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Health check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ollama": False,
                "hint": f"Ollama is not reachable at {OLLAMA_HOST}.",
            },
        )

    base = OLLAMA_MODEL.split(":")[0]
    available = any(name == OLLAMA_MODEL or name.split(":")[0] == base for name in models)
    return {
        "status": "healthy" if available else "degraded",
        "ollama": True,
        "model": OLLAMA_MODEL,
        "model_available": available,
        "models": models,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
    )
