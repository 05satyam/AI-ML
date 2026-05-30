"""FastAPI app: stateless, horizontally scalable, with health/readiness probes.

The pipeline holds in-process state (memory, caches, vector store) only to keep the
demo single-process. In production those move to Redis / a vector DB so any pod can
serve any request and the deployment is truly stateless + autoscalable.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .guardrails import GuardrailError
from .pipeline import RAGPipeline
from .schemas import ChatRequest, ChatResponse

pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    yield


app = FastAPI(title="SoFi RAG reference service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, bool]:
    """Readiness: dependencies (vector store) are loaded before we take traffic."""
    return {"ready": pipeline._retriever is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        return await pipeline.chat_turn(req.session_id, req.message)
    except GuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
