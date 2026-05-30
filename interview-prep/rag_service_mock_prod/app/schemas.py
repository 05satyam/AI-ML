"""API contract. Keep request/response models explicit and versionable."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable id for conversational memory")
    message: str = Field(..., min_length=1)


class Citation(BaseModel):
    doc_id: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    rewritten_query: str
    cache_hit: bool
    trace_id: str
    degraded: bool = False  # True if we served a fallback (e.g., LLM unavailable)
