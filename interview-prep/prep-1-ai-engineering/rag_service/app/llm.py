"""LLM + embedding clients.

Two implementations behind one interface:
- Real OpenAI client (used when RAG_OPENAI_API_KEY is set).
- Deterministic *fake* client (default) so the service runs anywhere with no
  secrets, no network, and reproducible outputs for tests/CI.

Talking point: programming to an interface like this is what makes a system
provider-agnostic and testable. Swapping Claude/Gemini/Bedrock is a new subclass.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .config import settings

EMBED_DIM = 256


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


def _hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic bag-of-words hashing embedding. Not semantic-quality, but
    stable and dependency-free — good enough to demo retrieval ranking."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]


class FakeChatModel:
    """Extractive 'LLM': stitches retrieved context into a grounded answer.

    It only uses provided context, which conveniently makes faithfulness high —
    handy for demonstrating the eval harness deterministically.
    """

    async def complete(self, system: str, user: str) -> str:
        marker = "CONTEXT:\n"
        context = user.split(marker, 1)[1] if marker in user else ""
        first = next((ln.strip("- ").strip() for ln in context.splitlines() if ln.strip()), "")
        if not first:
            return "I don't have enough information to answer that."
        return f"Based on the documentation: {first}"


class OpenAIEmbedder:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=settings.embed_model, input=texts)
        return [d.embedding for d in resp.data]


class OpenAIChatModel:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""


def build_clients() -> tuple[Embedder, ChatModel]:
    if settings.openai_api_key:
        return OpenAIEmbedder(), OpenAIChatModel()
    return FakeEmbedder(), FakeChatModel()
