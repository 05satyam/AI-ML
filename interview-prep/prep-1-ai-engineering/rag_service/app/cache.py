"""Caches: the cheapest, highest-leverage RAG optimization.

- EmbeddingCache: exact-match cache on text -> embedding. Embeddings are pure
  functions of (model, text), so caching is always safe and saves $ + latency.
- SemanticCache: returns a prior answer when a new query is *semantically* close
  to a previous one (cosine >= threshold). Turns repeat/near-duplicate questions
  into ~0-cost, ~0-latency responses. Trade-off: stale answers if the KB changes,
  so you key/version it by KB version and set a TTL.
"""

from __future__ import annotations

import numpy as np

from .config import settings
from .schemas import Citation


class EmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def get(self, text: str) -> list[float] | None:
        return self._store.get(text)

    def put(self, text: str, emb: list[float]) -> None:
        self._store[text] = emb


class SemanticCache:
    def __init__(self, threshold: float = settings.semantic_cache_threshold) -> None:
        self.threshold = threshold
        self._embs: list[np.ndarray] = []
        self._answers: list[tuple[str, list[Citation]]] = []

    def lookup(self, query_emb: list[float]) -> tuple[str, list[Citation]] | None:
        if not self._embs:
            return None
        q = np.asarray(query_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = [float(e @ q) for e in self._embs]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self._answers[best]
        return None

    def store(self, query_emb: list[float], answer: str, citations: list[Citation]) -> None:
        e = np.asarray(query_emb, dtype=np.float32)
        e = e / (np.linalg.norm(e) or 1.0)
        self._embs.append(e)
        self._answers.append((answer, citations))
