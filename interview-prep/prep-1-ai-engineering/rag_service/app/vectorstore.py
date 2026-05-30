"""In-memory cosine vector store.

This stands in for pgvector / Pinecone / Weaviate. The interface (upsert + search)
is intentionally the same shape you'd implement against a real DB, so "swap the
store" is a localized change.

Scaling talking points (when asked): in prod the vector DB is a separate, stateful
service scaled independently from the stateless API — sharding by tenant/namespace,
read replicas for query load, ANN indexes (HNSW/IVF) for sub-linear search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    doc_id: str
    score: float
    text: str


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None  # (n_docs, dim), L2-normalized

    def upsert(self, ids: list[str], texts: list[str], embeddings: list[list[float]]) -> None:
        mat = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self._ids = list(ids)
        self._texts = list(texts)
        self._matrix = mat

    def search(self, query_embedding: list[float], top_k: int) -> list[Match]:
        if self._matrix is None:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        n = np.linalg.norm(q) or 1.0
        q = q / n
        scores = self._matrix @ q  # cosine since both normalized
        order = np.argsort(-scores)[:top_k]
        return [Match(self._ids[i], float(scores[i]), self._texts[i]) for i in order]

    @property
    def texts_by_id(self) -> dict[str, str]:
        return dict(zip(self._ids, self._texts))
