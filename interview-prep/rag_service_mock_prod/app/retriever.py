"""Hybrid retrieval = dense (vector) + sparse (keyword) fused, then reranked.

Why hybrid: vectors capture semantics ("cost" ~ "price"); keyword catches exact
terms / rare tokens / product names that embeddings sometimes miss. Fusing both is
a cheap, reliable quality win — one of the first RAG optimizations to reach for.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .config import settings
from .llm import Embedder
from .vectorstore import InMemoryVectorStore, Match


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, store: InMemoryVectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        texts = self.store.texts_by_id
        self._docs = {doc_id: _tokenize(t) for doc_id, t in texts.items()}
        n = len(self._docs) or 1
        df: Counter[str] = Counter()
        for toks in self._docs.values():
            for term in set(toks):
                df[term] += 1
        self._idf = {term: math.log(1 + n / (1 + c)) for term, c in df.items()}

    def _keyword_scores(self, query: str) -> dict[str, float]:
        q_terms = _tokenize(query)
        scores: dict[str, float] = {}
        for doc_id, toks in self._docs.items():
            tf = Counter(toks)
            scores[doc_id] = sum(tf[t] * self._idf.get(t, 0.0) for t in q_terms)
        return scores

    @staticmethod
    def _minmax(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return {}
        lo, hi = min(d.values()), max(d.values())
        if hi - lo < 1e-9:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    async def retrieve(self, query: str) -> list[Match]:
        # Dense
        q_emb = (await self.embedder.embed([query]))[0]
        dense = self.store.search(q_emb, top_k=settings.rerank_top_n)
        dense_scores = self._minmax({m.doc_id: m.score for m in dense})

        # Sparse
        sparse_scores = self._minmax(self._keyword_scores(query))

        # Fuse
        alpha = settings.hybrid_alpha
        texts = self.store.texts_by_id
        fused: dict[str, float] = {}
        for doc_id in texts:
            fused[doc_id] = alpha * dense_scores.get(doc_id, 0.0) + (1 - alpha) * sparse_scores.get(
                doc_id, 0.0
            )

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        ranked = self._rerank(query, ranked)
        return [Match(doc_id, score, texts[doc_id]) for doc_id, score in ranked[: settings.top_k]]

    def _rerank(self, query: str, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Hook for a cross-encoder reranker (e.g., bge-reranker / Cohere rerank).

        Cross-encoders score (query, doc) jointly and are far more accurate than
        bi-encoder cosine, at higher cost — so you rerank only the top-N candidates.
        Here we no-op (already fused). In prod, replace with a model call.
        """
        return ranked
