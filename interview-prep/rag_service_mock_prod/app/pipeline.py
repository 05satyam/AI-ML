"""Orchestrates the full RAG flow. This is the file to walk an interviewer through.

Flow:
  validate -> rewrite follow-up -> embed query (cached) -> semantic cache check
  -> hybrid retrieve + rerank -> clamp context to token budget -> grounded generate
  (timeout + retry + breaker) -> persist memory -> emit trace.

Every step is a trace span, and every failure mode degrades gracefully instead of
500-ing the user.
"""

from __future__ import annotations

from .cache import EmbeddingCache, SemanticCache
from .config import settings
from .guardrails import clamp_context, validate_input
from .knowledge import DOCUMENTS
from .llm import ChatModel, Embedder, build_clients
from .memory import MemoryStore, QueryRewriter
from .resilience import CircuitBreaker, CircuitOpenError, call_with_resilience
from .retriever import HybridRetriever
from .schemas import ChatResponse, Citation
from .tracing import new_trace
from .vectorstore import InMemoryVectorStore

SYSTEM_PROMPT = (
    "You are SoFi's support assistant. Answer ONLY using the provided context. "
    "If the context is insufficient, say you don't know. Be concise. Cite nothing "
    "outside the context."
)


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder: Embedder
        self.chat: ChatModel
        self.embedder, self.chat = build_clients()
        self.store = InMemoryVectorStore()
        self.memory = MemoryStore()
        self.rewriter = QueryRewriter(self.chat, use_llm=bool(settings.openai_api_key))
        self.embed_cache = EmbeddingCache()
        self.semantic_cache = SemanticCache()
        self.breaker = CircuitBreaker(
            settings.breaker_fail_threshold, settings.breaker_reset_s
        )
        self._retriever: HybridRetriever | None = None

    async def startup(self) -> None:
        ids = [d["id"] for d in DOCUMENTS]
        texts = [d["text"] for d in DOCUMENTS]
        embs = await self.embedder.embed(texts)
        self.store.upsert(ids, texts, embs)
        self._retriever = HybridRetriever(self.store, self.embedder)

    async def _embed_cached(self, text: str) -> list[float]:
        cached = self.embed_cache.get(text)
        if cached is not None:
            return cached
        emb = (await self.embedder.embed([text]))[0]
        self.embed_cache.put(text, emb)
        return emb

    async def chat_turn(self, session_id: str, message: str) -> ChatResponse:
        assert self._retriever is not None, "call startup() first"
        trace = new_trace()

        with trace.span("validate"):
            validate_input(message)

        with trace.span("rewrite_query"):
            rewritten = await self.rewriter.rewrite(session_id, message, self.memory)

        with trace.span("embed_query"):
            q_emb = await self._embed_cached(rewritten)

        with trace.span("semantic_cache_lookup") as s:
            cached = self.semantic_cache.lookup(q_emb)
            s.attrs["hit"] = cached is not None
        if cached is not None:
            answer, citations = cached
            self.memory.append(session_id, "user", message)
            self.memory.append(session_id, "assistant", answer)
            return ChatResponse(
                answer=answer,
                citations=citations,
                rewritten_query=rewritten,
                cache_hit=True,
                trace_id=trace.trace_id,
            )

        with trace.span("retrieve") as s:
            matches = await self._retriever.retrieve(rewritten)
            s.attrs["doc_ids"] = [m.doc_id for m in matches]

        citations = [
            Citation(doc_id=m.doc_id, score=round(m.score, 4), snippet=m.text[:160])
            for m in matches
        ]

        with trace.span("build_context"):
            snippets = clamp_context([m.text for m in matches])
            context_block = "\n".join(f"- {s}" for s in snippets)
            user_prompt = f"Question: {rewritten}\n\nCONTEXT:\n{context_block}"

        degraded = False
        with trace.span("generate") as s:
            try:
                answer = await call_with_resilience(
                    lambda: self.chat.complete(SYSTEM_PROMPT, user_prompt),
                    timeout_s=settings.llm_timeout_s,
                    max_retries=settings.llm_max_retries,
                    breaker=self.breaker,
                )
            except (CircuitOpenError, Exception):  # noqa: BLE001
                # Graceful degradation: return retrieved snippets directly so the
                # user still gets grounded info even when the LLM is down.
                degraded = True
                answer = self._extractive_fallback(snippets)
            s.attrs["degraded"] = degraded

        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "assistant", answer)
        if not degraded:
            self.semantic_cache.store(q_emb, answer, citations)

        return ChatResponse(
            answer=answer,
            citations=citations,
            rewritten_query=rewritten,
            cache_hit=False,
            trace_id=trace.trace_id,
            degraded=degraded,
        )

    @staticmethod
    def _extractive_fallback(snippets: list[str]) -> str:
        if not snippets:
            return "I'm temporarily unable to answer. Please try again shortly."
        return "Service is degraded; here is the most relevant info:\n" + snippets[0]
