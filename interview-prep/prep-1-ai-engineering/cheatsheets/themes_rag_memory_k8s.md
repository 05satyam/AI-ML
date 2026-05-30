# Deep technical themes (the HM-round syllabus)

Your HM asked: **RAG, optimization, follow-ups, conversational memory, RAG deployment,
Kubernetes scalability.** Be able to whiteboard each in 2–3 min. Cross-references point
to runnable code in `../rag_service/`.

---

## 1. RAG fundamentals

**Pipeline:** ingest → chunk → embed → index (vector DB) → at query time: embed query →
retrieve → (rerank) → build grounded prompt with citations → generate → return + cite.

**Ingestion/chunking**
- Chunk by structure (headings/sentences) not blind fixed size; ~200–500 tokens with
  overlap (~10–20%). Too big = diluted retrieval + cost; too small = lost context.
- Store metadata (source, section, timestamp, ACL/tenant) for filtering + auditability.
- Re-embed on KB change; version the index so caches can be invalidated.

**Embeddings**
- Bi-encoder (dense) for fast ANN retrieval. Pick by domain + cost + dimensionality.
- Normalize → cosine similarity. (`rag_service/app/vectorstore.py`)

**Generation**
- System prompt: "answer ONLY from context; say you don't know otherwise." → reduces
  hallucination. Always pass citations. (`rag_service/app/pipeline.py`)

---

## 2. Retrieval quality / optimization

Order to reach for improvements (cheap → expensive):
1. **Hybrid search** (dense + sparse/BM25), fuse scores. Catches exact terms +
   semantics. (`rag_service/app/retriever.py`)
2. **Reranking** with a cross-encoder (bge-reranker / Cohere rerank) on top-N → top-k.
   Big precision win; cost-bounded by only reranking candidates.
3. **Query transformation**: rewriting, HyDE, multi-query (fan-out + dedupe).
4. **Better chunking / metadata filtering** (pre-filter by tenant/recency).
5. **Eval-driven iteration** — never guess; measure (§6).

**Latency optimization**
- Cache embeddings (pure function of model+text) and responses (semantic cache).
- Async I/O, batch embeddings, parallelize retrieval + reranking where possible.
- Smaller/faster models for routing & rewriting; big model only for final answer.
- Stream tokens to the user (perceived latency).

**Cost optimization**
- Semantic cache (kill duplicate questions), prompt compression / context clamping
  (`guardrails.clamp_context`), model routing, cap context tokens, cap agent steps.

---

## 3. Follow-ups & conversational memory

**The core problem:** "how much does it cost?" is meaningless to a retriever without
context. **Fix = query rewriting / contextualization**: rewrite the follow-up into a
standalone query using recent history, THEN retrieve.
(`rag_service/app/memory.py::QueryRewriter`)

**Memory types** (see your `interview-experience/ai_agents_memory_types.md`)
- **Short-term / working:** rolling window of last N turns in the prompt.
- **Summary memory:** LLM-summarize older turns to fit the window (long chats).
- **Long-term / semantic:** store facts/preferences in a vector store, retrieve when
  relevant (user profile, prior tickets).
- **Episodic:** past interactions/events; **procedural:** learned how-to/rules.

**Design rules**
- Keep **conversation memory separate from the knowledge base** (different store,
  lifecycle, TTL). Conversation state → Redis keyed by `session_id`, with a TTL.
- Bound it: window + summarization to control tokens/cost.
- For risk: log memory contents for audit; be careful with **PII retention**.

---

## 4. Agentic systems (JD core)

- **Patterns:** tool use, ReAct (reason+act), plan-and-execute, reflection/reflexion,
  router. (You have notebooks for all of these.)
- **Control:** ALWAYS bound steps + cost; add stop conditions; validate tool inputs.
- **Orchestration:** LangGraph (state machine of nodes/edges) when flows get nontrivial
  — gives you persistence, retries, human-in-the-loop checkpoints, observable state.
- **Context engineering:** what goes in the prompt window — instructions, retrieved
  context, tool schemas, memory, few-shot — and how you structure/trim it for
  reliability. This is explicitly in the JD.

---

## 5. Deployment & Kubernetes scalability

**Topology**
- **Stateless API** (FastAPI) → horizontally scalable behind a LB/ingress.
- **State externalized:** conversation memory + caches → Redis; embeddings/docs →
  vector DB (managed or its own stateful set). API pods stay disposable.
- Heavy/independent pieces scale separately: embedding service, reranker, ingestion
  workers (queue-based), the LLM (managed API or self-hosted GPU pool).

**Kubernetes specifics** (`rag_service/deploy/k8s/`)
- **Deployment** with replicas; **Service** for stable in-cluster addressing.
- **Probes:** liveness (`/health`, restart wedged pods) vs readiness (`/ready`, don't
  route until deps loaded). Different endpoints on purpose.
- **Resources:** requests (scheduling guarantee) vs limits (blast-radius cap). Right-size
  to avoid OOM kills / CPU throttling.
- **HPA:** autoscale on CPU is the default, but an LLM service is I/O-bound waiting on
  the model — prefer **custom metrics** (p95 latency, in-flight requests/QPS per pod)
  via Prometheus adapter. Tune scale-up fast / scale-down slow to avoid flapping.
- **GPU workloads:** separate node pool, GPU resource requests, often a separate
  Deployment; consider KEDA for queue-depth scaling.
- **Rollouts:** rolling update / canary; PodDisruptionBudget; HPA + cluster autoscaler.
- **Secrets:** k8s Secret (not ConfigMap, not baked into image); least-privilege.

**Scaling the bottlenecks**
- Vector DB: replicas for read QPS, sharding/namespaces by tenant, ANN index (HNSW/IVF).
- LLM: the usual ceiling → caching, batching, smaller models, provider rate-limit
  handling, circuit breaker + fallback (`rag_service/app/resilience.py`).
- Ingestion: async workers + queue; never block the serving path.

**Reliability**
- Timeout + retry/backoff + circuit breaker around every external call.
- Graceful degradation (serve retrieved snippets if LLM down) — implemented in pipeline.
- Rate limiting, idempotency keys, token/cost guardrails, backpressure.

---

## 6. Evaluation & observability (JD core + risk requirement)

**Retrieval metrics:** recall@k, precision@k, MRR, nDCG, context precision/recall.
**Generation metrics:** faithfulness/groundedness, answer relevance, correctness
(vs golden), citation accuracy. (`rag_service/eval/`)

**How:** golden eval set (version-controlled, grown from real traces + failures);
run in **CI** so regressions fail the build. LLM-as-judge for open-ended, with care
(bias, cost). Online: A/B, thumbs feedback, drift monitoring.

**Observability:** trace every step (retrieve→rerank→generate) with timing + attrs
(`rag_service/app/tracing.py`); export to **Langfuse / LangSmith**. Log inputs,
retrieved doc ids, prompt, output, cost, latency — essential for **debugging + audit**
in a regulated/risk setting.

---

## 7. Risk / regulated framing (say this, it's the org's whole identity)
Explainability, auditability (full trace of every decision), human-in-the-loop for
high-stakes actions, PII handling/redaction, guardrails (input + output), deterministic
fallbacks where correctness must be guaranteed, gradual rollout (shadow → assist →
automate). Your `policy_gated_mcp_agent` is a concrete example to reference.
