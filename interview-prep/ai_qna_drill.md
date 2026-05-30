# Rapid-fire Q&A drill (self-test)

Cover the answers. Say yours out loud in <60s, then check. Target: crisp, trade-off-aware.

---

### Q1. Walk me through a RAG pipeline end to end.
**A.** Ingest → chunk (structure-aware, ~200–500 tok, overlap) → embed → index in a
vector DB with metadata. At query: embed query → retrieve top-N → rerank → clamp to a
token budget → grounded prompt ("answer only from context") with citations → generate
→ return answer + sources. Wrap external calls with timeout/retry/breaker; trace every
step; evaluate offline.

### Q2. Retrieval quality is poor. How do you improve it, cheapest first?
**A.** (1) Hybrid dense+sparse search. (2) Cross-encoder reranker on top-N. (3) Query
rewriting / multi-query / HyDE. (4) Better chunking + metadata pre-filtering. (5) Always
measure with an eval set — don't guess. Then consider embedding model upgrade / fine-tune.

### Q3. How do you handle follow-up questions like "how much does it cost?"
**A.** Query rewriting: reformulate the follow-up into a standalone query using recent
conversation history (LLM or heuristic), THEN retrieve. Keep conversation memory
separate from the KB.

### Q4. Types of conversational memory?
**A.** Short-term/working (windowed turns), summary (compress old turns), long-term
semantic (facts/preferences in a vector store), episodic (past events), procedural
(rules/how-to). Bound tokens with windowing + summarization; store in Redis w/ TTL by
session; mind PII.

### Q5. How do you reduce LLM latency?
**A.** Cache embeddings + responses (semantic cache), async + batch, smaller models for
routing/rewriting, stream tokens, parallelize retrieval/rerank, cap context tokens.

### Q6. How do you reduce LLM cost?
**A.** Semantic cache to kill duplicate questions, model routing (cheap default,
escalate when needed), context/prompt compression + token budget, cap agent steps,
right-size top-k.

### Q7. How do you deploy a RAG service for scale?
**A.** Stateless FastAPI pods behind a LB, autoscaled via HPA; externalize state
(memory/cache → Redis, docs/embeddings → vector DB); scale embedding/rerank/ingestion
independently; async queue for ingestion; resilience (timeout/retry/breaker/fallback) +
rate limits + cost guardrails; full tracing + evals in CI.

### Q8. Kubernetes: liveness vs readiness probe?
**A.** Liveness = "is the process wedged? restart it." Readiness = "is it ready to serve?
don't route traffic until deps (vector store/Redis) are loaded." Different endpoints;
a slow dependency should fail readiness, not trigger restarts.

### Q9. What do you autoscale a RAG API on?
**A.** CPU is the simple default, but an LLM-bound service spends time waiting on the
model, so CPU under-reflects load. Prefer custom metrics: p95 latency or in-flight
requests/QPS per pod via Prometheus adapter. Scale up fast, down slow (avoid flapping).

### Q10. requests vs limits in k8s?
**A.** Requests = guaranteed resources the scheduler reserves (placement). Limits = hard
cap (blast radius); exceeding memory limit = OOMKill, CPU limit = throttle. Set both;
right-size from real usage.

### Q11. How do you make an LLM call reliable?
**A.** Timeout (never hang), retry with exponential backoff + jitter (transient 429/5xx),
circuit breaker (stop hammering a dead dep, fail fast), and graceful degradation
(serve retrieved snippets / cached answer when the model is down).

### Q12. How do you evaluate a RAG system?
**A.** Retrieval: recall@k, precision@k, MRR, nDCG, context precision/recall.
Generation: faithfulness/groundedness, answer relevance, correctness vs golden, citation
accuracy. Version a golden set from real traces; run in CI; LLM-as-judge for open-ended;
online A/B + feedback + drift monitoring.

### Q13. How do you stop hallucinations?
**A.** Ground strictly ("answer only from context, else say you don't know"), pass
citations, rerank for better context, measure faithfulness, add output guardrails,
human-in-the-loop for high-stakes. Lower temperature for factual tasks.

### Q14. When would you NOT use an LLM?
**A.** When correctness must be guaranteed/auditable, latency/cost is critical, the task
is deterministic, or failure is expensive — use rules/classic ML. LLM where flexibility
and natural language understanding pay off. (Big one for a risk org.)

### Q15. Agentic patterns you know?
**A.** Tool use, ReAct (reason+act), plan-and-execute, reflection/reflexion, router/
orchestrator. Always bound steps + cost, validate tool inputs, add HITL checkpoints.
Use LangGraph for stateful, observable, resumable flows.

### Q16. What is context engineering?
**A.** Deliberately designing what enters the model's context window — instructions,
retrieved context, tool schemas, memory, few-shot — and how it's structured, ordered,
and trimmed to maximize reliability within token/cost limits.

### Q17. How do you make AI auditable in a regulated/risk setting?
**A.** Trace + log every decision (inputs, retrieved docs, prompt, output, cost),
deterministic guardrails, explainable outputs with citations, HITL for high-stakes,
PII handling, gradual rollout (shadow→assist→automate), versioned prompts/models/evals.

### Q18. Vector DB choice + scaling?
**A.** pgvector (start simple, transactional), Pinecone/Weaviate/Qdrant (managed scale).
Scale via read replicas (QPS), sharding/namespaces (by tenant), ANN indexes (HNSW/IVF)
for sub-linear search; separate from the stateless API.

### Q19. Chunking strategy?
**A.** Structure-aware (headings/sentences), ~200–500 tokens, 10–20% overlap, attach
metadata. Validate via retrieval eval — chunking is the highest-leverage knob and is
data-dependent, so measure.

### Q20. Semantic cache — risks?
**A.** Stale answers if the KB changes, and false hits if the similarity threshold is too
loose. Mitigate: key/version by KB version, TTL, conservative threshold, exclude
personalized/session-specific queries.
