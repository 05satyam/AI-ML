# Production-style RAG service (interview reference)

A compact but realistic Retrieval-Augmented Generation service you can **read, run, and reason about out loud** in an interview. It deliberately shows the things SoFi's JD and your HM round asked about:

- Retrieval (hybrid: vector + keyword), reranking hook, and grounded generation
- **Conversational memory** + **follow-up query rewriting** (standalone-question reformulation)
- **Optimization**: semantic response cache, embedding cache, async I/O, batching
- **Reliability**: timeouts, retries w/ backoff, circuit breaker, graceful degradation, token/cost guardrails
- **Observability**: structured tracing of every step (retrieve → rerank → generate)
- **Evaluation**: retrieval + faithfulness metrics you can run offline
- **Deployment**: Dockerfile + Kubernetes manifests (Deployment, HPA, probes, resources)

It runs with **zero external services** by default: an in-memory cosine vector store and a deterministic "fake" LLM/embedder are used when no API key is set. Swap in real providers behind the same interfaces.

## Run it

```bash
cd interview-prep/sofi-staff-ai/rag_service
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
# then:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"s1","message":"What is SoFi Plus?"}' | python -m json.tool
# follow-up that needs rewriting:
curl -s -X POST localhost:8000/chat -H 'content-type: application/json' \
  -d '{"session_id":"s1","message":"how much does it cost?"}' | python -m json.tool
```

## Offline eval

```bash
python -m eval.run_eval
```

## How to talk about it (the architecture in one breath)

> Requests hit a stateless FastAPI pod. We load session memory from Redis (here: in-process),
> rewrite the user's follow-up into a standalone query using recent history, check a semantic
> cache, then do hybrid retrieval (vector + BM25-ish keyword) and rerank. We build a grounded
> prompt with citations, call the LLM behind a circuit breaker with timeout + retries, persist
> memory, and emit a trace for every step. Vector DB and embedding service scale independently;
> the API autoscales on latency/QPS via an HPA.

## File map

```
app/
  main.py          FastAPI app, routes, wiring
  config.py        Settings (env-driven)
  schemas.py       Pydantic request/response models
  memory.py        Conversation memory + follow-up query rewriting
  retriever.py     Hybrid retrieval + reranking hook
  vectorstore.py   In-memory cosine store (swap for pgvector/Pinecone)
  llm.py           LLM + embeddings clients (real or fake), with resilience
  cache.py         Semantic response cache + embedding cache
  resilience.py    Timeout, retry/backoff, circuit breaker
  tracing.py       Lightweight structured tracing
  guardrails.py    Token/cost budgets + simple input checks
  pipeline.py      Orchestrates the full RAG flow
  knowledge.py     Tiny seed corpus
eval/
  dataset.py       Q/A eval set
  metrics.py       recall@k, MRR, faithfulness (groundedness)
  run_eval.py      Runs the eval harness
deploy/
  Dockerfile
  k8s/             Deployment, Service, HPA, ConfigMap
```
