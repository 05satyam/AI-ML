# RAG — Interview Questions

Production-focused RAG questions from recent GenAI interview guides (2025–2026). Interviewers want pipelines you’ve broken and fixed, not “PDF → vector DB → LangChain.”

## Basic

1. What is RAG and what problem does it solve vs a standalone LLM?
2. Draw the two phases: ingestion vs query-time.
3. What is a chunk? Why not embed whole documents?
4. What is an embedding model’s job in RAG?
5. Dense retrieval vs keyword (BM25) — when does each win?
6. What is top-k retrieval?
7. Why include citations / source snippets in the answer?
8. How do you prompt the model to stay grounded in context?
9. Freshness: how does RAG help with knowledge that changes after training?
10. Name failure modes of naive RAG (bad chunks, wrong docs, ignored context).

**Cue notes:** RAG = retrieve relevant context → condition generation. Citations + “answer only from context / else I don’t know” are baseline guardrails.

## Intermediate

11. Chunking strategies: fixed size, overlap, semantic / structure-aware — tradeoffs.
12. Metadata filtering (ACL, tenant, date, doc type) — where does it run?
13. Hybrid search + Reciprocal Rank Fusion (RRF) — why combine dense + sparse?
14. Cross-encoder reranking — retrieve many, keep few; latency cost?
15. Query rewriting / multi-query / HyDE — when to use.
16. Conversational RAG: how do you handle “how much does it cost?” follow-ups?
17. Lost-in-the-middle — how do you place / select context?
18. Embedding model choice and versioning — what breaks if you change models?
19. Index update strategies: batch rebuild vs incremental upserts.
20. Multimodal / table / PDF parsing issues that kill retrieval quality.

**Cue notes:** Cheap fixes first: hybrid → rerank → rewrite → better chunking. Always measure with an eval set.

## Advanced / production

21. Metrics: Recall@k, Precision@k, MRR, nDCG for retrieval — define each.
22. Generation metrics: faithfulness / groundedness, answer relevance, citation accuracy.
23. RAGAS / TruLens / Phoenix-style eval — what would you put in CI?
24. Design end-to-end RAG for a 10M-doc enterprise corpus with ACLs.
25. Latency budget: retrieve + rerank + generate under ~3s — where do you spend ms?
26. Semantic cache for repeated questions — risks (stale / wrong cache hits)?
27. Corrective / Self-RAG / Agentic RAG — when is retrieve-then-generate not enough?
28. How do you detect retrieval regressions after a corpus or model change?
29. Security: prompt injection via retrieved docs; how do you harden?
30. Cost control: token budgets, smaller rewrite models, cascade rerankers.

**Cue notes:** Senior answers include eval harness + ownership of failure modes (stale index, ACL leaks, noisy chunks), not framework names.

## System design prompts

- Design a policy / support RAG with citations and “I don’t know” behavior.
- Your faithfulness score dropped 15% this week — debug plan.
- Compare pgvector vs managed vector DB for your constraints (ops, filters, scale).

## Hands-on in this repo

- [senior-ai-engineer.ipynb](../senior-ai-engineer.ipynb)
- [rag_service_mock_prod](../rag_service_mock_prod/)
- [ai_qna_drill.md](../ai_qna_drill.md) (Q1–Q2, Q12–Q13)

## References

- [RAG Interview Questions 2026 — gitGood.dev](https://gitgood.dev/blog/complete-guide-rag-interview-questions-2026)
- [Enterprise RAG Interview Guide 2026](https://maywise.in/blog/enterprise-rag-interview-guide-2026/)
- [RAG Pipeline Design Interview Guide — CalibreOS](https://www.calibreos.com/blog/rag-pipeline-design-interview-guide)
- [Designing RAG at Scale — PracHub](https://prachub.com/resources/designing-rag-architecture-at-scale-an-ai-engineering-interview-guide)
- [Hiring RAG Engineers 2026 — Kore1](https://www.kore1.com/hire-rag-engineers-2026/)
