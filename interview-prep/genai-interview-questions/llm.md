# LLMs & Transformers — Interview Questions

Basic → advanced questions common in 2025–2026 LLM / GenAI engineer loops (internals, fine-tuning, inference, serving, evals).

## Basic

1. What problem does the Transformer solve compared to RNNs/LSTMs?
2. Explain self-attention in plain language (Q, K, V).
3. Why divide attention scores by √d_k?
4. Multi-head attention — what does each head learn in principle?
5. Encoder-only vs decoder-only vs encoder–decoder — examples and use cases.
6. What is causal (autoregressive) masking and why do GPT-style models need it?
7. What is positional encoding / RoPE at a high level, and why are positions needed?
8. Tokenization: what is a subword tokenizer? Why not always characters or words?
9. Temperature, top-k, top-p — how do they change sampling?
10. Prompting vs fine-tuning vs RAG — when do you choose each?

**Cue notes:** Attention is all-pairs relevance; causal mask stops looking at future tokens. Prefer RAG for changing private knowledge; fine-tune for style/format/domain behavior.

## Intermediate

11. Walk through one forward pass of autoregressive decoding (prefill vs decode).
12. What is the KV cache? Why does it change decode complexity?
13. Rough formula for KV cache memory — what variables dominate?
14. MHA vs MQA vs GQA — impact on quality and KV memory.
15. What is teacher forcing during training?
16. Perplexity — what it measures and what it does **not** measure for product quality.
17. LoRA / QLoRA — what is adapted and why is it cheaper than full fine-tunes?
18. SFT vs preference tuning (RLHF / DPO) — what signal each uses.
19. Context window limits — “lost in the middle” and mitigation ideas.
20. Hallucination — types (factual, attribution) and mitigation levers without RAG.

**Cue notes:** Prefill = compute-bound; decode = often memory-bandwidth-bound. KV ≈ 2 × layers × kv_heads × head_dim × seq × batch × bytes.

## Advanced

21. Compute a back-of-envelope KV cache size for a given model config and context length.
22. What problem does PagedAttention (vLLM-style) solve?
23. Continuous batching — why static batching wastes GPU for LLM serving.
24. Speculative decoding — when it helps and when it does not.
25. Quantization (INT8 / INT4 / FP8) — tradeoffs for weights vs KV cache.
26. Tensor vs pipeline vs data parallelism for large-model serving — one sentence each.
27. How do you measure TTFT vs TPOT / tokens-per-second, and which matters for chat UX?
28. Design a model router: cheap default model vs escalate to a frontier model.
29. How would you evaluate an LLM feature with golden sets + LLM-as-judge + online metrics?
30. Guardrails: input/output filtering, tool allowlists, PII — where each sits in the stack.

**Cue notes:** Senior signal = numbers + bottlenecks (HBM, batch size, cache fragmentation), not just naming techniques.

## System design prompts (practice out loud)

- Design an LLM API gateway for 1k QPS with cost caps and abuse controls.
- Your p95 latency doubled overnight — how do you debug (model, prompt length, retrieval, cache, GPU saturation)?
- When would you host open-weights vs call a hosted API?

## References

- [LLM Engineer Interview Questions (Top 40)](https://www.interviewcoder.co/blog/llm-engineer-interview-questions)
- [LLM Interview Questions — MyEngineeringPath](https://myengineeringpath.dev/genai-engineer/llm-interview-questions/)
- [LLM Serving / Inference Interview Guide](https://www.calibreos.com/blog/genai-llm-serving-inference-interview-guide)
- [KV cache (OpenAI-style interview framing)](https://prachub.com/interview-questions/explain-kv-cache-in-transformer-inference)
- In-repo: [LLM architecture comparison](../comparison_of_major_llms_architectures(2017-2025)/) · [Stanford cheatsheet](../standford_transformer_llm_cheatsheet.pdf)
