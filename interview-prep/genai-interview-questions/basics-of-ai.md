# Basics of AI / ML — Interview Questions

Foundations interviewers expect before GenAI deep-dives. **Much of this already exists in-repo** — use this file as a checklist + pointer.

## Existing materials in this repo

| Resource | Use for |
|----------|---------|
| [ai_qna_drill.md](../ai_qna_drill.md) | Rapid-fire answers (RAG, latency, k8s, eval) |
| [interview-expereince/AI-ML-QnA.md](../interview-expereince/AI-ML-QnA.md) | Longer GenAI experience Q&A |
| [Contexual And GPT Embeddings.md](../Contexual%20And%20GPT%20Embeddings.md) | Embedding intuition |
| [standford_transformer_llm_cheatsheet.pdf](../standford_transformer_llm_cheatsheet.pdf) | Transformer cheat sheet |
| [basic_datastructure.ipynb](../basic_datastructure.ipynb) | Coding round patterns |

---

## Basic

1. Supervised vs unsupervised vs reinforcement learning — give one GenAI-related example each.
2. Bias–variance tradeoff — how does it show up in fine-tuning an LLM?
3. Overfitting vs underfitting — how do you detect each on a classification / retrieval task?
4. Precision, recall, F1, ROC-AUC — when do you prefer precision over recall?
5. What is an embedding? Difference between word2vec-style and contextual (transformer) embeddings.
6. Train / validation / test split — why not tune on the test set?
7. Classification vs regression vs generation — which metrics fit each?
8. What does “gradient descent” optimize, and what is a learning rate?
9. Batch vs mini-batch vs online training — tradeoffs.
10. What is regularization (L1/L2/dropout) and why does it help generalization?

**Cue notes:** Prefer definitions + one concrete failure mode. For embeddings: static vectors vs context-dependent hidden states.

## Intermediate

11. Cross-entropy loss for classification — what does it penalize?
12. Softmax temperature — effect on distribution sharpness.
13. Feature scaling / normalization — when does it matter for neural nets vs tree models?
14. Bagging vs boosting — one-line difference; name one algorithm each.
15. Curse of dimensionality — why nearest neighbors degrade in high-d embedding spaces.
16. Cosine similarity vs dot product vs Euclidean distance for embeddings — when to use which?
17. What is tokenization at a high level (BPE / WordPiece intuition)?
18. Transfer learning vs fine-tuning vs prompt-only adaptation.
19. Offline vs online metrics for a search / RAG product.
20. Data leakage — give an NLP/RAG example (e.g., test docs in the index).

**Cue notes:** Cosine = direction; dot product mixes magnitude. Leakage in RAG = evaluating on docs that were also in the retrieval corpus without a held-out set.

## Advanced / system-minded

21. Design an evaluation plan for a new ranking / retrieval feature before shipping.
22. How would you detect and mitigate training data / distribution drift in production ML?
23. Classic ML vs LLM for a task — when would you **not** use an LLM?
24. Explain calibration of model confidence and why it matters for routing to humans.
25. Sketch a simple A/B test for an AI feature (primary metric, guardrails, sample size intuition).

**Cue notes:** Interviewers want tradeoffs + measurement, not buzzwords. Tie answers to latency, cost, and failure cost.

## References

- In-repo drills linked above
- [AI Engineer Interview Questions (real loops summary)](https://adilshamim8.medium.com/every-ai-engineer-interview-question-you-need-to-know-in-2026-from-100-real-interviews-b5b7ae4b961a)
