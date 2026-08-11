# Interview Prep

Coding drills and **GenAI interview question banks** for AI / ML / LLM roles.

## Highlight: GenAI interview questions (topic-wise)

Start here → **[`genai-interview-questions/`](genai-interview-questions/)**

| Topic | File | What it covers |
|-------|------|----------------|
| Basics of AI / ML | [basics-of-ai.md](genai-interview-questions/basics-of-ai.md) | Foundations checklist + links to existing drills |
| LLMs | [llm.md](genai-interview-questions/llm.md) | Transformers, KV cache, fine-tuning, serving |
| RAG | [rag.md](genai-interview-questions/rag.md) | Chunking, hybrid search, rerank, evals, prod |
| AI Agents | [ai-agents.md](genai-interview-questions/ai-agents.md) | ReAct, planning, multi-agent, reliability |
| AI Agent Tools | [ai-agent-tools.md](genai-interview-questions/ai-agent-tools.md) | Function calling, schemas, MCP, safety |

Each file is ordered **Basic → Intermediate → Advanced**, with short cue notes and **external references** from recent interview guides.

Index + full reference list: [`genai-interview-questions/README.md`](genai-interview-questions/README.md)

### Related drills already in this folder

- [ai_qna_drill.md](ai_qna_drill.md) — rapid-fire answers
- [interview-expereince/AI-ML-QnA.md](interview-expereince/AI-ML-QnA.md) — longer experience-style Q&A
- [ai_agents_memory_types.md](ai_agents_memory_types.md) · [Contexual And GPT Embeddings.md](Contexual%20And%20GPT%20Embeddings.md)

---

## Notebooks

GitHub's built-in `.ipynb` preview sometimes shows **"Notebook not found"** (GitHub renderer issue).  
Use **nbviewer** to read them in the browser:

| Notebook | nbviewer |
|----------|----------|
| [basic_datastructure.ipynb](basic_datastructure.ipynb) | [Open in nbviewer](https://nbviewer.org/github/05satyam/AI-ML/blob/main/interview-prep/basic_datastructure.ipynb) |
| [senior-ai-engineer.ipynb](senior-ai-engineer.ipynb) | [Open in nbviewer](https://nbviewer.org/github/05satyam/AI-ML/blob/main/interview-prep/senior-ai-engineer.ipynb) |
| [dspy_hands_on.ipynb](dspy_hands_on/dspy_hands_on.ipynb) | [Open in nbviewer](https://nbviewer.org/github/05satyam/AI-ML/blob/main/interview-prep/dspy_hands_on/dspy_hands_on.ipynb) |

Or open locally in VS Code / Jupyter / Google Colab.

## Topics

- `genai-interview-questions/` — **topic-wise GenAI interview questions (basic → advanced)**
- `basic_datastructure.ipynb` — LRU cache, top-K, merge intervals, sliding window, rate limiter
- `senior-ai-engineer.ipynb` — embeddings, TF-IDF, chunking, RAG retrieval pipeline
- `dspy_hands_on/` — Colab-ready DSPy: signatures, modules, metrics, BootstrapFewShot, ReAct tools
- `rag_service_mock_prod/` — mock production RAG service sketch

## DSPy hands-on

Folder: [`dspy_hands_on/`](dspy_hands_on/)

**What DSPy is for:** build LLM apps as evaluable, optimizable programs (signatures + modules + metrics) instead of hand-tuned prompt strings.

**Run:** set `OPENAI_API_KEY` in the notebook env cell (or Colab Secrets), then run top to bottom. Do not commit real API keys.
