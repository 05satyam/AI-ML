# Interview Prep

Coding drills and AI-engineering practice notebooks.

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

- `basic_datastructure.ipynb` — LRU cache, top-K, merge intervals, sliding window, rate limiter
- `senior-ai-engineer.ipynb` — embeddings, TF-IDF, chunking, RAG retrieval pipeline
- `dspy_hands_on/` — Colab-ready DSPy hands-on: signatures, modules, metrics, `BootstrapFewShot`, ReAct tools

## DSPy hands-on

Folder: [`dspy_hands_on/`](dspy_hands_on/)

**What DSPy is for:** build LLM apps as evaluable, optimizable programs (signatures + modules + metrics) instead of hand-tuned prompt strings.

**Notebook covers:**
1. LM setup (OpenAI / Gemini) via env key
2. String + class-based signatures
3. `Predict` / `ChainOfThought` / custom `Module`
4. `Example`s + metric + `Evaluate`
5. `BootstrapFewShot` optimization (before/after)
6. Save / load compiled programs
7. Optional `ReAct` tool use

**Run:** set `OPENAI_API_KEY` in the notebook env cell (or Colab Secrets), then run top to bottom. Do not commit real API keys.
