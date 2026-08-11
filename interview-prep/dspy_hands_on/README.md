# DSPy Hands-On

Colab-ready notebook for learning DSPy: declarative LLM programs with signatures, modules, metrics, and optimizers.

## Quick start

1. Open [`dspy_hands_on.ipynb`](dspy_hands_on.ipynb) in Colab, VS Code, or Jupyter
2. Paste your key in the env cell: `os.environ["OPENAI_API_KEY"] = "sk-..."`
3. Run cells top to bottom

Optional: set `PROVIDER = "gemini"` and `GOOGLE_API_KEY` for Gemini.

## What you'll learn

| Concept | In this notebook |
|---------|------------------|
| Signature | Task I/O contract (`question -> answer`) |
| Module | `Predict`, `ChainOfThought`, custom multi-step, `ReAct` |
| Metric + Evaluate | Score predictions on a tiny FAQ/QA set |
| Optimizer | `BootstrapFewShot` few-shot demo compilation |
| Persist | `save` / `load` optimized program state |

Sample compiled state: [`short_answer_optimized.json`](short_answer_optimized.json)

## Docs

- https://dspy.ai/
