# Simple Transformer: Learning Addition

A **from-scratch, educational** implementation of a tiny decoder-only Transformer that learns to add natural numbers.

Instead of treating addition as a regression problem, we frame it as **next-token prediction** on strings like:

```text
<bos>23+45=68<eos>
```

After training, the model can **autoregressively generate** the sum when you give it a prompt such as `23+45=`.

This project is designed so you can trace every piece of the Transformer pipeline:

| File | What you learn |
|------|----------------|
| `data.py` | Tokenization, vocabulary, causal LM targets |
| `model.py` | Embeddings, positional encoding, multi-head attention, FFN, causal mask |
| `train.py` | Training loop, masked loss, checkpointing |
| `inference.py` | Greedy decoding / generation |
| `learn_transformers_addition.ipynb` | Step-by-step walkthrough with diagrams and experiments |

---

## Why addition?

Addition is a classic toy task for sequence models because:

1. **Input and output are both sequences of symbols** (digits and operators).
2. **Carry propagation** requires looking at multiple earlier positions — attention is useful.
3. **Success is easy to verify** — exact match against `a + b`.
4. The task is small enough to train on a laptop in minutes.

Real LLMs (GPT, LLaMA, etc.) use the same core ideas at much larger scale.

---

## Quick start

```bash
cd demo_applications/simple_transformer_addition
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (~2-5 min on CPU for default settings)
python train.py

# Inference
python inference.py --a 27 --b 15
python inference.py --interactive

# Or open the notebook for the full guided tour
jupyter notebook learn_transformers_addition.ipynb
```

---

## Architecture (high level)

```text
Prompt:  <bos> 2 7 + 1 5 =
           │   │ │ │ │ │ │
           ▼   ▼ ▼ ▼ ▼ ▼ ▼
      [ Token Embeddings + Positional Encoding ]
                           │
                           ▼
              ┌────────────────────────┐
              │  Transformer Block ×N  │
              │  • Multi-Head Attention│
              │  • Feed-Forward (FFN)  │
              │  • Residual + LayerNorm│
              └────────────────────────┘
                           │
                           ▼
              [ Linear head → next token logits ]
                           │
                           ▼
              Generated: 4 2 <eos>
```

---

## Default hyperparameters

| Setting | Value | Notes |
|---------|-------|-------|
| Number range | 0–9 (single-digit sums) | Scale with `--max-number 99 --epochs 30 --d-model 128` |
| `d_model` | 96 | Hidden dimension |
| `n_heads` | 4 | Parallel attention heads |
| `n_layers` | 3 | Transformer blocks |
| Training examples | 5,000 | Synthetic random pairs |
| Epochs | 20 | Reaches ~100% on single-digit sums |

---

## How Transformers fit together (reading order)

1. **Tokenization** — characters → integer IDs (`data.py`)
2. **Embeddings** — IDs → dense vectors; add position info
3. **Self-attention** — each token gathers context from earlier tokens (causal mask)
4. **Feed-forward** — per-token non-linear transform
5. **Stack blocks** — repeat attention + FFN with residuals
6. **Language modeling head** — predict next token distribution
7. **Training** — minimize cross-entropy on shifted targets
8. **Inference** — append greedily sampled tokens one at a time

See the notebook for intuition, tensor shapes, and attention visualizations.

---

## Extending the demo

- Train on 3-digit numbers: `python train.py --max-number 999 --epochs 20 --d-model 128`
- Swap sinusoidal positions for learned embeddings in `model.py`
- Implement beam search in `inference.py`
- Plot attention weights from `MultiHeadSelfAttention` (hook or return weights)

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT training code
- Olah, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
