"""Retrieval + generation metrics.

Retrieval:
  - recall@k: did we retrieve the relevant doc(s) in the top k?
  - MRR: how high up was the first relevant doc? (rewards ranking quality)
Generation:
  - faithfulness/groundedness: is the answer supported by retrieved context?
    Here approximated by lexical overlap; in prod use an LLM-as-judge or NLI model.
"""

from __future__ import annotations

import re


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    hits = sum(1 for r in relevant_ids if r in top)
    return hits / max(1, len(relevant_ids))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def faithfulness(answer: str, context: str) -> float:
    """Fraction of answer content-words supported by the context.
    Crude but directionally useful and fully offline."""
    a, c = _tokens(answer), _tokens(context)
    stop = {"the", "a", "an", "is", "are", "of", "to", "and", "based", "on", "for", "your"}
    a -= stop
    if not a:
        return 1.0
    return len(a & c) / len(a)
