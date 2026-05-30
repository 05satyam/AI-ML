"""Golden eval set: question -> the doc id(s) that SHOULD be retrieved.

A small, version-controlled eval set is what turns "the RAG feels better" into a
number you can defend. Grow this from real production traces + failure cases.
"""

from __future__ import annotations

EVAL_SET: list[dict] = [
    {"q": "What is SoFi Plus?", "relevant": ["sofi-plus"]},
    {"q": "Does SoFi Plus cost money?", "relevant": ["sofi-plus-cost"]},
    {"q": "How do I organize my savings into goals?", "relevant": ["vault"]},
    {"q": "Can I freeze my card if it's stolen?", "relevant": ["fraud-alerts"]},
    {"q": "Are there fees on personal loans?", "relevant": ["loan-rates"]},
]
