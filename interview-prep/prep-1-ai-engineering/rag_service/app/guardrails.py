"""Input validation + cost/token guardrails.

For a *risk org*, this is not optional polish — bounding inputs, context size, and
cost is part of making an AI system safe to run in production. Mention PII redaction
and prompt-injection screening here as the next layers.
"""

from __future__ import annotations

from .config import settings


class GuardrailError(ValueError):
    pass


def validate_input(message: str) -> None:
    if not message.strip():
        raise GuardrailError("empty message")
    if len(message) > settings.max_input_chars:
        raise GuardrailError(f"message exceeds {settings.max_input_chars} chars")


def approx_tokens(text: str) -> int:
    # ~4 chars/token heuristic; replace with tiktoken in prod.
    return max(1, len(text) // 4)


def clamp_context(snippets: list[str], budget_tokens: int | None = None) -> list[str]:
    """Greedily include retrieved snippets until the token budget is hit.
    Protects both cost and the model's context window."""
    budget = budget_tokens or settings.max_context_tokens
    used = 0
    kept: list[str] = []
    for s in snippets:
        t = approx_tokens(s)
        if used + t > budget:
            break
        kept.append(s)
        used += t
    return kept
