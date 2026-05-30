"""Member-facing FAQ assistant (mock production module).

Small, pure-Python (no deps) so it's easy to run in an interview environment.
Contains intentionally realistic bugs — see TASK.md. Do NOT peek at SOLUTION.md first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FAQS: list[dict[str, str]] = [
    {"id": "reset-password", "text": "To reset your password, open the app, tap Profile, then Security, then Reset Password."},
    {"id": "freeze-card", "text": "You can freeze your card instantly from the app under Cards, then Freeze Card."},
    {"id": "direct-deposit", "text": "Set up direct deposit by sharing your account and routing number with your employer."},
    {"id": "transfer-limits", "text": "Daily transfer limits depend on your account type and can be viewed in Settings, then Limits."},
    {"id": "close-account", "text": "To close your account, contact support; ensure your balance is zero first."},
]


def _vec(text: str) -> dict[str, float]:
    """Tiny bag-of-words term-frequency 'embedding'."""
    v: dict[str, float] = {}
    for tok in text.lower().split():
        v[tok] = v.get(tok, 0.0) + 1.0
    return v


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(x * x for x in a.values())) or 1.0
    nb = math.sqrt(sum(x * x for x in b.values())) or 1.0
    return dot / (na * nb)


@dataclass
class FaqBot:
    top_k: int = 2

    def _retrieve(self, query: str) -> list[tuple[str, float]]:
        q = _vec(query)
        scored = [(faq["id"], _cosine(q, _vec(faq["text"]))) for faq in FAQS]
        # BUG-2 lives here.
        ranked = sorted(scored, key=lambda kv: kv[1])
        return ranked[: self.top_k]

    def answer(self, session_id: str, query: str, _history: list[str] = []) -> dict:
        # BUG-1 lives in this signature: a mutable default argument is created ONCE
        # and shared across every call and every session.
        _history.append(query)

        results = self._retrieve(query)
        top_id = results[0][0] if results else None
        text = next((f["text"] for f in FAQS if f["id"] == top_id), "I don't know.")

        return {
            "answer": text,
            "citations": [r[0] for r in results],
            "history_len": len(_history),
        }
