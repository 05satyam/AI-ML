"""Conversational memory + follow-up query rewriting.

Two responsibilities, kept separate on purpose:
1. Memory store: per-session rolling window of turns (here in-process; in prod Redis
   with a TTL, keyed by session_id). Conversation state is deliberately NOT the same
   store as the knowledge base.
2. Query rewriting: turn a context-dependent follow-up ("how much does it cost?")
   into a standalone query ("how much does SoFi Plus cost?") so retrieval works.
   This is THE fix for "follow-up questions retrieve garbage".
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .config import settings
from .llm import ChatModel


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


class MemoryStore:
    def __init__(self, max_turns: int = settings.max_history_turns):
        self._max = max_turns
        self._sessions: dict[str, deque[Turn]] = defaultdict(lambda: deque(maxlen=self._max * 2))

    def history(self, session_id: str) -> list[Turn]:
        return list(self._sessions[session_id])

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append(Turn(role, content))

    def as_text(self, session_id: str) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.history(session_id))


class QueryRewriter:
    """Rewrites follow-ups using recent history. Uses the LLM when available;
    falls back to a heuristic (append last user topic) so it works offline.

    `use_llm=False` (no real model configured) skips the LLM call entirely and
    uses the heuristic, so the offline demo still rewrites follow-ups sensibly."""

    def __init__(self, chat: ChatModel, use_llm: bool = True):
        self.chat = chat
        self.use_llm = use_llm

    async def rewrite(self, session_id: str, message: str, memory: MemoryStore) -> str:
        history = memory.history(session_id)
        if not history:
            return message  # first turn is already standalone

        if not self.use_llm:
            return self._heuristic(message, history)

        history_text = memory.as_text(session_id)
        system = (
            "You rewrite a user's latest message into a standalone search query, "
            "resolving pronouns and ellipsis using the conversation. "
            "Return ONLY the rewritten query."
        )
        user = f"Conversation:\n{history_text}\n\nLatest message: {message}\n\nStandalone query:"
        try:
            rewritten = (await self.chat.complete(system, user)).strip()
            # Guard against a degenerate/empty rewrite.
            return rewritten or message
        except Exception:
            # Heuristic fallback: if the message looks like a follow-up (short /
            # pronoun-y), prepend the most recent user topic.
            return self._heuristic(message, history)

    @staticmethod
    def _heuristic(message: str, history: list[Turn]) -> str:
        followup_markers = ("it", "that", "this", "they", "those", "how much", "what about")
        looks_followup = len(message.split()) <= 6 or any(
            m in message.lower() for m in followup_markers
        )
        if not looks_followup:
            return message
        last_user = next((t.content for t in reversed(history) if t.role == "user"), "")
        return f"{message} (regarding: {last_user})" if last_user else message
