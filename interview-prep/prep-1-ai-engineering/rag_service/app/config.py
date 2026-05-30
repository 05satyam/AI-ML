"""Environment-driven settings.

Everything that differs between local / staging / prod lives here so the rest of
the code is environment-agnostic. In an interview, point at this file when asked
"how would you configure this per environment?".
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Providers. If openai_api_key is empty we fall back to deterministic fakes,
    # so the service is runnable in CI / on a laptop with no secrets.
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"

    # Retrieval
    top_k: int = 4
    rerank_top_n: int = 8  # retrieve more, then rerank down to top_k
    hybrid_alpha: float = 0.6  # weight on vector vs keyword score (0..1)

    # Memory
    max_history_turns: int = 6

    # Resilience
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2
    breaker_fail_threshold: int = 5
    breaker_reset_s: float = 30.0

    # Guardrails
    max_input_chars: int = 4000
    max_context_tokens: int = 3000  # approx; protects cost + context window

    # Cache
    semantic_cache_threshold: float = 0.95  # cosine sim to count as a cache hit


settings = Settings()
