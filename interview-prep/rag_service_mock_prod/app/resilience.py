"""Reliability primitives for calling flaky external services (LLM/embeddings).

- timeout: never let a hung provider call block a request forever
- retry with exponential backoff + jitter: ride out transient 429/5xx
- circuit breaker: stop hammering a dead dependency; fail fast and degrade

These are intentionally dependency-free so you can explain every line.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


class CircuitBreaker:
    """Trips open after N consecutive failures, half-opens after a cooldown."""

    def __init__(self, fail_threshold: int, reset_s: float):
        self.fail_threshold = fail_threshold
        self.reset_s = reset_s
        self._failures = 0
        self._opened_at: float | None = None

    def _half_open_ready(self) -> bool:
        return self._opened_at is not None and (time.time() - self._opened_at) >= self.reset_s

    def before_call(self) -> None:
        if self._opened_at is not None and not self._half_open_ready():
            raise CircuitOpenError("circuit open")

    def on_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold:
            self._opened_at = time.time()


async def call_with_resilience(
    fn: Callable[[], Awaitable[T]],
    *,
    timeout_s: float,
    max_retries: int,
    breaker: CircuitBreaker | None = None,
) -> T:
    breaker and breaker.before_call()
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout_s)
            breaker and breaker.on_success()
            return result
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last_exc = exc
            breaker and breaker.on_failure()
            if attempt < max_retries:
                backoff = (2 ** attempt) * 0.25 + random.uniform(0, 0.1)
                await asyncio.sleep(backoff)
    assert last_exc is not None
    raise last_exc
