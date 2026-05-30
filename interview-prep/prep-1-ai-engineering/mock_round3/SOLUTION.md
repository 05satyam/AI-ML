# Mock Round 3 — solution walkthrough

Read this **after** attempting `TASK.md`. The point isn't just the fixes — it's the
*process* you narrate.

## BUG-1: cross-session memory leak (the high-signal one)

**Root cause:** `def answer(self, session_id, query, _history=[])`. The default list is
created **once at function definition** and shared by every call/session — the classic
Python mutable-default-argument trap. So "alice"'s turns accumulate into "bob"'s history.

**Why it's a great find to verbalize:** it's a real correctness + **privacy** bug (one
member seeing another's context) — exactly the kind of thing that matters in a risk org.

**Fix:** store memory as instance state keyed by `session_id`.

```python
@dataclass
class FaqBot:
    top_k: int = 2
    _memory: dict[str, list[str]] = field(default_factory=dict)  # per-session

    def answer(self, session_id: str, query: str) -> dict:
        history = self._memory.setdefault(session_id, [])
        history.append(query)
        ...
        return {..., "history_len": len(history)}
```

(Add `from dataclasses import field`.) In real prod: Redis keyed by `session_id` with a
TTL, not in-process — mention this.

## BUG-2: ranking reversed

**Root cause:** `sorted(scored, key=lambda kv: kv[1])` sorts **ascending**, so the
*lowest* cosine score comes first → least relevant FAQ cited.

**Fix:**

```python
ranked = sorted(scored, key=lambda kv: kv[1], reverse=True)
```

**Talking point:** add a regression test asserting the top citation for "freeze my card"
is `freeze-card` so this can't silently regress.

## FEATURE-1: timeout + input validation

**Validation:**

```python
MAX_QUERY_CHARS = 2000

def answer(self, session_id: str, query: str) -> dict:
    if not query or not query.strip():
        raise ValueError("query must be non-empty")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query exceeds {MAX_QUERY_CHARS} chars")
    ...
```

**Timeout (concept):** retrieval here is in-process and fast, but the *pattern* they
want: wrap any external/slow call so one request can't hang the endpoint. With async:

```python
import asyncio
result = await asyncio.wait_for(self._retrieve_async(query), timeout=2.0)
```

For sync code, run it in a thread/executor with a timeout, or enforce at the request
layer (e.g., server worker timeout). **Say the trade-off:** a hard per-request timeout
protects tail latency + capacity, at the cost of occasionally failing slow-but-valid
requests — pair it with a retry or graceful fallback.

## What "passing" looks like to the interviewer
- You reproduced each bug before fixing (ran the tests).
- You named the **mutable-default** root cause precisely and flagged the privacy angle.
- You added regression tests.
- You used the AI assistant to help but **reviewed and corrected** its output.
- You discussed the prod-grade version (Redis memory, real timeout strategy), not just
  the local fix.

Run `python3 mock_round3/run_tests.py` — all three should pass after your fixes.
