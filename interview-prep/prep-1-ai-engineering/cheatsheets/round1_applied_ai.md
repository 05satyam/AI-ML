# Round 1 — Ankush (Staff AI Eng) — 45m HackerRank, applied AI

**Goal:** build a small applied-AI feature *well*. Likely: a RAG endpoint, an agent
with a tool, an extraction/classification pipeline, or extend/fix an LLM service.

## Minute-by-minute
- **0–5 Clarify.** Inputs/outputs? Latency & cost budget? Expected scale/QPS? What's
  the success/eval criteria? Any data provided? Sync or streaming?
- **5–10 Design out loud.** Data flow diagram in words: where the LLM call sits,
  interfaces, failure modes, what you'd mock.
- **10–30 Build the happy path.** Clean interfaces, type hints, small functions.
- **30–40 Harden.** Timeouts, retries w/ backoff, input validation, structured-output
  parsing (+ a re-ask on parse failure), graceful fallback when the LLM fails.
- **40–45 Verify + extend.** A tiny test or eval. Name what you'd do with more time.

## Staff-level "tells" (do these, most candidates don't)
- Add **structured output + validation** (Pydantic) and handle parse failures.
- Add a **tiny eval** even if unasked ("here's how I'd know this works").
- Talk about **idempotency, observability, and cost** without being asked.
- Choose the **cheapest model that passes the bar**; mention a routing/escalation idea.
- Say what you'd **mock vs. call for real**, and why.

## Reusable Python snippets to have in muscle memory

Structured output with retry-on-parse-failure:

```python
import json
from pydantic import BaseModel, ValidationError

class Extraction(BaseModel):
    intent: str
    amount: float | None = None

async def extract(llm, text: str, max_tries: int = 2) -> Extraction:
    prompt = f"Return JSON with keys intent, amount. Text: {text}"
    for _ in range(max_tries + 1):
        raw = await llm.complete(system="Return ONLY valid JSON.", user=prompt)
        try:
            return Extraction.model_validate_json(raw)
        except ValidationError as e:
            prompt = f"{prompt}\nYour last output was invalid: {e}. Return valid JSON."
    raise ValueError("could not parse structured output")
```

A minimal tool-using agent loop (ReAct-style, no framework needed):

```python
async def agent(llm, tools: dict, question: str, max_steps: int = 5) -> str:
    scratch = ""
    for _ in range(max_steps):
        plan = await llm.complete(
            system="Decide next action. Reply 'TOOL <name> <arg>' or 'FINAL <answer>'.",
            user=f"Q: {question}\nScratch:\n{scratch}",
        )
        if plan.startswith("FINAL"):
            return plan[len("FINAL"):].strip()
        _, name, arg = plan.split(" ", 2)
        scratch += f"\n{plan}\nOBS: {tools[name](arg)}"
    return "stopped: step budget exhausted"   # always bound the loop
```

## Traps to avoid
- Pasting AI output without reading it. **Read it aloud, critique it.**
- Unbounded agent loops (always cap steps + cost).
- No error handling on the LLM call.
- Over-engineering before the happy path runs.
