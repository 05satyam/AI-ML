# Round 3 — Faisal — 60m Zoom, production codebase + AI tools

> "Work through a production codebase with AI tools available and encouraged. Use AI
> the way you would on your first week at SoFi."

This is **not** leetcode. It's: dropped into unfamiliar code — can you orient, reason,
and make a *safe* change while using AI well?

## The 5-phase playbook (rehearse this exact flow)

### 1. Orient (5–10 min) — resist the urge to type code
- Map entry points, data flow, key modules, dependencies, how it's tested/run.
- Use AI to accelerate, then **verify against the real files**:
  - *"Summarize this repo's architecture and where a request enters."*
  - *"What are the main modules and how do they depend on each other?"*
- Skim the README, config, tests. Run the tests / start the app if you can.

### 2. Clarify the task (2–5 min)
- Restate it back. Confirm scope, constraints, definition of done, eval bar.
- Ask: should I prioritize correctness, perf, or readability? Any code I must NOT touch?

### 3. Plan (2–3 min)
- State the change, files you'll touch, the smallest correct increment, how you'll test.
- Call out blast radius and backward compatibility up front.

### 4. Implement incrementally
- Smallest change that works → run → iterate. Don't big-bang.
- Use AI for boilerplate/scaffolding; **read + critique every suggestion aloud**.
- Match existing patterns/style in the repo (don't impose your own).

### 5. Verify + communicate
- Run tests / reproduce the scenario. Add/adjust a test for your change.
- Summarize: what changed, why it's safe, how to roll back, what you'd do next.

## How to use AI here (they're literally watching this)
- **Good:** scoped prompts, asking it to explain code, generate a test, draft a diff —
  then you review and adjust.
- **Great:** catching when it hallucinates an API, misses an edge case, or introduces
  a security/PII issue, and saying so.
- **Bad:** pasting a large AI diff you can't explain; letting it drive blindly.

## Debugging variant (if it's a bug task)
1. Reproduce first. 2. Form a hypothesis. 3. Localize (logs/trace/bisect). 4. Minimal
fix. 5. Add a regression test. 6. State root cause + prevention. Narrate the hypothesis
loop — that's the signal.

## Refactor / feature variant
- Identify the seam/interface. Keep changes localized. Preserve behavior (tests prove it).
- For AI features specifically: add timeout/retry/fallback, structured output, a trace,
  and a token/cost guardrail — these are the Staff-level touches.

## Practice
Use `../mock_round3/` — open `TASK.md`, work it under 30 min with an AI assistant,
then read `SOLUTION.md`.
