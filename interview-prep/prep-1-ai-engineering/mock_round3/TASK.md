# Mock Round 3 — production codebase task (timebox ~30 min)

You've joined the team. This `faq_bot.py` module powers a member-facing FAQ assistant.
QA filed two issues. Work it like the real interview: **orient → clarify → plan →
fix → verify**, narrating your AI usage. Run the tests with:

```bash
python3 mock_round3/run_tests.py
```

## Tickets

**BUG-1 (P1):** "Members report the assistant sometimes answers using *another*
member's previous conversation." Reproduce, find root cause, fix.

**BUG-2 (P2):** "Search results look backwards — the assistant cites the *least*
relevant FAQ, not the most relevant."

**FEATURE-1 (if time):** Add a per-request **timeout/guardrail** so a single slow
retrieval can't hang the endpoint, and add **input validation** (reject empty/oversized
queries). Add a test.

## Rules of engagement (what the interviewer is watching)
- Reproduce before fixing. State your hypothesis out loud.
- Add/adjust a **regression test** for each bug.
- Keep changes minimal and explain blast radius.
- Use the AI assistant to help, but **review every suggestion aloud**.

When done (or at 30 min), read `SOLUTION.md`.
