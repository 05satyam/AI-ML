# Round 2 — Amol (Director, Eng & AI) — 45m behavioral / judgment

No coding. Tests: systems thinking, ownership, collaboration, AI judgment, culture.

## Format every answer as STAR, lead with impact
**S**ituation → **T**ask → **A**ction (what *you* did) → **R**esult (with a number).
Keep it ~2 min. Then stop and let them dig.

## Likely questions + the angle to hit
- **"Tell me about an ambiguous problem you turned into a working system."**
  → Show the scoping move: how you cut ambiguity into a v1 with a measurable goal.
- **"How do you decide when NOT to use an LLM / AI?"**  ← *huge for a risk org*
  → Deterministic rules where correctness/auditability matter; LLM where flexibility
    wins. Cost, latency, explainability, failure cost.
- **"How do you get non-technical (risk) stakeholders to trust an AI system?"**
  → Evals tied to their metrics, human-in-the-loop, traceability/audit logs, gradual
    rollout (shadow → assist → automate), clear failure modes.
- **"Describe an AI system that failed in production. What did you do?"**
  → Detection (observability), containment (fallback/rollback), root cause, the
    guardrail/eval you added so it can't recur.
- **"A teammate disagrees with your technical approach."**
  → Seek their model, find the disconfirming experiment, disagree-and-commit.
- **"How do you balance speed vs. reliability?"**
  → Reversible vs irreversible decisions; ship reversible fast, gate irreversible.

## SoFi-specific framing to weave in
- It's an **independent risk org** → emphasize **auditability, explainability,
  regulated-environment** thinking, human oversight, measurable risk reduction.
- JD words to mirror: *agentic systems, context engineering, experience layer,
  observability/evals, cross-functional, ownership end-to-end.*
- Anthony Noto / 2026 roadmap, mobile-first, "member" (not "user") — shows you did homework.

## Questions to ask them (pick 2–3)
- How does the risk org measure ROI / success of agentic systems today?
- What does the "experience layer" look like for your internal AI tools?
- Where's the line between human-in-the-loop and full automation for risk decisions?
- What does the eval/observability stack look like (Langfuse/LangSmith)?
- Biggest reliability or trust challenge with AI in production right now?

## Tells of seniority
- You quantify outcomes. You name trade-offs unprompted. You talk about other people
  (collaboration, mentoring). You admit a failure cleanly and show the lesson.
