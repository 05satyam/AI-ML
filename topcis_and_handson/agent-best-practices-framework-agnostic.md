# Best Practices for Building AI Agents (Framework-Agnostic)

This document captures practical lessons learned while building real-world AI agents
across RAG systems, agentic workflows, production APIs, and enterprise constraints.

These practices are **framework-agnostic** and apply whether you use LangChain,
LlamaIndex, Agno, LangGraph, or a fully custom orchestration layer.

The goal is not to build “smart demos”, but **systems that don’t break in production**. 

Note: Depending on the use case, some, all, or additional practices may apply—these guidelines are based on my own experience building and running agent systems in production.

---

## 1. First question: do you even need an agent?
Most problems do **not** need an agent.

From experience:
- If steps are known → use a **workflow**
- If decisions are dynamic, inputs are unpredictable, and tool choice matters → consider an **agent**

Agents add:
- nondeterminism
- higher debugging cost
- evaluation complexity

If a simple pipeline works, prefer it.

---

## 2. Define the agent contract before writing prompts
Treat an agent like a backend service.

Always define:
- **Goal**: what “done” means
- **Inputs**: allowed fields, size limits
- **Outputs**: strict schema (JSON preferred)
- **Failure behavior**: what happens when things go wrong
- **Non-goals**: what the agent must never attempt
- **Tool limits**: max calls, time, cost

This avoids prompt creep and uncontrolled behavior later.

---

## 3. Tools are the real system boundary (not the LLM)
Most production failures come from tools, not models.

Good tools:
- are **strictly typed**
- validate inputs
- return structured responses (`status`, `data`, `error`)
- behave deterministically

Bad tools:
- accept free-form text
- silently fail
- do too many things

Never rely on “prompt instructions” alone to keep tools safe.
Put safety checks **inside the tool implementation**.

---

## 4. Keep agent state explicit and boring
Do not let state live “inside the model”.

Maintain an explicit state object containing:
- inputs
- intermediate tool outputs
- decisions taken
- final response

Store tool outputs verbatim for debugging.
Summarize only when persisting long-term.

Avoid storing chain-of-thought.
If needed, store **short decision summaries**, not reasoning tokens.

---

## 5. Planning should be minimal and incremental
Over-engineered planning causes more failures than it prevents.

What works well in practice:
1. Understand the request
2. Decide the next best action
3. Execute one tool
4. Observe the result
5. Re-evaluate

This loop is easier to debug, test, and monitor than multi-step pre-plans.

---

## 6. Guardrails must exist outside prompts
If safety only exists in the prompt, it will fail.

Add guardrails at three levels:

### Input
- size limits
- schema validation
- injection detection
- PII checks

### Action
- allowlists (domains, commands, APIs)
- confirmation for irreversible actions
- confidence thresholds

### Output
- schema validation
- redaction
- refusal patterns with clear messaging

Guardrails belong in **code**, not just instructions.

---

## 7. Human-in-the-loop is a feature, not a weakness
In real systems, full autonomy is often a bad idea.

Add human checkpoints for:
- destructive actions
- financial impact
- customer-facing decisions
- compliance-sensitive flows

This dramatically reduces risk and increases trust.

---

## 8. Expect failures and design for them
Production agents must assume:
- tools will time out
- dependencies will fail
- models will hallucinate

Always implement:
- retries with backoff
- timeouts
- partial success handling
- fallback logic (simpler workflow or degraded mode)

Never block the entire system on one agent decision.

---

## 9. Evaluation is not optional
If you cannot measure agent behavior, you cannot trust it.

From experience:
- build a **fixed evaluation set**
- track success rate, latency, cost, and failure types
- include adversarial inputs
- re-run evals on every prompt or logic change

Agents regress silently if you don’t test them.

---

## 10. Observability matters more than model choice
When something breaks, you will not care which model you used.

You *will* care about:
- request ID
- agent version
- tool calls + arguments
- intermediate outputs
- guardrail triggers
- final response

Log everything needed to replay and debug issues.
Redact sensitive data early.

---

## 11. Scale with reusable “skills”, not more agents
Instead of building many agents:
- build one core agent
- add reusable skills (search, summarize, validate, extract, decide)

This reduces maintenance and improves consistency across use cases.

---

## 12. Production readiness checklist
Before shipping an agent:
- [ ] strict input/output schemas
- [ ] tool allowlists and validation
- [ ] retries and timeouts
- [ ] human approval where required
- [ ] evaluation dataset with metrics
- [ ] logging and traceability
- [ ] cost and latency budgets
- [ ] kill switch / rollback plan

If any box is unchecked, it’s not production-ready.

---

## Final note
Good agents feel boring when they work.
They are predictable, constrained, observable, and easy to debug.

If your agent feels “magical”, it’s probably fragile.
