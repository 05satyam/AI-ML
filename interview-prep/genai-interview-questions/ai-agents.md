# AI Agents — Interview Questions

Questions on agent loops, planning patterns, memory, multi-agent orchestration, and production reliability (common in 2025–2026 GenAI loops).

## Basic

1. What is an AI agent vs a single LLM call / chatbot turn?
2. Describe the agent loop: observe → reason → act → observe…
3. What is ReAct (Reason + Act)?
4. What is a tool in an agent system?
5. When should you **not** build an agent (use a workflow / RAG / rules instead)?
6. What is chain-of-thought vs an agent that takes actions?
7. What is human-in-the-loop (HITL) and when is it mandatory?
8. Name two agent failure modes (infinite loops, wrong tool, hallucinated args).
9. Short-term vs long-term memory for agents — one example each.
10. What does “grounding” mean for an agent that answers from tools / docs?

**Cue notes:** Agent = LLM-driven control flow with side effects. Prefer deterministic workflows when steps are fixed.

## Intermediate

11. ReAct vs Plan-and-Execute — when is each better?
12. Reflection / self-critique patterns — cost vs quality.
13. Router / supervisor agent — how do you dispatch to specialists?
14. Multi-agent patterns: orchestrator–workers vs peer debate — tradeoffs.
15. How do you prevent infinite tool loops (max steps, budgets, repeated-state detection)?
16. Memory types: working, summary, episodic, semantic, procedural — when to use each.
17. How do you evaluate an agent (task success, step efficiency, cost, safety)?
18. Deterministic graph workflows (e.g., LangGraph-style) vs free-form agent loops.
19. Idempotency and retries when tools have side effects (payments, tickets).
20. Observability: traces, span per tool call, replay for debugging.

**Cue notes:** Production systems often **compose** patterns (plan outer, ReAct inner). Always mention step limits + cost caps.

## Advanced

21. Design a multi-agent system for “research → draft → fact-check → publish” with ownership of failures.
22. How do you sandboxes / permission tiers for tools (read-only vs write / prod)?
23. Agent eval harness: golden trajectories vs outcome-only metrics — pros/cons.
24. Cost explosion: how do you budget tokens across planner, workers, and critics?
25. Long-running agents: checkpointing, resume, and context compression strategies.
26. When does multi-agent **hurt** (coordination overhead, contradictory tools)?
27. Policy / compliance: audit logs for every tool call that mutated state.
28. Compare building with a framework vs plain Python orchestration — what do you need first?
29. How would you A/B test agent policy changes safely?
30. Security: indirect prompt injection from tool outputs / retrieved web pages — mitigations.

**Cue notes:** Strong answers specify contracts between agents (schemas, SLAs), not just “add more agents.”

## System design prompts

- Customer-support agent with RAG + ticket tools + escalation to human.
- Coding agent with repo tools — stop conditions and test gates.
- Compare single-agent ReAct vs supervisor + specialists for your product.

## Hands-on in this repo

- [ai_agents_memory_types.md](../ai_agents_memory_types.md)
- [dspy_hands_on](../dspy_hands_on/) (ReAct module)
- [build_multi_agent_from_scratch](../build_multi_agent_from_scratch/) (if present locally)

## References

- [AI Agents & Tool Use — Rubduck](https://rubduck.ai/questions/ai-agents-and-tool-use)
- [Agentic AI Interview Questions — LockedIn AI](https://www.lockedinai.com/blog/agentic-ai-interview-questions)
- [Agentic AI Interview Questions (Junior & Senior)](https://interviewbaba.com/agentic-ai-interview-questions/)
- [ReAct vs Plan-and-Execute](https://buildingagenticai.com/blog/react-vs-plan-and-execute/)
- [Agentic design patterns — Addy Osmani](https://addyosmani.com/agents/04-agentic-design-patterns/)
