# AI Agent Tools — Interview Questions

Focused on **tool / function calling**: schemas, selection, reliability, MCP-style tool servers, and safety. Often a separate deep-dive from “agents” in interviews.

## Basic

1. What is function / tool calling in modern LLM APIs?
2. Why pass a JSON schema instead of free-text “use this API”?
3. What belongs in a tool description so the model selects it correctly?
4. Parallel tool calls vs sequential — when is each required?
5. What is the difference between the model **proposing** a tool call and your runtime **executing** it?
6. How do you return tool results back into the conversation / agent state?
7. Give examples of good tools vs bad mega-tools (“do_everything”).
8. What happens if a tool returns an error — what should the agent see?
9. Why keep tools **idempotent** when possible?
10. Structured outputs vs tool calling — overlapping use cases?

**Cue notes:** Model emits structured call → your code validates → executes → feeds observation. Never trust model args blindly.

## Intermediate

11. Design a tool schema for `search_docs(query, top_k, filters)` — required fields and constraints.
12. How do you validate / coerce arguments before execution?
13. Tool selection mistakes: too many similar tools — how do you fix routing?
14. Timeouts, retries, and circuit breakers around tool I/O.
15. Caching tool results (search, HTTP GET) — invalidation rules.
16. AuthN/Z: how does the agent inherit user permissions for tool calls?
17. Rate limits and budget caps per tool / per session.
18. Logging & redaction: what must never hit logs (secrets, PII)?
19. MCP (Model Context Protocol) or similar — why standardize tool servers?
20. Retriever-as-tool vs always-on RAG pipeline — when to expose search as a tool.

**Cue notes:** Permissions at the **tool boundary**; schemas should be small and unambiguous; prefer fewer sharp tools.

## Advanced

21. Dangerous tools (email send, DB write, shell): approval gates and dry-run modes.
22. Prompt injection via tool output (web page / email) that tries to trigger other tools — defenses.
23. Transactional tools: exactly-once vs at-least-once; compensating actions.
24. How do you version tools without breaking running agents?
25. Multi-tenant tool registries — isolate credentials and data planes.
26. Evaluating tool-use: exact match on tool name/args vs outcome success.
27. Designing tools for coding agents (read_file, run_tests) with sandboxing.
28. Cost: tools that themselves call LLMs (rerank, extract) — how do you account for nested spend?
29. Fallback strategies when the primary tool is down.
30. End-to-end: design tool layer for a support agent (search KB, create ticket, refund) with policy checks.

**Cue notes:** Interviewers look for **validation + auth + observability + kill switches**, not just OpenAI function-calling syntax.

## Mini design drills

- Spec 5 tools for a travel-booking agent; mark which need HITL.
- A tool suddenly returns HTML with “ignore previous instructions…” — your response path.
- Compare embedding search tool vs SQL tool for analytics questions.

## Hands-on in this repo

- [dspy_hands_on](../dspy_hands_on/) — `ReAct` + Python tools
- MCP demos in the main repo (see root README → MCP section)

## References

- [AI Agents & Tool Use — Rubduck](https://rubduck.ai/questions/ai-agents-and-tool-use)
- [Agentic AI Interview Questions — LockedIn AI](https://www.lockedinai.com/blog/agentic-ai-interview-questions)
- [Agentic design patterns — Tool Use](https://addyosmani.com/agents/04-agentic-design-patterns/)
- OpenAI / Anthropic / Gemini tool-calling docs (latest provider docs)
