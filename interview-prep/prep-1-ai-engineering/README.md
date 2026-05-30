# Staff AI Engineer — Interview Prep

Prep package for the 3-session virtual panel (~2.5h). Built around the exact themes
the hiring manager round surfaced (RAG, optimization, follow-ups, conversational memory,
RAG deployment, Kubernetes scalability) and the JD (agentic systems, context engineering,
evals, observability, risk/regulated context).

## The panel

| # | Interviewer | Len | Type | What it tests |
|---|-------------|-----|------|----------------|
| 1 | Ankush (Staff AI Eng) | 45m | HackerRank, hands-on applied AI | Can you *build* an applied AI feature well |
| 2 | Amol (Director, Eng & AI) | 45m | Behavioral / judgment | Systems thinking, culture, collaboration |
| 3 | Faisal | 60m (Zoom) | Production codebase + AI tools | Orient in unfamiliar code, make a safe change |

## AI tools you'll have
Claude 4.6, ChatGPT 4.5, Gemini — **single-assistant, Cursor-like, NO multi-agent**.
Use it like a pair: narrate prompts, **review every suggestion out loud**, catch its
mistakes. They grade *judgment*, not speed.

## Contents
- `rag_service/` — runnable production-style RAG reference (read it, run it, explain it)
- `cheatsheets/round1_applied_ai.md`
- `cheatsheets/round2_director_behavioral.md`
- `cheatsheets/round3_production_codebase.md`
- `cheatsheets/themes_rag_memory_k8s.md` — the deep technical syllabus
- `cheatsheets/ai_pairing_playbook.md` — how to use AI on camera
- `behavioral/star_stories.md` — fill-in story bank
- `drills/qna_drill.md` — Q&A with answers (self-test)
- `production_codebase_drill/` — a planted-bug "production" mini-codebase + TASK.md

## 30-second pre-interview ritual (every round)
1. **Clarify** inputs/outputs, constraints (latency, cost, scale), and the eval bar.
2. **State a plan** before coding.
3. **Build happy path → harden** (timeouts, retries, validation, fallback).
4. **Verify** + discuss blast radius / rollback.
5. **Narrate AI usage**; review what it generates.
