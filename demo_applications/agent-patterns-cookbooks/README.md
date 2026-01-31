## Agent Patterns Pack (PDF Policy QA)

This mini-project demonstrates **three agent patterns** solving the **same task**:  
**answer questions from a PDF policy using retrieval + page citations**.

### What's included

- **Shared module: `claim_verifier.py`**
  - Loads a PDF, chunks it, builds a persistent **Chroma** vector index
  - Exposes one tool: `claim_verifier(query)` → returns **cited snippets with page numbers**

- **Notebook 01 — ReAct Agent**
  - Tool-using agent loop (reason → tool call → observe → answer)
  - Shows why ReAct prompts require placeholders like `{tools}`, `{tool_names}`, `{agent_scratchpad}`

- **Notebook 02 — Plan & Execute**
  - Planner creates a short step-by-step plan
  - Executor runs each step via `claim_verifier`, collects cited notes, then synthesizes the final answer

- **Notebook 03 — Reflexion / Self-check**
  - Draft answer using retrieval
  - Critique pass checks grounding + citation quality
  - Revision pass improves the final answer (optionally with a second retrieval)

### Goal

Compare **when each pattern helps** and what it costs (latency / number of model calls), while keeping answers **grounded** with **(page X)** citations.

---

```mermaid
flowchart TD
  %% =========================
  %% Shared module (Option A)
  %% =========================
  S[claim_verifier.py (shared)] --> S1[build_retriever(pdf_path)]
  S1 --> L1[PyPDFLoader: load PDF pages]
  L1 --> L2[RecursiveCharacterTextSplitter: chunk pages]
  L2 --> L3[OpenAIEmbeddings: embed chunks]
  L3 --> DB[(Chroma DB: persist_directory)]
  DB --> R[retriever = as_retriever(k)]
  S --> T[make_search_tool(retriever)]
  T --> TOOL[Tool: claim_verifier(query) -> cited snippets (page X)]

  %% =========================
  %% Notebooks reuse shared module
  %% =========================
  N1[Notebook 01: ReAct] --> S
  N2[Notebook 02: Plan & Execute] --> S
  N3[Notebook 03: Reflexion] --> S

  %% =========================
  %% Notebook 01: ReAct (format-sensitive)
  %% =========================
  N1 --> A1[ChatOpenAI]
  A1 --> A2[create_react_agent + AgentExecutor]
  TOOL --> A2
  A2 --> A3[Prompt must include: {tools}, {tool_names}, {agent_scratchpad}]
  A2 --> A4{ReAct loop}
  A4 -->|Action| A5[call claim_verifier]
  A5 -->|Observation| A4
  A4 -->|Final| OUT1[Answer w/ citations]
  A4 -->|Bad format| ERR[Parser loop risk (iteration limit)]

  %% =========================
  %% Notebook 02: Plan & Execute (stable)
  %% =========================
  N2 --> P1[Planner LLM: make_plan(question)]
  P1 --> P2[For each step: generate search query]
  P2 --> P3[Direct tool call: claim_verifier(query)]
  P3 --> P4[Summarize step findings (keep (page X))]
  P4 --> P5[Final synthesis: answer with citations]
  P5 --> OUT2[Final Answer]

  %% =========================
  %% Notebook 03: Reflexion (quality-focused)
  %% =========================
  N3 --> R1[Draft: tool-first retrieval + draft answer]
  R1 --> R2[Critique: check grounding + citations]
  R2 --> R3[Revise: optional second retrieval guided by critique]
  R3 --> OUT3[Improved Final Answer w/ citations]

```