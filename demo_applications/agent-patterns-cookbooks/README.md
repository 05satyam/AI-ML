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
flowchart LR
  subgraph Shared["Shared module (Option A): claim_verifier.py"]
    direction TB
    S1["build_retriever <br/> (pdf_path)"]
    S2["PyPDFLoader → <br/> load pages"]
    S3["TextSplitter → <br/> chunk"]
    S4["Embeddings → <br/> OpenAIEmbeddings"]
    S5["Chroma DB <br/>(persist_directory)"]
    S6["retriever = as_retriever(k)"]
    S7["make_search_tool<br/> (retriever)"]
    S8["Tool: claim_verifier(query) → cited snippets (page X)"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 --> S7 --> S8
  end

  subgraph NB1["Notebook 01: ReAct (format-sensitive)"]
    direction TB
    A1["ChatOpenAI"]
    A2["create_react_agent + AgentExecutor"]
    A3["Prompt requires: {tools}, {tool_names}, {agent_scratchpad}"]
    A4{ReAct loop}
    A5["Action: claim_verifier(query)"]
    A6["Observation: cited snippets"]
    A7["Final: answer w/ citations"]
    A8["Risk: parser loop / iteration limit"]

    A1 --> A2 --> A3 --> A4
    A4 -->|Action| A5 --> A6 --> A4
    A4 -->|Final| A7
    A4 -->|Bad format| A8
  end

  subgraph NB2["Notebook 02: Plan & Execute (stable)"]
    direction TB
    P1["Plan: make_plan(question)"]
    P2["Step loop: generate <br/> search query"]
    P3["Direct call: <br/> claim_verifier(query)"]
    P4["Summarize findings <br/> (page X)"]
    P5["Synthesize final <br/> answer + citations"]

    P1 --> P2 --> P3 --> P4 --> P5
  end

  subgraph NB3["Notebook 03: Reflexion (quality-focused)"]
    direction TB
    R1["Draft (tool-first)"]
    R2["Critique (grounding <br/> + citations)"]
    R3["Revise (+ optional <br/> 2nd retrieval)"]
    R4["Final improved answer <br/> (page X)"]

    R1 --> R2 --> R3 --> R4
  end

  %% Reuse links (kept minimal to avoid crossings)
  Shared --> NB1
  Shared --> NB2
  Shared --> NB3

```
