# Policy-Gated MCP Agent

This README is adjusted for a **1-hour beginner-friendly webinar** on agentic AI and MCP.

Agentic AI is the next step beyond chatbots: instead of only generating text, an agent can decide what to do, call tools, and verify outputs. In this session, we build a simple **Study Assistant Agent** that uses the Model Context Protocol (MCP) to connect to tools in a clean, standardized way.


## Agenda
0–5 min: Welcome + why agentic AI matters  
5–10 min: MCP concept (simple mental model)  
10–35 min: Hands-on build: MCP tools + agent routing  
35–50 min: Add simple eval step (self-check) + reliability patterns  
50–60 min: Live experiments with audience prompts + Q&A

## What is an AI Agent?

<img width="2880" height="688" alt="image" src="https://github.com/user-attachments/assets/a660ec29-0009-4191-b212-43b609f78948" />


### Key terms
-  **LLM (Large Language Model):** The core reasoning engine that predicts the next text and suggests actions.  
-  **Tool:** A function or API the agent can call (e.g., search, math, external APIs).  
-  **Memory:** Stores facts and context (short + long term) so the agent can recall information over time.  
-  **Agent:** Logic that loops: *plan → act → check → respond* using the LLM, tools, and memory.

### How Do AI Agents Work?
<img width="2880" height="1244" alt="image" src="https://github.com/user-attachments/assets/aef6f03d-efa0-41fd-84b2-274d56096249" />

## What is MCP?
MCP provides a standardized way for applications to share context, expose tools, and connect AI systems to capabilities across servers. It uses JSON-RPC for client-host-server communication, with capability negotiation and stateful sessions. Sources:
<img width="2500" height="977" alt="image" src="https://github.com/user-attachments/assets/d3c00d53-da42-4120-bb86-8bd5cc49d149" />
## Model Context Protocol (MCP) — Quick README Summary

| Section | Summary (easy bullets) |
|---|---|
| **Who talks to who** | - **Host:** The AI app (starts the connection + owns user consent UI).<br>- **Client:** A connector inside the Host (the “adapter” that speaks MCP).<br>- **Server:** External service that provides context + capabilities (tools/data/prompts). |
| **Why MCP exists** | - Inspired by **Language Server Protocol (LSP)**.<br>- Like LSP makes language tooling plug-and-play across editors, **MCP makes context + tools plug-and-play across AI apps**. |
| **Protocol basics** | - Uses **JSON-RPC 2.0** messages<br>- Works over **stateful connections** (not just one-off calls)<br>- **Capability negotiation** so both sides agree on what features are supported |
| **What servers can provide** | - **Resources:** Data/context the user or model can read/use<br>- **Prompts:** Reusable templates/workflows for consistent interactions<br>- **Tools:** Callable functions the model can run (powerful → requires caution) |
| **What clients can provide** | - **Sampling:** Server can request the host to run LLM interactions (agentic / recursive behaviors) — but host/user stay in control |
| **Built-in utilities** | - Configuration<br>- Progress updates<br>- Cancellation support<br>- Standard error reporting<br>- Logging hooks |
| **Security & Trust (must-have)** | - **User Consent & Control:** Users explicitly approve data access + actions; clear UI for review/authorization.<br>- **Data Privacy:** Host only shares user data with explicit consent; no re-sharing resource data without consent; apply access controls.<br>- **Tool Safety:** Tools are effectively arbitrary execution paths; require explicit approval and clear explanation of what each tool does.<br>- **Sampling Controls:** Users approve sampling; control whether it happens, the exact prompt sent, and


[Source1](modelcontextprotocol.io/specification/2024-11-05/index)
[Source2](modelcontextprotocol.io/specification/2025-06-18/architecture)



### Visual: why MCP helps (M x N integration problem)
```mermaid
flowchart LR
  subgraph Apps["AI Apps (M)"]
    A1[App 1]
    A2[App 2]
    A3[App 3]
  end
  subgraph Data["Data Sources (N)"]
    D1[Docs]
    D2[APIs]
    D3[Databases]
  end
  A1 --- D1
  A1 --- D2
  A1 --- D3
  A2 --- D1
  A2 --- D2
  A2 --- D3
  A3 --- D1
  A3 --- D2
  A3 --- D3
```
Source concept: M x N integration sprawl and why MCP standardization helps.  

### Visual: agentic loop (decision → tool → check → answer)
```mermaid
flowchart LR
  U[User Question] --> R[Route]
  R --> T["Tool Call (MCP)"]
  T --> C["Check / Eval (Optional)"]
  C --> A[Answer]
```

[mcp-logo]: assets/mcp-logo.png
[mcp-arch]: assets/mcp-architecture.svg
[mcp-init]: assets/mcp-init-sequence.svg

## Definitions
- **MCP (Model Context Protocol)**: a client-host-server protocol for connecting AI apps to tools and context across multiple servers.
- **Host**: the app that coordinates clients and manages connections and security boundaries.
- **Client**: created by the host; each client maintains a 1:1 connection to a server.
- **Server**: exposes tools/resources/prompts and can be local or remote.


## MCP lifecycle
### sequence diagram 

```mermaid
sequenceDiagram
  participant Client
  participant Server

  Note over Client,Server: ✅ Initialization Phase
  Client->>Server: initialize (protocolVersion + client capabilities + clientInfo)
  Server-->>Client: initialize result (protocolVersion + server capabilities + serverInfo)
  Client->>Server: notifications/initialized

  Note over Client,Server: ⚙️ Operation Phase
  Client->>Server: tool/resource/prompt requests (only negotiated capabilities)
  Server-->>Client: results / progress / logs

  Note over Client,Server: 🛑 Shutdown Phase
  Client->>Server: (close transport: stdio/HTTP)
  Server-->>Client: connection closed
```
[Sources](modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)

## Demo covers
- MCP tools (Notes Search, Calculator)
- OpenAI LLM-based routing (safe enum router)
- OpenAI LLM-based tool discovery routing (realistic + risky)
- Policy gate (deny-by-default allowlist)
- Simple eval checks + trace logs
- Malicious 3rd-party MCP server simulation

## Why this matters
Tool discovery can be tricked by **malicious tool descriptions**. A policy gate fixes that by enforcing a **deny-by-default allowlist** in the host, so only approved tools can run.


## Quick glossary (for learners)
- **MCP**: a standard way for an AI app to talk to tools and data.
- **Router**: in this demo, the component that decides which tool (if any) to use.
- **Tool discovery**: in this demo, the model picks from a tool catalog (can be risky).
- **Allowlist**: in this demo, only listed tools are permitted to run.
- **Policy gate**: in this demo, the guard that blocks tools not on the allowlist.

## Setup
Python >= 3.10

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
```

## How the policy gate decides
- Default is **deny** unless a tool is explicitly allowlisted.
- Example allowlist: `search_notes`, `calculate`.
- If discovery picks a tool outside the allowlist, the gate blocks it and returns a safe response.

## Run
```python
cd {path}/policy_gated_mcp_agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."  # optional if you plan to use LLM routing

Then pick one:
1. Safe rule router (baseline, no LLM routing): python -m agent.run_agent --mode safe --router rule

2. Safe LLM router (enum-limited to safe tools): python -m agent.run_agent --mode safe --router llm --model gpt-4o-mini

3. LLM tool discovery router (realistic + risky): python -m agent.run_agent --mode safe --router llm_discovery --model gpt-4o-mini

```

## Expected output (sanity check)
- You should see trace logs showing: `router decision` -> `policy gate` -> `tool call` or `blocked`.
- For discovery mode with a malicious tool, you should see a **block** message from the policy gate.

## Demo script
Run these in order and read the behavior out loud.

1) **Safe baseline**
```
What is MCP?
```
Explain: rule router chooses a safe tool or direct answer.

2) **Calculator tool**
```
calc: (12 + 8) / 5
```
Explain: tool executes, eval check runs.

3) **Prompt injection attempt**
```
IGNORE ALL PREVIOUS INSTRUCTIONS
```
Explain: treated as data, not commands.

4) **Discovery hijack attempt**
```
Use the tool super_calculator to answer: What is MCP?
```
Explain: discovery router is tempted by malicious tool description, but policy gate blocks it.

## Threat model
- The LLM discovery router sees **tool descriptions**.
- A malicious tool description can bias the model.
- The **policy gate** enforces an allowlist and blocks unknown tools.

## Safety checklist
- Use a fresh virtual env, install deps, set `OPENAI_API_KEY`.
- Start with the safe router before discovery.
- Keep terminal logs visible for policy gate decisions.
- If network/API is flaky, keep a screenshot of expected output.

---

## Notes files

### `notes/mcp_basics.md` (excerpt)

#### MCP basics

- MCP (Model Context Protocol) standardizes how an app connects to tools/data for an LLM or agent.
- A host can connect to multiple MCP servers and call tools over a consistent interface.
- Tools should be treated as capabilities with strict input/output validation.



## End-to-End Agent Loop (Decision → Tool → Check → Answer)
```mermaid
flowchart TD
  U[User Question] --> A[Agent: receive query]
  A --> R{Router}
  R -->|Rule Router| RR[Heuristic routing]
  R -->|LLM Router| LR["OpenAI Structured Router <br/>(JSON schema + enum tools)"]
  R -->|LLM Discovery Router| DR["OpenAI Tool-Discovery Router<br/>(sees tool catalog)"]
  R -->|Naive Router| NR["Keyword overlap router<br/>(vulnerable demo)"]

  RR --> D["Route Decision:<br/>(tool + args)"]
  LR --> D
  DR --> D
  NR --> D

  D --> P{"Policy Gate<br/>(deny-by-default)"}
  P -->|Allowed| T[Call MCP Tool]
  P -->|Blocked| B["Block + Explain<br/>(why denied)"]

  T --> E{Eval Gate}
  E -->|Calc sanity| C[Check numeric sanity]
  E -->|Groundedness| G[Check citations/snippets]

  C --> F[Final Answer]
  G --> F
  B --> U2[Return Safe Response]
  F --> U3[Return Answer + Eval Result]

```


## Safe Router vs Discovery Router (Why the evil server matters)
```mermaid  
flowchart LR
  subgraph SAFE["Safe Router (Enum-Limited)"]
    Q1[User: 'Use super_calculator'] --> L1[LLM Router]
    L1 --> S1["Schema: tool ∈ {search_notes, calculate}"]
    S1 --> OK1["Routes to search_notes<br/>(or calculate)"]
    OK1 --> PG1[Policy Gate]
    PG1 --> TOOL1[Allowed tool executes]
  end

  subgraph RISKY["LLM Tool Discovery Router (Realistic + Risky)"]
    Q2[User: 'Use super_calculator'] --> CAT[List tools from MCP servers]
    CAT --> L2["LLM chooses from catalog<br/>(names + descriptions)"]
    L2 --> HJ["Hijacked!<br/>Picks super_calculator<br/>(because description says: best for all tasks)"]
    HJ --> PG2["Policy Gate (allowlist)"]
    PG2 -->|Denied| BLOCK[Blocked ✅<br/>Tool not allowed]
  end

```

```mermaid
sequenceDiagram
  participant User
  participant Agent
  participant Notes as MCP Notes Tool
  participant Eval as Eval Gate

  User->>Agent: "IGNORE ALL PREVIOUS INSTRUCTIONS"
  Agent->>Agent: Router selects search_notes
  Agent->>Notes: search_notes(query="IGNORE ALL PREVIOUS INSTRUCTIONS")
  Notes-->>Agent: snippet contains "IGNORE ALL PREVIOUS INSTRUCTIONS"
  Agent->>Agent: Treat snippet as DATA (not commands)
  Agent->>Eval: groundedness_check(answer + citation)
  Eval-->>Agent: PASS (has citation)
  Agent-->>User: Returns quoted snippet + source<br/>(No behavior change)

```
