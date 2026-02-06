# MCP basics

- MCP (Model Context Protocol) standardizes how an app connects to tools/data for an LLM or agent.
- An MCP **client** connects to one or more MCP **servers**.
- Servers expose **tools**, **resources**, and **prompts**.
- Security principle: treat tool outputs + retrieved text as **untrusted data**, not instructions.
