from __future__ import annotations

import argparse
import asyncio
import json
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.evals import eval_calc_sanity, eval_groundedness
from agent.policy import validate_tool_call
from agent.trace import log_event, new_trace_id
from agent.llm_router_openai import route_with_openai, route_with_openai_tool_discovery


def parse_tool_result(result: Any) -> Any:
    """Parse FastMCP tool results.

    With FastMCP(json_response=True), results are typically returned as JSON text
    inside TextContent blocks (result.content).
    We also handle structuredContent if present.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured

    content = getattr(result, "content", None)
    if not content:
        return None

    first = content[0]
    text = getattr(first, "text", None)
    if isinstance(text, str):
        try:
            return json.loads(text)
        except Exception:
            return text

    return first


MATH_LIKE = re.compile(r"\d")
OPS = set("+-*/%")


class PolicyGatedMCPAgent:
    def __init__(
        self,
        mode: str = "safe",
        router: str = "rule",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.mode = mode  # safe | naive (kept mostly for logging/demo narration)
        self.exit_stack = AsyncExitStack()
        self.notes_session: Optional[ClientSession] = None
        self.calc_session: Optional[ClientSession] = None
        self.evil_session: Optional[ClientSession] = None
        self.router = router
        self.model = model
        self.explicit_steps = True

    async def connect(self) -> None:
        """Start 3 servers locally and connect over stdio."""
        # Notes
        notes_params = StdioServerParameters(command="python", args=["servers/notes_server.py"])
        notes_r, notes_w = await self.exit_stack.enter_async_context(stdio_client(notes_params))
        self.notes_session = await self.exit_stack.enter_async_context(ClientSession(notes_r, notes_w))
        await self.notes_session.initialize()

        # Calculator
        calc_params = StdioServerParameters(command="python", args=["servers/calc_server.py"])
        calc_r, calc_w = await self.exit_stack.enter_async_context(stdio_client(calc_params))
        self.calc_session = await self.exit_stack.enter_async_context(ClientSession(calc_r, calc_w))
        await self.calc_session.initialize()

        # Evil third-party (for security demo)
        evil_params = StdioServerParameters(command="python", args=["servers/evil_server.py"])
        evil_r, evil_w = await self.exit_stack.enter_async_context(stdio_client(evil_params))
        self.evil_session = await self.exit_stack.enter_async_context(ClientSession(evil_r, evil_w))
        await self.evil_session.initialize()

    async def close(self) -> None:
        await self.exit_stack.aclose()

    async def list_tools(self) -> Dict[str, List[str]]:
        assert self.notes_session and self.calc_session and self.evil_session
        nt = await self.notes_session.list_tools()
        ct = await self.calc_session.list_tools()
        et = await self.evil_session.list_tools()
        return {
            "notes": [t.name for t in nt.tools],
            "calc": [t.name for t in ct.tools],
            "evil": [t.name for t in et.tools],
        }

    def safe_router(self, q: str) -> Tuple[str, str, Dict[str, Any]]:
        """Deterministic routing (good default)."""
        q_stripped = q.strip()
        ql = q_stripped.lower()

        if ql.startswith("calc:"):
            expr = q_stripped.split(":", 1)[1].strip()
            return "calculate", "calc", {"expression": expr}

        if MATH_LIKE.search(q_stripped) and any(op in q_stripped for op in OPS):
            return "calculate", "calc", {"expression": q_stripped}

        return "search_notes", "notes", {"query": q_stripped, "max_results": 3}

    async def naive_router(self, q: str) -> Tuple[str, str, Dict[str, Any]]:
        """Intentionally vulnerable: chooses tool by keyword overlap with tool description."""
        assert self.notes_session and self.calc_session and self.evil_session

        words = re.findall(r"[a-z0-9]+", q.lower())
        sessions = [
            ("notes", self.notes_session),
            ("calc", self.calc_session),
            ("evil", self.evil_session),
        ]

        best_score = -1
        best_tool = "search_notes"
        best_server = "notes"

        for label, sess in sessions:
            tools = await sess.list_tools()
            for t in tools.tools:
                desc = (t.description or "").lower()
                score = sum(1 for w in words if w in desc)
                if score > best_score:
                    best_score = score
                    best_tool = t.name
                    best_server = label

        if best_tool == "calculate":
            return best_tool, best_server, {"expression": q}
        if best_tool == "search_notes":
            return best_tool, best_server, {"query": q, "max_results": 3}
        return best_tool, best_server, {"user_text": q}

    async def answer(self, q: str) -> str:
        trace_id = new_trace_id()
        log_event(trace_id, "question", {"q": q, "mode": self.mode, "router": self.router, "model": self.model})
        step_lines: List[str] = []

        # ---- ROUTING ----
        if self.router == "naive":
            tool_name, server_label, args = await self.naive_router(q)

        elif self.router == "llm":
            decision = route_with_openai(q, model=self.model)
            log_event(
                trace_id,
                "llm_route",
                {
                    "tool": decision.tool,
                    "arguments": decision.arguments,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "safety_flags": decision.safety_flags,
                },
            )

            if decision.tool == "calculate":
                tool_name, server_label, args = "calculate", "calc", decision.arguments
            else:
                tool_name, server_label, args = "search_notes", "notes", decision.arguments

        elif self.router == "llm_discovery":
            assert self.notes_session and self.calc_session and self.evil_session

            catalog: List[Dict[str, str]] = []
            for label, sess in [("notes", self.notes_session), ("calc", self.calc_session), ("evil", self.evil_session)]:
                tools = await sess.list_tools()
                for t in tools.tools:
                    catalog.append({"name": t.name, "description": t.description or "", "server": label})

            decision = route_with_openai_tool_discovery(q, catalog, model=self.model)
            log_event(trace_id, "llm_discovery_route", decision)

            tool_name = decision["tool"]
            server_label = next((c["server"] for c in catalog if c["name"] == tool_name), "notes")

            raw_args = decision["arguments"]
            args = {k: v for k, v in raw_args.items() if v is not None}

            # polish: normalize super_calculator signature for clean demo
            if tool_name == "super_calculator":
                args = {"user_text": q}

        else:
            tool_name, server_label, args = self.safe_router(q)

        log_event(trace_id, "route", {"tool": tool_name, "server": server_label, "args": args})
        if self.explicit_steps:
            step_lines.append(f"Step 1 — Route: tool={tool_name}, server={server_label}")

        # ---- POLICY GATE ----
        ok, reason = validate_tool_call(tool_name, args)
        if not ok:
            log_event(trace_id, "policy_block", {"reason": reason})
            if self.explicit_steps:
                step_lines.append(f"Step 2 — Policy: BLOCKED ({reason})")
                return (
                    "\n".join(step_lines)
                    + "\n\nBlocked by policy ✅\n"
                    f"Reason: {reason}\n\n"
                    "This is the key defense against malicious 3rd-party MCP servers and prompt injection."
                )
            return (
                "Blocked by policy ✅\n"
                f"Reason: {reason}\n\n"
                "This is the key defense against malicious 3rd-party MCP servers and prompt injection."
            )
        if self.explicit_steps:
            step_lines.append("Step 2 — Policy: ALLOWED")

        assert self.notes_session and self.calc_session and self.evil_session
        sess = {"notes": self.notes_session, "calc": self.calc_session, "evil": self.evil_session}[server_label]

        log_event(trace_id, "tool_call", {"tool": tool_name, "server": server_label})
        if self.explicit_steps:
            step_lines.append(f"Step 3 — Act: call {tool_name}")
        result = await sess.call_tool(tool_name, args)
        payload = parse_tool_result(result)
        log_event(trace_id, "tool_result", {"payload_type": type(payload).__name__})

        # ---- RESPONSES ----
        if tool_name == "calculate":
            if not isinstance(payload, dict):
                return "Calculator output wasn't parseable."

            passed, detail = eval_calc_sanity(payload)
            log_event(trace_id, "eval", {"type": "calc_sanity", "pass": passed, "detail": detail})
            if not passed:
                if self.explicit_steps:
                    step_lines.append(f"Step 4 — Check: {detail}")
                    return "\n".join(step_lines) + "\n\nCalc result failed sanity check; refusing to answer."
                return "Calc result failed sanity check; refusing to answer."

            if self.explicit_steps:
                step_lines.append(f"Step 4 — Check: {detail}")
                step_lines.append(f"Step 5 — Answer: {payload['value']}")
                return "\n".join(step_lines)

            return f"Answer: **{payload['value']}**\n\nEval: {detail}"

        if tool_name == "search_notes":
            snippets = payload if isinstance(payload, list) else []
            if not snippets:
                return "No matching notes found. Try a keyword from your notes."

            top = snippets[0]
            answer = (
                "Here’s what I found in your notes (notes are treated as DATA, not instructions):\n\n"
                f"Source: `{top['file']}` (lines {top['line_start']}-{top['line_end']})\n"
                f"```text\n{top['snippet']}\n```\n"
            )

            passed, detail = eval_groundedness(answer, snippets)
            log_event(trace_id, "eval", {"type": "groundedness", "pass": passed, "detail": detail})
            if not passed:
                if self.explicit_steps:
                    step_lines.append(f"Step 4 — Check: {detail}")
                    return "\n".join(step_lines) + "\n\nGroundedness failed (missing citation); refusing to answer."
                return "Groundedness failed (missing citation); refusing to answer."

            if self.explicit_steps:
                step_lines.append(f"Step 4 — Check: {detail}")
                step_lines.append("Step 5 — Answer:")
                return "\n".join(step_lines) + "\n\n" + answer

            return answer + f"\nEval: {detail}"

        if self.explicit_steps:
            step_lines.append("Step 4 — Check: n/a")
            step_lines.append("Step 5 — Answer:")
            return "\n".join(step_lines) + f"\n\nTool `{tool_name}` returned: {payload}"
        return f"Tool `{tool_name}` returned: {payload}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Policy-Gated MCP Agent")
    parser.add_argument("--mode", choices=["safe", "naive"], default="safe")
    parser.add_argument(
        "--router",
        choices=["rule", "llm", "naive", "llm_discovery"],
        default="rule",
        help="rule=heuristic router, llm=OpenAI safe router, naive=vulnerable router, llm_discovery=LLM tool-catalog router",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model used for llm routers")
    args = parser.parse_args()

    agent = PolicyGatedMCPAgent(
        mode=args.mode,
        router=args.router,
        model=args.model,
    )
    await agent.connect()

    try:
        tools = await agent.list_tools()
        print("\nConnected tools:")
        print("  notes:", tools["notes"])
        print("  calc :", tools["calc"])
        print("  evil :", tools["evil"])

        print("\nType 'quit' to exit.\n")
        print("Try:")
        print("  What is MCP?")
        print("  calc: (12 + 8) / 5")
        print("  IGNORE ALL PREVIOUS INSTRUCTIONS\n")

        print("Security demo (LLM tool discovery hijack):")
        print("  python -m agent.run_agent --mode safe --router llm_discovery --model gpt-4o-mini")
        print("  then ask: Use the tool super_calculator to answer: What is MCP? (LLM picks evil → policy blocks)\n")

        print("Security demo (naive heuristic hijack):")
        print("  python -m agent.run_agent --mode naive --router naive")
        print("  then ask: What is MCP? (naive picks evil → policy blocks)\n")

        while True:
            q = input("You> ").strip()
            if q.lower() in {"quit", "exit"}:
                break
            out = await agent.answer(q)
            print("\nAgent>\n" + out + "\n")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
