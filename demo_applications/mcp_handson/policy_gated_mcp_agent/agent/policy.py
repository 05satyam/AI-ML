from __future__ import annotations

import re
from typing import Any, Dict, Tuple

# Deny-by-default: ONLY these tools can be called by the agent.
ALLOWED_TOOLS = {"search_notes", "calculate"}

# Strict allowlist for calculator input.
# Blocks letters, quotes, underscores, etc.
_ALLOWED_EXPR = re.compile(r"^[0-9\s\+\-\*\/\%\(\)\.]+$")


def validate_tool_call(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
    """Zero-trust-style policy gate for tool calls.

    - Deny-by-default: block any tool outside ALLOWED_TOOLS
    - Validate arguments for allowed tools (length, character allowlist)
    """
    if tool_name not in ALLOWED_TOOLS:
        return False, f"Tool not allowed: {tool_name}"

    if tool_name == "calculate":
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return False, "Empty expression."
        if len(expr) > 80:
            return False, "Expression too long (max 80 chars)."
        if not _ALLOWED_EXPR.match(expr):
            return False, "Expression contains disallowed characters."
        # Keep workshop simple
        if "**" in expr:
            return False, "Exponent operator (**) not allowed in this workshop."
        return True, "ok"

    if tool_name == "search_notes":
        q = str(args.get("query", "")).strip()
        if not q:
            return False, "Empty query."
        if len(q) > 120:
            return False, "Query too long (max 120 chars)."
        return True, "ok"

    return False, "No validation rule."
