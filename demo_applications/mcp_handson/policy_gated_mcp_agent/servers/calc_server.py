from __future__ import annotations

import ast
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calc-server", json_response=True)

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)


def safe_eval(expr: str) -> float:
    """Safely evaluate basic arithmetic.

    Supports: +, -, *, /, //, %, parentheses, unary +/-.  (No names, calls, etc.)
    """
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")

    value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    if not isinstance(value, (int, float)):
        raise ValueError("Expression did not evaluate to a number")
    return float(value)


@mcp.tool()
def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a safe arithmetic expression."""
    return {"expression": expression, "value": safe_eval(expression)}


if __name__ == "__main__":
    mcp.run("stdio")
