from __future__ import annotations

from typing import Any, Dict, List, Tuple


def eval_groundedness(answer: str, snippets: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Groundedness gate:
    - Must have at least one snippet
    - Must cite a file that came from snippets
    """
    if not snippets:
        return False, "FAIL: no snippets retrieved"

    files = {s.get("file", "") for s in snippets if s.get("file")}
    if not files:
        return False, "FAIL: snippets missing file fields"

    if any(f in answer for f in files):
        return True, "PASS: groundedness (has citation)"

    return False, "FAIL: missing citation"


def eval_calc_sanity(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Correctness/sanity gate for calc tool output."""
    if not isinstance(payload, dict):
        return False, "FAIL: calc payload not a dict"
    if "value" not in payload:
        return False, "FAIL: calc payload missing 'value'"
    try:
        float(payload["value"])
    except Exception:
        return False, "FAIL: calc value not numeric"
    return True, "PASS: calc sanity"
