from __future__ import annotations

from typing import Any, Dict, List, Tuple


_INJECTION_PATTERNS = (
    "ignore all previous instructions",
    "disregard previous instructions",
    "system prompt",
    "reveal secrets",
    "exfiltrate",
    "override",
)


def eval_groundedness(answer: str, snippets: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Groundedness gate:
    - Must have at least one snippet
    - Must cite a file that came from snippets
    - Must not contain obvious prompt-injection strings
    """
    if not snippets:
        return False, "FAIL: no snippets retrieved"

    files = {s.get("file", "") for s in snippets if s.get("file")}
    if not files:
        return False, "FAIL: snippets missing file fields"

    answer_l = answer.lower()
    if any(p in answer_l for p in _INJECTION_PATTERNS):
        return False, "FAIL: prompt-injection string detected in answer"

    if any(f in answer for f in files):
        return True, "PASS"

    return False, "FAIL"


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
