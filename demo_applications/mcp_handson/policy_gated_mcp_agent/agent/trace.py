from __future__ import annotations

import time
import uuid
from typing import Any, Dict


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def log_event(trace_id: str, event: str, payload: Dict[str, Any]) -> None:
    ts = time.strftime("%H:%M:%S")
    # Print to stdout is fine in the client (not in MCP servers)
    print(f"[{ts}] trace={trace_id} {event} {payload}")
