"""Lightweight structured tracing.

In production this would be OpenTelemetry spans exported to Langfuse / LangSmith /
Jaeger. The point for an interview: *every step of the RAG flow emits a span with
timing + key attributes* so you can debug "why did this answer happen?".
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attrs: Any):
        s = Span(name=name, start=time.time(), attrs=dict(attrs))
        self.spans.append(s)
        try:
            yield s
        finally:
            s.end = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [
                {"name": s.name, "duration_ms": s.duration_ms, "attrs": s.attrs}
                for s in self.spans
            ],
        }


def new_trace() -> Trace:
    return Trace(trace_id=uuid.uuid4().hex[:12])
