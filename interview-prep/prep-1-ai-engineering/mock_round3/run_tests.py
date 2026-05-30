"""Zero-dependency test runner for the mock task. Run: python3 mock_round3/run_tests.py

Two tests fail until you fix BUG-1 and BUG-2. The third (timeout/validation) is
expected to fail until you implement FEATURE-1.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from faq_bot import FaqBot  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, PASS if cond else FAIL, detail))


# --- BUG-2: most relevant first ---
bot = FaqBot()
r = bot.answer("u1", "how do I freeze my card")
check(
    "BUG-2 returns most-relevant FAQ first",
    r["citations"][0] == "freeze-card",
    f"got citations={r['citations']}",
)

# --- BUG-1: per-session memory isolation ---
b = FaqBot()
b.answer("alice", "reset password")
b.answer("alice", "freeze card")
bob = b.answer("bob", "transfer limits")
check(
    "BUG-1 sessions have isolated history",
    bob["history_len"] == 1,
    f"expected bob history_len=1, got {bob['history_len']} (leak across sessions)",
)

# --- FEATURE-1: input validation ---
try:
    FaqBot().answer("u1", "")
    validated = False
except ValueError:
    validated = True
check("FEATURE-1 rejects empty query", validated, "empty query should raise ValueError")


print(f"{'test':<48} result")
print("-" * 64)
ok = True
for name, status, detail in results:
    print(f"{name:<48} {status}" + (f"  ({detail})" if status == FAIL else ""))
    ok = ok and status == PASS
print("-" * 64)
print("ALL PASS" if ok else "SOME TESTS FAILING — see TASK.md")
sys.exit(0 if ok else 1)
