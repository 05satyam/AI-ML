"""Offline eval harness. Run: python -m eval.run_eval

Prints retrieval (recall@k, MRR) and generation (faithfulness) aggregates. Wire
this into CI so a prompt/model/chunking change that regresses quality fails the
build instead of shipping silently.
"""

from __future__ import annotations

import asyncio

from app.config import settings
from app.pipeline import RAGPipeline

from .dataset import EVAL_SET
from .metrics import faithfulness, recall_at_k, reciprocal_rank


async def main() -> None:
    pipe = RAGPipeline()
    await pipe.startup()

    recalls, mrrs, faiths = [], [], []
    print(f"{'question':<45} r@k   rr    faith")
    print("-" * 70)
    for ex in EVAL_SET:
        matches = await pipe._retriever.retrieve(ex["q"])  # type: ignore[union-attr]
        ids = [m.doc_id for m in matches]
        r = recall_at_k(ids, ex["relevant"], settings.top_k)
        rr = reciprocal_rank(ids, ex["relevant"])

        resp = await pipe.chat_turn(f"eval-{ex['q']}", ex["q"])
        context = " ".join(c.snippet for c in resp.citations)
        f = faithfulness(resp.answer, context)

        recalls.append(r)
        mrrs.append(rr)
        faiths.append(f)
        print(f"{ex['q'][:44]:<45} {r:.2f}  {rr:.2f}  {f:.2f}")

    n = len(EVAL_SET)
    print("-" * 70)
    print(
        f"{'AVERAGE':<45} {sum(recalls)/n:.2f}  {sum(mrrs)/n:.2f}  {sum(faiths)/n:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
