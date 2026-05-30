"""Tiny seed corpus. In production this is your ingestion pipeline output
(chunked, embedded, written to a vector DB). Kept inline so the demo is runnable.
"""

from __future__ import annotations

DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "sofi-plus",
        "text": (
            "SoFi Plus is SoFi's premium membership tier. Members get higher savings APY, "
            "rewards boosts, and access to financial planning. It is included for members "
            "with qualifying direct deposit."
        ),
    },
    {
        "id": "sofi-plus-cost",
        "text": (
            "SoFi Plus has no separate subscription fee when you qualify via direct deposit. "
            "Otherwise it is available for a monthly membership cost. Pricing can change; "
            "check the SoFi app for current terms."
        ),
    },
    {
        "id": "vault",
        "text": (
            "Vaults let SoFi members organize savings into separate goals within one account, "
            "each tracked toward a target amount."
        ),
    },
    {
        "id": "fraud-alerts",
        "text": (
            "SoFi sends real-time fraud alerts for suspicious card transactions. Members can "
            "freeze and unfreeze their card instantly from the app."
        ),
    },
    {
        "id": "loan-rates",
        "text": (
            "SoFi personal loan rates depend on creditworthiness, loan term, and amount. "
            "Autopay can lower the rate. There are no origination fees on personal loans."
        ),
    },
]
