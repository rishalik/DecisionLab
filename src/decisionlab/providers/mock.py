"""Offline keyword baseline, so tests and CI need no API key.

It only knows the support-ticket triage fields used by examples/billing_support, and it is
intentionally crude. Do not tune it to the dataset: its job is to be a fixed, honest floor.
"""

from __future__ import annotations

from typing import Any

# First matching group wins.
_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, str]], ...] = (
    (
        ("charged twice", "double charge", "duplicate charge", "refund"),
        ("billing", "duplicate_charge"),
    ),
    (("invoice", "receipt", "payment method", "card"), ("billing", "other_billing")),
    (("password", "log in", "login", "locked out"), ("account", "access")),
    (("crash", "error", "bug", "not working", "broken"), ("technical", "bug")),
    (("cancel", "unsubscribe"), ("account", "cancellation")),
)
_URGENT_WORDS = ("urgent", "asap", "immediately", "twice", "locked out", "can't", "cannot")
_TEAMS = {"billing": "billing", "technical": "engineering"}


class MockProvider:
    name = "mock"

    def complete(self, system: str, message: str) -> dict[str, Any]:
        text = message.lower().replace("’", "'")
        intent, sub_category, matched = "general", "other", None
        for keywords, labels in _KEYWORDS:
            matched = next((k for k in keywords if k in text), None)
            if matched:
                intent, sub_category = labels
                break
        urgency = min(5, 2 + sum(word in text for word in _URGENT_WORDS))
        return {
            "intent": intent,
            "sub_category": sub_category,
            "urgency": urgency,
            "escalate": urgency >= 4,
            "team": _TEAMS.get(intent, "support"),
            "confidence": 0.6 if intent == "general" else 0.85,
            "reasoning": f"keyword baseline: matched '{matched}'"
            if matched
            else "keyword baseline: no keyword matched",
        }
