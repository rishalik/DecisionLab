"""Run a provider against a schema, retrying with the validation errors fed back."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from decisionlab.providers.base import Provider
from decisionlab.schema import DecisionSchema, ValidationError

_INSTRUCTIONS = (
    'Also include "confidence" (a number from 0 to 1: how likely your decision is correct) '
    'and "reasoning" (one short sentence).\n'
    "Respond with a single JSON object only."
)


@dataclass
class Decision:
    values: dict[str, Any]
    confidence: float | None = None
    reasoning: str | None = None
    attempts: int = 1
    abstained: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _clean_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return None
    return min(1.0, max(0.0, float(value)))


class Decider:
    def __init__(
        self,
        schema: DecisionSchema,
        provider: Provider,
        max_retries: int = 2,
        abstain_below: float | None = None,
    ):
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.schema = schema
        self.provider = provider
        self.max_retries = max_retries
        self.abstain_below = abstain_below

    def system_prompt(self) -> str:
        return f"{self.schema.describe()}\n\n{_INSTRUCTIONS}"

    def decide(self, message: str) -> Decision:
        system = self.system_prompt()
        prompt = message
        errors: list[str] = []
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            raw = self.provider.complete(system, prompt)
            try:
                values = self.schema.validate(raw)
            except ValidationError as exc:
                errors = exc.errors
                prompt = (
                    f"{message}\n\nYour previous answer was invalid: {'; '.join(errors)}. Fix it."
                )
                continue
            confidence = _clean_confidence(raw.get("confidence"))
            reasoning = raw.get("reasoning")
            return Decision(
                values=values,
                confidence=confidence,
                reasoning=reasoning if isinstance(reasoning, str) else None,
                attempts=attempt,
                abstained=(
                    self.abstain_below is not None
                    and confidence is not None
                    and confidence < self.abstain_below
                ),
            )
        return Decision(values={}, attempts=attempts, errors=errors)
