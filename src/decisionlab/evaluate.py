"""Score a Decider against a labelled dataset: accuracy, confusion, and calibration."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decisionlab.decider import Decider

MAX_FAILURES_SHOWN = 10


@dataclass(frozen=True)
class Example:
    message: str
    expected: dict[str, Any]


@dataclass
class Failure:
    index: int
    message: str
    expected: dict[str, Any]
    got: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    schema_name: str
    provider_name: str
    total: int
    invalid: int
    abstained: int
    field_accuracy: dict[str, float]
    exact_match: float
    confusion: dict[str, dict[tuple[Any, Any], int]]
    ece: float | None
    mean_attempts: float
    failures: list[Failure]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "provider": self.provider_name,
            "total": self.total,
            "invalid": self.invalid,
            "abstained": self.abstained,
            "field_accuracy": self.field_accuracy,
            "exact_match": self.exact_match,
            "confusion": {
                name: [
                    {"expected": expected, "predicted": predicted, "count": count}
                    for (expected, predicted), count in counts.items()
                ]
                for name, counts in self.confusion.items()
            },
            "ece": self.ece,
            "mean_attempts": self.mean_attempts,
            "failures": [
                {
                    "index": f.index,
                    "message": f.message,
                    "expected": f.expected,
                    "got": f.got,
                    "errors": f.errors,
                }
                for f in self.failures
            ],
        }

    def format_report(self) -> str:
        lines = [
            f"Decision Lab eval: {self.schema_name} (provider: {self.provider_name})",
            f"examples: {self.total}   invalid: {self.invalid}   abstained: {self.abstained}"
            f"   mean attempts: {self.mean_attempts:.2f}",
            "",
            "Field accuracy",
        ]
        width = max((len(name) for name in self.field_accuracy), default=0)
        for name, accuracy in self.field_accuracy.items():
            lines.append(f"  {name:<{width}}  {accuracy:6.1%}")
        lines += [
            "",
            f"Exact match: {self.exact_match:.1%}",
            "ECE: " + ("n/a" if self.ece is None else f"{self.ece:.3f}"),
        ]
        if self.failures:
            shown = self.failures[:MAX_FAILURES_SHOWN]
            lines += ["", f"Failures ({len(shown)} of {len(self.failures)} shown)"]
            for failure in shown:
                lines.append(f"  [{failure.index}] {failure.message}")
                if failure.got is None:
                    lines.append("      invalid: " + "; ".join(failure.errors))
                    continue
                for name, expected in failure.expected.items():
                    got = failure.got.get(name)
                    if not _same(expected, got):
                        lines.append(f"      {name}: expected {expected!r}, got {got!r}")
        return "\n".join(lines)


def _same(expected: Any, got: Any) -> bool:
    # Type-strict so that an expected 1 is not satisfied by a predicted True.
    return type(expected) is type(got) and expected == got


def load_dataset(path: str | Path) -> list[Example]:
    """Load JSONL rows of {"message": str, "expected": {...}}. Blank lines are skipped."""
    path = Path(path)
    examples = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{where}: row must be a JSON object")
            if not isinstance(row.get("message"), str):
                raise ValueError(f"{where}: 'message' must be a string")
            if not isinstance(row.get("expected"), dict):
                raise ValueError(f"{where}: 'expected' must be an object")
            examples.append(Example(message=row["message"], expected=row["expected"]))
    return examples


def expected_calibration_error(pairs: Iterable[tuple[float, bool]], bins: int = 10) -> float | None:
    """Weighted mean of |mean confidence - accuracy| over equal-width confidence bins.

    `pairs` are (confidence, was_correct). Returns None when there is nothing to measure.
    """
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    total = 0
    for confidence, correct in pairs:
        buckets[min(bins - 1, max(0, int(confidence * bins)))].append((confidence, correct))
        total += 1
    if not total:
        return None
    ece = 0.0
    for bucket in buckets:
        if bucket:
            mean_confidence = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
            ece += len(bucket) / total * abs(mean_confidence - accuracy)
    return ece


def evaluate(decider: Decider, examples: Iterable[Example]) -> EvalReport:
    """Run every example through the decider. Invalid decisions count as wrong."""
    examples = list(examples)
    names = decider.schema.field_names
    correct = Counter()
    labelled = Counter()
    confusion: dict[str, Counter] = {name: Counter() for name in names}
    failures: list[Failure] = []
    pairs: list[tuple[float, bool]] = []
    invalid = abstained = exact = attempts = 0

    for index, example in enumerate(examples, start=1):
        unknown = [name for name in example.expected if name not in names]
        if unknown:
            raise ValueError(
                f"example {index}: expected has fields not in the schema: {', '.join(unknown)}"
            )
        decision = decider.decide(example.message)
        attempts += decision.attempts
        invalid += not decision.valid
        abstained += decision.abstained
        all_correct = decision.valid
        for name, expected in example.expected.items():
            got = decision.values.get(name)
            hit = decision.valid and _same(expected, got)
            labelled[name] += 1
            correct[name] += hit
            confusion[name][(expected, got)] += 1
            all_correct = all_correct and hit
        exact += all_correct
        if decision.confidence is not None:
            pairs.append((decision.confidence, all_correct))
        if not all_correct:
            failures.append(
                Failure(
                    index=index,
                    message=example.message,
                    expected=example.expected,
                    got=decision.values if decision.valid else None,
                    errors=decision.errors,
                )
            )

    total = len(examples)
    return EvalReport(
        schema_name=decider.schema.name,
        provider_name=decider.provider.name,
        total=total,
        invalid=invalid,
        abstained=abstained,
        field_accuracy={
            name: correct[name] / labelled[name] if labelled[name] else 0.0 for name in names
        },
        exact_match=exact / total if total else 0.0,
        confusion={name: dict(counts) for name, counts in confusion.items()},
        ece=expected_calibration_error(pairs),
        mean_attempts=attempts / total if total else 0.0,
        failures=failures,
    )
