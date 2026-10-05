"""Command line interface: validate a schema, make one decision, or evaluate a dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from decisionlab.decider import Decider
from decisionlab.evaluate import evaluate, load_dataset
from decisionlab.providers import get_provider
from decisionlab.schema import load_schema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decisionlab", description="Define, run, and measure structured AI decisions."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check that a schema file is well formed")
    validate.add_argument("--schema", required=True, help="path to a schema YAML file")

    def add_run_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--schema", required=True, help="path to a schema YAML file")
        sub.add_argument("--provider", default="mock", help="mock or anthropic[:model]")
        sub.add_argument("--retries", type=int, default=2, help="retries after an invalid answer")
        sub.add_argument(
            "--abstain-below", type=float, default=None, help="abstain under this confidence"
        )

    decide = commands.add_parser("decide", help="make one decision for a message")
    add_run_options(decide)
    decide.add_argument("message", help="the message to decide on")

    evaluate_cmd = commands.add_parser("eval", help="score a provider against a dataset")
    add_run_options(evaluate_cmd)
    evaluate_cmd.add_argument("--dataset", required=True, help="path to a JSONL dataset")
    evaluate_cmd.add_argument("--json", action="store_true", help="print the report as JSON")
    evaluate_cmd.add_argument(
        "--min-exact", type=float, default=None, help="exit 1 if exact match is below this"
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema)
    if args.command == "validate":
        print(f"OK: {schema.name} ({len(schema.fields)} fields, {len(schema.rules)} rules)")
        print()
        print(schema.describe())
        return 0

    decider = Decider(
        schema,
        get_provider(args.provider),
        max_retries=args.retries,
        abstain_below=args.abstain_below,
    )
    if args.command == "decide":
        decision = decider.decide(args.message)
        output = {
            "decision": decision.values,
            "confidence": decision.confidence,
            "abstained": decision.abstained,
            "reasoning": decision.reasoning,
            "errors": decision.errors,
        }
        print(json.dumps(output, indent=2))
        return 0 if decision.valid else 1

    report = evaluate(decider, load_dataset(args.dataset))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.format_report())
    if args.min_exact is not None and report.exact_match < args.min_exact:
        print(
            f"exact match {report.exact_match:.3f} is below --min-exact {args.min_exact}",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    # SchemaError and ValidationError are ValueErrors.
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
