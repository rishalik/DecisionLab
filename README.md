# Decision Lab

Define, run, and measure structured AI decisions, and check whether those decisions and their
confidence can be trusted.

Teams increasingly let a model make small operational calls: what is this message about, how
urgent is it, should a human look, who owns it. Decision Lab treats each of those calls as a
typed, testable decision rather than free text.

Take one support message:

> I was charged twice for my subscription and I need the second charge refunded.

The decision you want back is not a paragraph. It is this:

| Field          | Value              | Checked by                                 |
| -------------- | ------------------ | ------------------------------------------ |
| `intent`       | `billing`          | enum                                       |
| `sub_category` | `duplicate_charge` | enum                                       |
| `urgency`      | `4`                | int, 1 to 5                                |
| `escalate`     | `true`             | bool, and rule: urgency >= 4 must escalate |
| `team`         | `billing`          | enum, and rule: billing intent goes to billing |
| `confidence`   | `0.85`             | calibration (ECE) across a labelled dataset |

Decision Lab validates every field, enforces the cross-field rules, retries with the errors fed
back when the model gets it wrong, and then tells you how often the decisions were right and
whether the stated confidence meant anything.

## Features

- **YAML decision schemas** with `enum`, `int` (min/max), `bool`, and `str` fields.
- **Cross-field rules**: `when` / `then` consistency checks with exact match, `_gte`, and `_lte`.
- **Strict validation** that reports every problem at once (`true` is not an int, `"yes"` is not a bool).
- **Retry loop** that sends the validation errors back to the model.
- **Abstention** below a confidence threshold.
- **Evaluation** on a JSONL dataset: per-field accuracy, exact match, confusion counts, expected
  calibration error (ECE), mean attempts, and a list of failures.
- **Providers**: an offline keyword mock and Anthropic Claude. One small protocol to add your own.
- **CI friendly**: `--min-exact` fails the build when accuracy drops; no API key needed for the mock.

## Quick start

```bash
pip install -e .                 # or: pip install -e ".[anthropic]"

decisionlab validate --schema examples/billing_support/schema.yaml

decisionlab decide --schema examples/billing_support/schema.yaml \
  "I was charged twice for my subscription and I need the second charge refunded."

decisionlab eval --schema examples/billing_support/schema.yaml \
  --dataset examples/billing_support/dataset.jsonl
```

To use Claude instead of the mock, set `ANTHROPIC_API_KEY` and pass
`--provider anthropic` (or `--provider anthropic:<model>`).

Other options: `--retries N`, `--abstain-below 0.7`, and for `eval`, `--json` and `--min-exact 0.8`.

Exit codes: `0` success, `1` an invalid decision or exact match below `--min-exact`, `2` an
error such as a bad schema, a missing file, or an unknown provider.

## Sample eval output

This is the real output of the `eval` command above, using the mock provider:

```text
Decision Lab eval: support_ticket_triage (provider: mock)
examples: 6   invalid: 0   abstained: 0   mean attempts: 1.00

Field accuracy
  intent        100.0%
  sub_category  100.0%
  urgency        50.0%
  escalate       83.3%
  team          100.0%

Exact match: 50.0%
ECE: 0.308

Failures (3 of 3 shown)
  [1] I was charged twice for my subscription and I need the second charge refunded.
      urgency: expected 4, got 3
      escalate: expected True, got False
  [4] The app crashes every time I open the export page.
      urgency: expected 3, got 2
  [6] Do you have a dark mode planned?
      urgency: expected 1, got 2
```

**About the mock.** It is a keyword matcher, not a model, and it is a rough baseline on purpose.
It gets half of the example dataset fully right, it misjudges urgency on the double-charge
message from the pitch above, and its confidence is a fixed number that says nothing about the
input (hence the poor ECE). It exists so the tests and CI run offline, and so there is a floor to
compare a real model against. It is not tuned to the dataset and should not be.

## Python API

```python
from decisionlab import Decider, evaluate, get_provider, load_dataset, load_schema

schema = load_schema("examples/billing_support/schema.yaml")
decider = Decider(schema, get_provider("mock"), max_retries=2, abstain_below=0.7)

decision = decider.decide("I'm locked out of my account and can't log in!")
print(decision.valid, decision.values, decision.confidence, decision.abstained)

report = evaluate(decider, load_dataset("examples/billing_support/dataset.jsonl"))
print(report.format_report())
print(report.exact_match, report.ece)
```

A provider is any object with a `name` and a `complete(system, message) -> dict` method. It can
return garbage; the `Decider` validates whatever comes back.

## Roadmap

See [REQUIREMENTS.md](REQUIREMENTS.md) for who this is for, what v0.1 covers, and what is planned.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## License

MIT. See [LICENSE](LICENSE).
