import json
from pathlib import Path

from decisionlab.cli import main

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "billing_support"
SCHEMA = str(EXAMPLE_DIR / "schema.yaml")
DATASET = str(EXAMPLE_DIR / "dataset.jsonl")


def test_validate(capsys):
    assert main(["validate", "--schema", SCHEMA]) == 0
    assert "support_ticket_triage" in capsys.readouterr().out


def test_validate_missing_file(capsys):
    assert main(["validate", "--schema", "does-not-exist.yaml"]) == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_decide(capsys):
    assert main(["decide", "--schema", SCHEMA, "I was charged twice, please refund me"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["decision"]["intent"] == "billing"
    assert output["decision"]["team"] == "billing"
    assert output["errors"] == []
    assert set(output) == {"decision", "confidence", "abstained", "reasoning", "errors"}


def test_eval_json(capsys):
    assert main(["eval", "--schema", SCHEMA, "--dataset", DATASET, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["total"] == 6
    assert report["field_accuracy"]["intent"] == 1.0


def test_eval_min_exact_fails():
    assert main(["eval", "--schema", SCHEMA, "--dataset", DATASET, "--min-exact", "1.01"]) == 1


def test_unknown_provider(capsys):
    assert main(["decide", "--schema", SCHEMA, "--provider", "nope", "hello"]) == 2
    assert "unknown provider" in capsys.readouterr().err
