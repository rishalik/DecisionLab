from pathlib import Path

import pytest

from decisionlab.decider import Decider
from decisionlab.evaluate import Example, evaluate, expected_calibration_error, load_dataset
from decisionlab.providers import MockProvider
from decisionlab.schema import load_schema

EXAMPLE_DIR = Path(__file__).parent.parent / "examples" / "billing_support"

GOOD = {
    "intent": "billing",
    "sub_category": "duplicate_charge",
    "urgency": 4,
    "escalate": True,
    "team": "billing",
}


class ScriptedProvider:
    """Returns queued answers in order and records the messages it was sent."""

    name = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.messages = []

    def complete(self, system, message):
        self.messages.append(message)
        return self.answers.pop(0)


@pytest.fixture(scope="module")
def schema():
    return load_schema(EXAMPLE_DIR / "schema.yaml")


def test_retry_then_success_feeds_errors_back(schema):
    provider = ScriptedProvider([{**GOOD, "intent": "sales"}, {**GOOD, "confidence": 0.9}])
    decision = Decider(schema, provider).decide("charged twice")
    assert decision.valid
    assert decision.values == GOOD
    assert decision.attempts == 2
    assert provider.messages[0] == "charged twice"
    assert "Your previous answer was invalid" in provider.messages[1]
    assert "intent" in provider.messages[1]


def test_gives_up_after_max_retries(schema):
    provider = ScriptedProvider([{}, "garbage", None, GOOD])
    decision = Decider(schema, provider, max_retries=2).decide("hello")
    assert not decision.valid
    assert decision.attempts == 3
    assert decision.errors
    assert decision.values == {}
    assert len(provider.messages) == 3


def test_abstains_below_threshold(schema):
    low = Decider(schema, ScriptedProvider([{**GOOD, "confidence": 0.4}]), abstain_below=0.5)
    high = Decider(schema, ScriptedProvider([{**GOOD, "confidence": 0.6}]), abstain_below=0.5)
    assert low.decide("x").abstained
    assert not high.decide("x").abstained


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(7, 1.0), (-1, 0.0), (0.25, 0.25), ("high", None), (True, None), (None, None)],
)
def test_confidence_is_clamped(schema, raw, expected):
    provider = ScriptedProvider([{**GOOD, "confidence": raw}])
    assert Decider(schema, provider).decide("x").confidence == expected


def test_mock_on_example_dataset(schema):
    examples = load_dataset(EXAMPLE_DIR / "dataset.jsonl")
    report = evaluate(Decider(schema, MockProvider()), examples)
    assert report.total == 6
    assert report.invalid == 0
    assert report.field_accuracy["intent"] == 1.0
    # Regression floor, not a target: the mock is deliberately crude on urgency.
    assert report.exact_match >= 0.5
    assert list(report.field_accuracy) == schema.field_names
    assert report.ece is not None
    assert "Exact match" in report.format_report()
    assert report.to_dict()["total"] == 6


def test_invalid_decisions_count_as_wrong(schema):
    provider = ScriptedProvider([{}])
    report = evaluate(Decider(schema, provider, max_retries=0), [Example("hi", GOOD)])
    assert report.invalid == 1
    assert report.exact_match == 0.0
    assert report.field_accuracy["intent"] == 0.0
    assert report.confusion["intent"] == {("billing", None): 1}
    assert len(report.failures) == 1
    assert "invalid" in report.format_report()


def test_ece():
    assert expected_calibration_error([(1.0, True)] * 5) == 0
    assert expected_calibration_error([(0.9, False)] * 5) > 0.8
    assert expected_calibration_error([]) is None


def test_load_dataset_skips_blanks_and_reports_bad_rows(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text('{"message": "a", "expected": {}}\n\n{"message": "b", "expected": {}}\n')
    assert [e.message for e in load_dataset(good)] == ["a", "b"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"message": "a", "expected": {}}\n{"message": "b"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_dataset(bad)
