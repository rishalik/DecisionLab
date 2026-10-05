from pathlib import Path

import pytest

from decisionlab.schema import SchemaError, ValidationError, load_schema, parse_schema

EXAMPLE = Path(__file__).parent.parent / "examples" / "billing_support" / "schema.yaml"

VALID = {
    "intent": "billing",
    "sub_category": "duplicate_charge",
    "urgency": 4,
    "escalate": True,
    "team": "billing",
}


@pytest.fixture(scope="module")
def schema():
    return load_schema(EXAMPLE)


def test_example_loads(schema):
    assert schema.name == "support_ticket_triage"
    assert len(schema.fields) == 5
    assert len(schema.rules) == 2


def test_valid_decision_passes_and_drops_extras(schema):
    assert schema.validate({**VALID, "confidence": 0.9, "reasoning": "x"}) == VALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent", "sales"),
        ("urgency", 6),
        ("urgency", 0),
        ("urgency", "4"),
        ("urgency", 4.0),
        ("urgency", True),
        ("escalate", "yes"),
        ("escalate", 1),
    ],
)
def test_bad_values_rejected(schema, field, value):
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({**VALID, field: value})
    assert field in str(excinfo.value)


def test_missing_field_rejected(schema):
    raw = {k: v for k, v in VALID.items() if k != "team"}
    with pytest.raises(ValidationError, match="missing field 'team'"):
        schema.validate(raw)


def test_all_problems_reported(schema):
    with pytest.raises(ValidationError) as excinfo:
        schema.validate({"intent": "sales", "urgency": 9, "escalate": True})
    assert len(excinfo.value.errors) == 4  # two bad values, two missing fields


def test_non_object_rejected(schema):
    with pytest.raises(ValidationError):
        schema.validate(["billing"])


def test_urgency_rule_enforced(schema):
    with pytest.raises(ValidationError, match="urgent_must_escalate"):
        schema.validate({**VALID, "escalate": False})
    assert schema.validate({**VALID, "urgency": 3, "escalate": False})


def test_billing_rule_enforced(schema):
    with pytest.raises(ValidationError, match="billing_goes_to_billing"):
        schema.validate({**VALID, "team": "support"})


def test_lte_operator():
    schema = parse_schema(
        {
            "name": "s",
            "fields": [{"name": "n", "type": "int"}, {"name": "low", "type": "bool"}],
            "rules": [{"name": "r", "when": {"n_lte": 2}, "then": {"low": True}}],
        }
    )
    assert schema.validate({"n": 3, "low": False})
    with pytest.raises(ValidationError):
        schema.validate({"n": 2, "low": False})


def test_describe_mentions_fields_and_rules(schema):
    text = schema.describe()
    for name in schema.field_names:
        assert name in text
    assert "urgency >= 4" in text


FIELD = {"name": "a", "type": "str"}


@pytest.mark.parametrize(
    "data",
    [
        {"name": "s"},
        {"name": "s", "fields": []},
        {"name": "s", "fields": [{"name": "a", "type": "float"}]},
        {"name": "s", "fields": [{"name": "a", "type": "enum"}]},
        {"name": "s", "fields": [{"name": "a", "type": "enum", "values": []}]},
        {"name": "s", "fields": [FIELD, FIELD]},
        {"name": "s", "fields": [FIELD], "rules": [{"when": {"b": 1}, "then": {"a": "x"}}]},
        {"name": "s", "fields": [FIELD], "rules": [{"when": {"a": "x"}, "then": {"b_gte": 1}}]},
    ],
    ids=[
        "no-fields",
        "empty-fields",
        "unknown-type",
        "enum-without-values",
        "enum-empty-values",
        "duplicate-names",
        "rule-unknown-field-when",
        "rule-unknown-field-then",
    ],
)
def test_malformed_schema_raises(data):
    with pytest.raises(SchemaError):
        parse_schema(data)
