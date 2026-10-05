"""Decision Lab: define, run, and measure structured AI decisions."""

from decisionlab.decider import Decider, Decision
from decisionlab.evaluate import (
    EvalReport,
    Example,
    evaluate,
    expected_calibration_error,
    load_dataset,
)
from decisionlab.providers import AnthropicProvider, MockProvider, Provider, get_provider
from decisionlab.schema import (
    DecisionSchema,
    Field,
    Rule,
    SchemaError,
    ValidationError,
    load_schema,
    parse_schema,
)

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "Decider",
    "Decision",
    "DecisionSchema",
    "EvalReport",
    "Example",
    "Field",
    "MockProvider",
    "Provider",
    "Rule",
    "SchemaError",
    "ValidationError",
    "evaluate",
    "expected_calibration_error",
    "get_provider",
    "load_dataset",
    "load_schema",
    "parse_schema",
]
