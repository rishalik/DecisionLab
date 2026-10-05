"""Decision schemas: typed fields plus cross-field consistency rules, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FIELD_TYPES = ("enum", "int", "bool", "str")
_OPERATORS = (("_gte", "gte"), ("_lte", "lte"))
_SYMBOLS = {"eq": "=", "gte": ">=", "lte": "<="}


class SchemaError(ValueError):
    """The schema definition itself is malformed."""


class ValidationError(ValueError):
    """A decision does not conform to the schema. `errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _show(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    description: str = ""
    values: tuple[Any, ...] = ()
    min: int | None = None
    max: int | None = None

    def check(self, value: Any) -> str | None:
        """Return an error message for `value`, or None if it is acceptable."""
        if self.type == "enum":
            # Compare by type as well, so True does not pass for an enum containing 1.
            if not any(type(value) is type(v) and value == v for v in self.values):
                allowed = ", ".join(_show(v) for v in self.values)
                return f"field '{self.name}': {value!r} is not one of [{allowed}]"
        elif self.type == "int":
            if not _is_int(value):
                return f"field '{self.name}': expected an integer, got {value!r}"
            if self.min is not None and value < self.min:
                return f"field '{self.name}': {value} is below the minimum {self.min}"
            if self.max is not None and value > self.max:
                return f"field '{self.name}': {value} is above the maximum {self.max}"
        elif self.type == "bool":
            if not isinstance(value, bool):
                return f"field '{self.name}': expected true or false, got {value!r}"
        elif not isinstance(value, str):
            return f"field '{self.name}': expected a string, got {value!r}"
        return None

    def describe(self) -> str:
        if self.type == "enum":
            kind = "one of: " + ", ".join(_show(v) for v in self.values)
        elif self.type == "int":
            kind = "integer"
            if self.min is not None and self.max is not None:
                kind += f" from {self.min} to {self.max}"
            elif self.min is not None:
                kind += f" >= {self.min}"
            elif self.max is not None:
                kind += f" <= {self.max}"
        elif self.type == "bool":
            kind = "boolean: true or false"
        else:
            kind = "string"
        line = f"- {self.name} ({kind})"
        return f"{line}: {self.description}" if self.description else line


@dataclass(frozen=True)
class Rule:
    """If every `when` condition matches the decision, every `then` condition must too."""

    name: str
    when: tuple[tuple[str, str, Any], ...]
    then: tuple[tuple[str, str, Any], ...]

    @staticmethod
    def _matches(conditions: tuple[tuple[str, str, Any], ...], decision: dict[str, Any]) -> bool:
        for field, op, target in conditions:
            value = decision[field]
            if op == "gte":
                ok = value >= target
            elif op == "lte":
                ok = value <= target
            else:
                ok = type(value) is type(target) and value == target
            if not ok:
                return False
        return True

    def check(self, decision: dict[str, Any]) -> str | None:
        if self._matches(self.when, decision) and not self._matches(self.then, decision):
            return f"rule '{self.name}' violated: {self.describe()}"
        return None

    def describe(self) -> str:
        def side(conditions: tuple[tuple[str, str, Any], ...]) -> str:
            return " and ".join(f"{f} {_SYMBOLS[op]} {_show(t)}" for f, op, t in conditions)

        return f"if {side(self.when)} then {side(self.then)}"


@dataclass(frozen=True)
class DecisionSchema:
    name: str
    description: str
    fields: tuple[Field, ...]
    rules: tuple[Rule, ...] = ()

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def validate(self, raw: Any) -> dict[str, Any]:
        """Return a dict holding only the schema's fields, or raise ValidationError.

        Every field problem is reported at once. Rules are checked only when all fields
        are individually valid, since a rule over a bad value is meaningless.
        """
        if not isinstance(raw, dict):
            raise ValidationError([f"expected a JSON object, got {type(raw).__name__}"])
        errors: list[str] = []
        cleaned: dict[str, Any] = {}
        for field in self.fields:
            if field.name not in raw:
                errors.append(f"missing field '{field.name}'")
                continue
            error = field.check(raw[field.name])
            if error:
                errors.append(error)
            else:
                cleaned[field.name] = raw[field.name]
        if not errors:
            errors = [e for e in (rule.check(cleaned) for rule in self.rules) if e]
        if errors:
            raise ValidationError(errors)
        return cleaned

    def describe(self) -> str:
        """Describe the expected output for a human or an LLM."""
        lines = [f"Decision: {self.name}"]
        if self.description:
            lines.append(self.description)
        lines += ["", "Return a JSON object with exactly these fields:"]
        lines += [field.describe() for field in self.fields]
        if self.rules:
            lines += ["", "The fields must be consistent with these rules:"]
            lines += [f"- {rule.name}: {rule.describe()}" for rule in self.rules]
        return "\n".join(lines)


def _parse_field(raw: Any, index: int) -> Field:
    if not isinstance(raw, dict):
        raise SchemaError(f"field #{index + 1} must be a mapping")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise SchemaError(f"field #{index + 1} needs a non-empty string 'name'")
    ftype = raw.get("type")
    if ftype not in FIELD_TYPES:
        raise SchemaError(
            f"field '{name}': unknown type {ftype!r} (expected one of {', '.join(FIELD_TYPES)})"
        )
    values: tuple[Any, ...] = ()
    if ftype == "enum":
        raw_values = raw.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise SchemaError(f"field '{name}': enum needs a non-empty 'values' list")
        values = tuple(raw_values)
    low, high = raw.get("min"), raw.get("max")
    if ftype == "int":
        for label, bound in (("min", low), ("max", high)):
            if bound is not None and not _is_int(bound):
                raise SchemaError(f"field '{name}': '{label}' must be an integer")
        if low is not None and high is not None and low > high:
            raise SchemaError(f"field '{name}': min {low} is greater than max {high}")
    else:
        low = high = None
    return Field(
        name=name,
        type=ftype,
        description=str(raw.get("description") or ""),
        values=values,
        min=low,
        max=high,
    )


def _parse_conditions(
    raw: Any, rule_name: str, part: str, fields: dict[str, Field]
) -> tuple[tuple[str, str, Any], ...]:
    if not isinstance(raw, dict) or not raw:
        raise SchemaError(f"rule '{rule_name}': '{part}' must be a non-empty mapping")
    conditions = []
    for key, target in raw.items():
        key = str(key)
        # An exact field name wins, so a field literally named "x_gte" stays addressable.
        field_name, op = key, "eq"
        if key not in fields:
            for suffix, operator in _OPERATORS:
                if key.endswith(suffix) and key[: -len(suffix)] in fields:
                    field_name, op = key[: -len(suffix)], operator
                    break
            else:
                raise SchemaError(f"rule '{rule_name}': '{key}' refers to an unknown field")
        if op != "eq" and (fields[field_name].type != "int" or not _is_int(target)):
            raise SchemaError(
                f"rule '{rule_name}': '{key}' needs an int field and an integer value"
            )
        conditions.append((field_name, op, target))
    return tuple(conditions)


def parse_schema(data: Any) -> DecisionSchema:
    if not isinstance(data, dict):
        raise SchemaError("schema must be a mapping")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise SchemaError("schema needs a non-empty string 'name'")
    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise SchemaError("schema needs a non-empty 'fields' list")
    fields: dict[str, Field] = {}
    for index, raw_field in enumerate(raw_fields):
        field = _parse_field(raw_field, index)
        if field.name in fields:
            raise SchemaError(f"duplicate field name '{field.name}'")
        fields[field.name] = field

    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        raise SchemaError("'rules' must be a list")
    rules = []
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise SchemaError(f"rule #{index + 1} must be a mapping")
        rule_name = str(raw_rule.get("name") or f"rule_{index + 1}")
        rules.append(
            Rule(
                name=rule_name,
                when=_parse_conditions(raw_rule.get("when"), rule_name, "when", fields),
                then=_parse_conditions(raw_rule.get("then"), rule_name, "then", fields),
            )
        )
    return DecisionSchema(
        name=name,
        description=str(data.get("description") or ""),
        fields=tuple(fields.values()),
        rules=tuple(rules),
    )


def load_schema(path: str | Path) -> DecisionSchema:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return parse_schema(data)
    except SchemaError as exc:
        raise SchemaError(f"{path}: {exc}") from exc
