"""Claude via the Anthropic API. Requires the optional `anthropic` extra."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in `text`, or {} if there is none."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return {}


class AnthropicProvider:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the anthropic provider needs the 'anthropic' package: "
                'pip install "decisionlab[anthropic]"'
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, message: str) -> dict[str, Any]:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": message}],
            )
        except self._anthropic.AnthropicError as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc
        text = "".join(block.text for block in response.content if block.type == "text")
        return extract_json_object(text)
