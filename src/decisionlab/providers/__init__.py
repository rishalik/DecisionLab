"""Decision providers and the `get_provider` factory."""

from __future__ import annotations

from decisionlab.providers.anthropic import AnthropicProvider
from decisionlab.providers.base import Provider
from decisionlab.providers.mock import MockProvider

__all__ = ["AnthropicProvider", "MockProvider", "Provider", "get_provider"]


def get_provider(spec: str) -> Provider:
    """Build a provider from a spec: "mock", "anthropic", or "anthropic:<model>"."""
    kind, _, model = spec.partition(":")
    if kind == "mock" and not model:
        return MockProvider()
    if kind == "anthropic":
        return AnthropicProvider(model) if model else AnthropicProvider()
    raise ValueError(f"unknown provider '{spec}' (expected 'mock' or 'anthropic[:model]')")
