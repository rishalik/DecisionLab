"""The interface every decision provider implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, system: str, message: str) -> dict[str, Any]:
        """Return the schema's fields, plus optional `confidence` (0..1) and `reasoning`.

        The result is not trusted: it may be incomplete or garbage. The Decider validates it.
        """
        ...
