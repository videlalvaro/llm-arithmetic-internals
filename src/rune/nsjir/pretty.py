"""Pretty serialization for NSJIR objects."""

from __future__ import annotations

import json
from typing import Protocol


class Serializable(Protocol):
    def to_dict(self) -> dict: ...


def dumps(value: Serializable) -> str:
    return json.dumps(value.to_dict(), indent=2, sort_keys=True)