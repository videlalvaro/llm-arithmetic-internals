"""Parse JSON-serialized NSJIR objects."""

from __future__ import annotations

import json
from typing import TypeVar

T = TypeVar("T")


def loads(text: str, cls: type[T]) -> T:
    return cls.from_dict(json.loads(text))