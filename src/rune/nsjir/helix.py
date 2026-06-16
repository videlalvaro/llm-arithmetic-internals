"""Helix arithmetic NSJIR types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HelixBasis:
    """Symbolic helix basis over a bounded integer range."""

    periods: tuple[int, ...]
    affine: bool
    input_range: tuple[int, int]

    def basis_dim(self) -> int:
        return 2 * len(self.periods) + int(self.affine)

    def encode(self, n: int) -> tuple[float, ...]:
        lo, hi = self.input_range
        if n < lo or n > hi:
            raise ValueError(f"{n} is outside HelixBasis range [{lo}, {hi}]")
        values: list[float] = []
        if self.affine:
            values.append(float(n))
        for period in self.periods:
            angle = 2.0 * math.pi * float(n) / float(period)
            values.extend((math.cos(angle), math.sin(angle)))
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "periods": list(self.periods),
            "affine": self.affine,
            "input_range": list(self.input_range),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HelixBasis:
        return cls(
            periods=tuple(payload["periods"]),
            affine=payload["affine"],
            input_range=tuple(payload["input_range"]),
        )


@dataclass(frozen=True)
class HelixEmbedding:
    """Where a helix basis is linearly embedded in a model component."""

    basis: HelixBasis
    C: tuple[tuple[float, ...], ...]
    C_pinv: tuple[tuple[float, ...], ...]
    layer: str
    token_role: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis.to_dict(),
            "C": [list(row) for row in self.C],
            "C_pinv": [list(row) for row in self.C_pinv],
            "layer": self.layer,
            "token_role": self.token_role,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> HelixEmbedding:
        return cls(
            basis=HelixBasis.from_dict(payload["basis"]),
            C=tuple(tuple(row) for row in payload["C"]),
            C_pinv=tuple(tuple(row) for row in payload["C_pinv"]),
            layer=payload["layer"],
            token_role=payload["token_role"],
        )


@dataclass(frozen=True)
class ClockAdd:
    """Semantic addition law over a HelixBasis."""

    operand_range: tuple[int, int]
    result_range: tuple[int, int]
    basis: HelixBasis

    def to_dict(self) -> dict[str, Any]:
        return {
            "operand_range": list(self.operand_range),
            "result_range": list(self.result_range),
            "basis": self.basis.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClockAdd:
        return cls(
            operand_range=tuple(payload["operand_range"]),
            result_range=tuple(payload["result_range"]),
            basis=HelixBasis.from_dict(payload["basis"]),
        )
