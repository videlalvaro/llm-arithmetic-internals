"""NSJIR type expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypeExpr:
    name: str
    args: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": [_encode_arg(arg) for arg in self.args]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TypeExpr:
        return cls(
            name=payload["name"],
            args=tuple(_decode_arg(arg) for arg in payload.get("args", [])),
        )


def _encode_arg(value: Any) -> Any:
    if isinstance(value, TypeExpr):
        return {"type": value.to_dict()}
    return value


def _decode_arg(value: Any) -> Any:
    if isinstance(value, dict) and "type" in value:
        return TypeExpr.from_dict(value["type"])
    return value


Bit = TypeExpr("Bit")
Tok = TypeExpr("Tok")


def IntMod(modulus: int) -> TypeExpr:
    return TypeExpr("Int", ("mod", modulus))


def IntRange(low: int, high: int) -> TypeExpr:
    return TypeExpr("Int", ("range", low, high))


def Real(dim: int) -> TypeExpr:
    return TypeExpr("Real", (dim,))


def Seq(item: TypeExpr) -> TypeExpr:
    return TypeExpr("Seq", (item,))


def SetType(item: TypeExpr) -> TypeExpr:
    return TypeExpr("Set", (item,))


def MapType(key: TypeExpr, value: TypeExpr) -> TypeExpr:
    return TypeExpr("Map", (key, value))


def Graph(vertex: TypeExpr, edge: TypeExpr) -> TypeExpr:
    return TypeExpr("Graph", (vertex, edge))


def Tree(item: TypeExpr) -> TypeExpr:
    return TypeExpr("Tree", (item,))


def State(count: int) -> TypeExpr:
    return TypeExpr("State", (count,))


def Dist(item: TypeExpr) -> TypeExpr:
    return TypeExpr("Dist", (item,))