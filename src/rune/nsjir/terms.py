"""NSJIR term expressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Term:
    op: str
    args: tuple[Term, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [arg.to_dict() for arg in self.args],
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Term:
        return cls(
            op=payload["op"],
            args=tuple(cls.from_dict(arg) for arg in payload.get("args", [])),
            attrs=dict(payload.get("attrs", {})),
        )


def const(value: Any) -> Term:
    return Term("const", attrs={"value": value})


def var(name: str) -> Term:
    return Term("var", attrs={"name": name})


def call(op: str, *args: Term, **attrs: Any) -> Term:
    return Term(op=op, args=tuple(args), attrs=attrs)


@dataclass(frozen=True)
class Policy:
    op: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "attrs": self.attrs}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Policy:
        return cls(op=payload["op"], attrs=dict(payload.get("attrs", {})))


def fire_one(realization_id: str) -> Policy:
    return Policy("fire_one", {"realization_id": realization_id})


def fire_active(confidence: str = "argmax") -> Policy:
    return Policy("fire_active", {"confidence": confidence})


def fire_quorum(k: int, agreement_metric: str) -> Policy:
    return Policy("fire_quorum", {"k": k, "agreement_metric": agreement_metric})


def fire_union(realization_ids: list[str]) -> Policy:
    return Policy("fire_union", {"realization_ids": realization_ids})


def abstain_on_disagreement() -> Policy:
    return Policy("abstain_on_disagreement")