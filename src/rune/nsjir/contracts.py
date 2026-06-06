"""NSJIR contract metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rune.nsjir.terms import Term
from rune.nsjir.types import TypeExpr


@dataclass(frozen=True)
class EdgeMask:
    model_graph_id: str
    edges: frozenset[str]
    ablation: str = "zero"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_graph_id": self.model_graph_id,
            "edges": sorted(self.edges),
            "ablation": self.ablation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeMask:
        return cls(
            model_graph_id=payload["model_graph_id"],
            edges=frozenset(payload["edges"]),
            ablation=payload.get("ablation", "zero"),
        )


@dataclass(frozen=True)
class ContractRealization:
    id: str
    layer_in: str
    layer_out: str
    read_projection: str
    write_projection: str
    support: EdgeMask
    input_type: TypeExpr
    output_type: TypeExpr
    read: Term
    write: Term
    semantics: Term
    error_bound: float
    abstain: Term
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer_in": self.layer_in,
            "layer_out": self.layer_out,
            "read_projection": self.read_projection,
            "write_projection": self.write_projection,
            "support": self.support.to_dict(),
            "input_type": self.input_type.to_dict(),
            "output_type": self.output_type.to_dict(),
            "read": self.read.to_dict(),
            "write": self.write.to_dict(),
            "semantics": self.semantics.to_dict(),
            "error_bound": self.error_bound,
            "abstain": self.abstain.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContractRealization:
        return cls(
            id=payload["id"],
            layer_in=payload["layer_in"],
            layer_out=payload["layer_out"],
            read_projection=payload["read_projection"],
            write_projection=payload["write_projection"],
            support=EdgeMask.from_dict(payload["support"]),
            input_type=TypeExpr.from_dict(payload["input_type"]),
            output_type=TypeExpr.from_dict(payload["output_type"]),
            read=Term.from_dict(payload["read"]),
            write=Term.from_dict(payload["write"]),
            semantics=Term.from_dict(payload["semantics"]),
            error_bound=payload["error_bound"],
            abstain=Term.from_dict(payload["abstain"]),
            metadata=dict(payload.get("metadata", {})),
        )