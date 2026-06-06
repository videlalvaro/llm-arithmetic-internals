"""NSJIR mechanism family objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from rune.nsjir.contracts import ContractRealization
from rune.nsjir.eval import evaluate
from rune.nsjir.terms import Term

AggregationPolicy = Literal["one_of", "quorum", "ensemble"]


@dataclass(frozen=True)
class OverlapCert:
    pairwise_iou: tuple[tuple[float, ...], ...]
    mutual_iou: float
    node_iou: tuple[tuple[float, ...], ...]
    edge_iou: tuple[tuple[float, ...], ...]
    chance_iou_baseline: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairwise_iou": self.pairwise_iou,
            "mutual_iou": self.mutual_iou,
            "node_iou": self.node_iou,
            "edge_iou": self.edge_iou,
            "chance_iou_baseline": self.chance_iou_baseline,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OverlapCert:
        return cls(
            pairwise_iou=_matrix(payload["pairwise_iou"]),
            mutual_iou=payload["mutual_iou"],
            node_iou=_matrix(payload["node_iou"]),
            edge_iou=_matrix(payload["edge_iou"]),
            chance_iou_baseline=payload["chance_iou_baseline"],
        )


@dataclass(frozen=True)
class MechanismFamily:
    id: str
    semantics: Term
    realizations: tuple[ContractRealization, ...]
    overlap: OverlapCert
    aggregation: AggregationPolicy = "quorum"
    invariants: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "semantics": self.semantics.to_dict(),
            "realizations": [realization.to_dict() for realization in self.realizations],
            "overlap": self.overlap.to_dict(),
            "aggregation": self.aggregation,
            "invariants": list(self.invariants),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MechanismFamily:
        return cls(
            id=payload["id"],
            semantics=Term.from_dict(payload["semantics"]),
            realizations=tuple(
                ContractRealization.from_dict(realization)
                for realization in payload["realizations"]
            ),
            overlap=OverlapCert.from_dict(payload["overlap"]),
            aggregation=payload.get("aggregation", "quorum"),
            invariants=tuple(payload.get("invariants", [])),
            metadata=dict(payload.get("metadata", {})),
        )

    def evaluate_realizations(self, env: dict[str, Any]) -> tuple[list[Any], NDArray[np.float64]]:
        values = [evaluate(realization.semantics, env) for realization in self.realizations]
        return values, np.asarray(self.overlap.pairwise_iou, dtype=np.float64)


def _matrix(value: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(cell) for cell in row) for row in value)