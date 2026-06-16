"""Staged NSJIR mechanism-family types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from rune.nsjir.contracts import ContractRealization
from rune.nsjir.terms import Term
from rune.nsjir.types import TypeExpr


class MechanismStage(Enum):
    TRANSPORT = "transport"
    CONSTRUCT = "construct"
    READOUT = "readout"


@dataclass(frozen=True)
class MechanismStageContract:
    stage: MechanismStage
    layer_range: tuple[str, str]
    input_type: object
    output_type: object
    semantics: object

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "layer_range": list(self.layer_range),
            "input_type": _encode_obj(self.input_type),
            "output_type": _encode_obj(self.output_type),
            "semantics": _encode_obj(self.semantics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MechanismStageContract:
        return cls(
            stage=MechanismStage(payload["stage"]),
            layer_range=tuple(payload["layer_range"]),
            input_type=_decode_obj(payload["input_type"]),
            output_type=_decode_obj(payload["output_type"]),
            semantics=_decode_obj(payload["semantics"]),
        )


@dataclass(frozen=True)
class StagedMechanismFamily:
    """A MechanismFamily decomposed into transport / construct / readout stages."""

    id: str
    semantics: object
    stages: tuple[MechanismStageContract, ...]
    stage_interfaces: tuple[object, ...]
    realizations: tuple[object, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "semantics": _encode_obj(self.semantics),
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_interfaces": [_encode_obj(interface) for interface in self.stage_interfaces],
            "realizations": [_encode_obj(realization) for realization in self.realizations],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StagedMechanismFamily:
        return cls(
            id=payload["id"],
            semantics=_decode_obj(payload["semantics"]),
            stages=tuple(
                MechanismStageContract.from_dict(stage) for stage in payload.get("stages", [])
            ),
            stage_interfaces=tuple(
                _decode_obj(interface) for interface in payload.get("stage_interfaces", [])
            ),
            realizations=tuple(
                _decode_obj(realization) for realization in payload.get("realizations", [])
            ),
        )


def _encode_obj(value: object) -> object:
    if isinstance(value, TypeExpr):
        return {"kind": "TypeExpr", "payload": value.to_dict()}
    if isinstance(value, Term):
        return {"kind": "Term", "payload": value.to_dict()}
    if isinstance(value, ContractRealization):
        return {"kind": "ContractRealization", "payload": value.to_dict()}
    if hasattr(value, "to_dict"):
        return {
            "kind": value.__class__.__name__,
            "payload": value.to_dict(),  # type: ignore[attr-defined]
        }
    return value


def _decode_obj(value: object) -> object:
    if not isinstance(value, dict) or "kind" not in value:
        return value
    kind = value["kind"]
    payload = value["payload"]
    if kind == "TypeExpr":
        return TypeExpr.from_dict(payload)
    if kind == "Term":
        return Term.from_dict(payload)
    if kind == "ContractRealization":
        return ContractRealization.from_dict(payload)
    return payload
