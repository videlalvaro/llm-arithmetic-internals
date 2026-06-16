"""Redundant-carrier modular-addition model with planted Fourier mechanisms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from rune.models.modadd import enumerate_modadd_data

CarrierName = Literal["carrier_a", "carrier_b"]


@dataclass(frozen=True)
class RedundantModAddConfig:
    modulus: int
    carrier_scale: float = 10.0

    @property
    def vocab_size(self) -> int:
        return self.modulus + 1

    @property
    def equals_token(self) -> int:
        return self.modulus


class RedundantModAddModel(nn.Module):
    """Exact modadd model with two causally sufficient Fourier-logit carriers."""

    def __init__(self, config: RedundantModAddConfig) -> None:
        super().__init__()
        self.config = config
        classes = torch.arange(config.modulus, dtype=torch.float32)
        self.register_buffer("classes", classes, persistent=False)

    def carrier_logits(self, tokens: Tensor, carrier: CarrierName) -> Tensor:
        if carrier not in {"carrier_a", "carrier_b"}:
            raise ValueError(f"Unknown carrier: {carrier}")
        sums = (tokens[:, 0] + tokens[:, 1]).remainder(self.config.modulus).float().unsqueeze(-1)
        phase = 2 * torch.pi * (sums - self.classes.unsqueeze(0)) / self.config.modulus
        logits = self.config.carrier_scale * torch.cos(phase)
        if carrier == "carrier_b":
            return logits.clone()
        return logits

    def forward(
        self,
        tokens: Tensor,
        *,
        ablate_carriers: set[CarrierName] | frozenset[CarrierName] | None = None,
    ) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 3:
            raise ValueError("RedundantModAddModel expects tokens with shape (batch, 3)")
        ablated = ablate_carriers or frozenset()
        logits = torch.zeros(tokens.shape[0], self.config.modulus, device=tokens.device)
        if "carrier_a" not in ablated:
            logits = logits + self.carrier_logits(tokens, "carrier_a")
        if "carrier_b" not in ablated:
            logits = logits + self.carrier_logits(tokens, "carrier_b")
        return logits


def redundant_carrier_accuracy(
    model: RedundantModAddModel,
    tokens: Tensor,
    targets: Tensor,
    *,
    ablate_carriers: set[CarrierName] | frozenset[CarrierName] | None = None,
) -> float:
    model.eval()
    with torch.inference_mode():
        predictions = model(tokens, ablate_carriers=ablate_carriers).argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def save_redundant_modadd_checkpoint(
    path: str | Path,
    model: RedundantModAddModel,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.redundant_modadd_checkpoint",
        "config": asdict(model.config),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_redundant_modadd_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> RedundantModAddModel:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.redundant_modadd_checkpoint":
        raise ValueError("Not a Rune redundant modular-addition checkpoint")
    return RedundantModAddModel(RedundantModAddConfig(**checkpoint["config"]))


def enumerate_redundant_modadd_data(
    modulus: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    return enumerate_modadd_data(modulus, device=device)