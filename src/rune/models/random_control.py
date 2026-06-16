"""Synthetic random-control model helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rune.models.regular import (
    RegularConfig,
    RegularLanguageTransformer,
    enumerate_even_parity_data,
)


@dataclass(frozen=True)
class RandomControlConfig:
    input_range: int = 100
    output_classes: int = 199
    seed: int = 0


class RandomControlTransformer(nn.Module):
    """Stateless random-label negative control over integer operand pairs."""

    def __init__(self, config: RandomControlConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Identity()
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        logits = torch.randn(
            config.input_range * config.input_range,
            config.output_classes,
            generator=generator,
        )
        self.register_buffer("logit_table", logits)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 2:
            raise ValueError("RandomControlTransformer expects tokens with shape (batch, 2)")
        pair_index = tokens[:, 0] * self.config.input_range + tokens[:, 1]
        hidden = self.encoder(self.logit_table[pair_index])
        return hidden


def enumerate_random_control_data(
    max_length: int,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary strings with deterministic random binary labels."""

    tokens, _ = enumerate_even_parity_data(max_length, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randint(0, 2, (tokens.shape[0],), generator=generator, dtype=torch.long)
    return tokens, labels.to(device=tokens.device)


def save_random_control_checkpoint(
    path: str | Path,
    model: RegularLanguageTransformer,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.random_control_checkpoint",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_random_control_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> RegularLanguageTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.random_control_checkpoint":
        raise ValueError("Not a Rune random-control checkpoint")
    model = RegularLanguageTransformer(RegularConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model
