"""Synthetic integer-addition model with planted helix coordinates in the embedding."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HelixAddConfig:
    input_range: int = 100
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    d_mlp: int = 256
    dropout: float = 0.0
    periods: tuple[int, ...] = (2, 5, 10, 100)

    @property
    def answer_vocab_size(self) -> int:
        return 2 * self.input_range - 1

    @property
    def basis_dim(self) -> int:
        return 1 + 2 * len(self.periods)


def helix_basis_table(config: HelixAddConfig) -> Tensor:
    """Return (input_range, d_model) with first basis_dim columns set to the planted helix."""
    values = torch.zeros(config.input_range, config.d_model)
    numbers = torch.arange(config.input_range, dtype=torch.float32)
    values[:, 0] = numbers / float(config.input_range)
    for index, period in enumerate(config.periods):
        angle = 2.0 * math.pi * numbers / float(period)
        values[:, 1 + 2 * index] = torch.cos(angle)
        values[:, 2 + 2 * index] = torch.sin(angle)
    return values


class HelixAddTransformer(nn.Module):
    """Vanilla transformer over a 2-token sequence (a, b) predicting a+b.

    The token embedding is initialized with planted helix coordinates in its first basis_dim
    columns. The remaining d_model - basis_dim columns are zero at init. The transformer
    encoder, position embedding, final norm, and unembed are all standard learnable modules.

    The model has no special architecture for arithmetic. It must learn to use the planted
    embedding through standard self-attention and MLPs.
    """

    def __init__(self, config: HelixAddConfig | None = None) -> None:
        super().__init__()
        self.config = config or HelixAddConfig()
        if self.config.d_model < self.config.basis_dim:
            raise ValueError("d_model must be at least basis_dim to hold the planted helix")

        self.token_embedding = nn.Embedding(self.config.input_range, self.config.d_model)
        self.position_embedding = nn.Parameter(torch.empty(2, self.config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.d_mlp,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.unembed = nn.Linear(self.config.d_model, self.config.answer_vocab_size)
        self.reset_parameters()

    @property
    def period_columns(self) -> dict[int, tuple[int, int]]:
        return {
            period: (1 + 2 * index, 2 + 2 * index)
            for index, period in enumerate(self.config.periods)
        }

    @property
    def all_period_columns(self) -> tuple[int, ...]:
        columns: list[int] = []
        for cos_col, sin_col in self.period_columns.values():
            columns.extend((cos_col, sin_col))
        return tuple(columns)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight.copy_(helix_basis_table(self.config))
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 2:
            raise ValueError("HelixAddTransformer expects tokens with shape (batch, 2)")
        if tokens.min().item() < 0 or tokens.max().item() >= self.config.input_range:
            raise ValueError(f"tokens must be in [0, {self.config.input_range - 1}]")
        h = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        h = self.encoder(h)
        return self.unembed(self.final_norm(h[:, -1]))


def enumerate_helix_add_data(
    input_range: int = 100,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    if input_range < 2:
        raise ValueError("input_range must be at least 2")
    rows = [(a, b) for a in range(input_range) for b in range(input_range)]
    tokens = torch.tensor(rows, dtype=torch.long, device=device)
    targets = tokens[:, 0] + tokens[:, 1]
    return tokens, targets


@torch.inference_mode()
def helix_add_accuracy(
    model: HelixAddTransformer,
    tokens: Tensor,
    targets: Tensor,
) -> float:
    model.eval()
    predictions = model(tokens).argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def save_helix_add_checkpoint(
    path: str | Path,
    model: HelixAddTransformer,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.helix_add_checkpoint",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_helix_add_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> HelixAddTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.helix_add_checkpoint":
        raise ValueError("Not a Rune helix-add checkpoint")
    config = checkpoint["config"]
    if isinstance(config.get("periods"), list):
        config["periods"] = tuple(config["periods"])
    model = HelixAddTransformer(HelixAddConfig(**config))
    model.load_state_dict(checkpoint["state_dict"])
    return model
