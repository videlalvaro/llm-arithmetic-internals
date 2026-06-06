"""Synthetic modular-addition transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModAddConfig:
    modulus: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_mlp: int = 256
    dropout: float = 0.0

    @property
    def vocab_size(self) -> int:
        return self.modulus + 1

    @property
    def equals_token(self) -> int:
        return self.modulus


class ModAddTransformer(nn.Module):
    """Small encoder-only transformer trained to predict `(a + b) mod m`."""

    def __init__(self, config: ModAddConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.empty(3, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_mlp,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.modulus)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 3:
            raise ValueError("ModAddTransformer expects tokens with shape (batch, 3)")
        hidden = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        hidden = self.encoder(hidden)
        return self.unembed(self.final_norm(hidden[:, -1]))


def enumerate_modadd_data(
    modulus: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return all `(a, b, =) -> (a + b) mod m` examples."""

    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    rows = [(a, b, modulus) for a in range(modulus) for b in range(modulus)]
    tokens = torch.tensor(rows, dtype=torch.long, device=device)
    targets = (tokens[:, 0] + tokens[:, 1]) % modulus
    return tokens, targets


@torch.inference_mode()
def modadd_accuracy(model: nn.Module, tokens: Tensor, targets: Tensor) -> float:
    model.eval()
    predictions = model(tokens).argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def save_modadd_checkpoint(
    path: str | Path,
    model: ModAddTransformer,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.modadd_checkpoint",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_modadd_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ModAddTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.modadd_checkpoint":
        raise ValueError("Not a Rune modular-addition checkpoint")
    model = ModAddTransformer(ModAddConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model