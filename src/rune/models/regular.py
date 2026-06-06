"""Synthetic regular-language transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

PAD_TOKEN = 0
ZERO_TOKEN = 1
ONE_TOKEN = 2
CLS_TOKEN = 3


@dataclass(frozen=True)
class RegularConfig:
    max_length: int = 12
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_mlp: int = 256
    dropout: float = 0.0

    @property
    def vocab_size(self) -> int:
        return 4

    @property
    def sequence_length(self) -> int:
        return self.max_length + 1


class RegularLanguageTransformer(nn.Module):
    """Small encoder-only transformer for binary regular-language classification."""

    def __init__(self, config: RegularConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.empty(config.sequence_length, config.d_model))
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
        self.classifier = nn.Linear(config.d_model, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.config.sequence_length:
            expected = self.config.sequence_length
            raise ValueError(f"RegularLanguageTransformer expects shape (batch, {expected})")
        hidden = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        hidden = self.encoder(hidden)
        return self.classifier(self.final_norm(hidden[:, -1]))


def enumerate_even_parity_data(
    max_length: int,
    *,
    device: torch.device | str | None = None,
    include_empty: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return padded binary strings labeled by even parity of `1` tokens."""

    if max_length < 1:
        raise ValueError("max_length must be at least 1")

    rows: list[list[int]] = []
    labels: list[int] = []
    min_length = 0 if include_empty else 1
    for length in range(min_length, max_length + 1):
        for value in range(2**length):
            bits = [(value >> shift) & 1 for shift in reversed(range(length))]
            token_bits = [ONE_TOKEN if bit else ZERO_TOKEN for bit in bits]
            padded = token_bits + [PAD_TOKEN] * (max_length - length) + [CLS_TOKEN]
            rows.append(padded)
            labels.append(int(sum(bits) % 2 == 0))

    return torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )


@torch.inference_mode()
def regular_accuracy(model: nn.Module, tokens: Tensor, targets: Tensor) -> float:
    model.eval()
    predictions = model(tokens).argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def save_regular_checkpoint(
    path: str | Path,
    model: RegularLanguageTransformer,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.regular_checkpoint",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_regular_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> RegularLanguageTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.regular_checkpoint":
        raise ValueError("Not a Rune regular-language checkpoint")
    model = RegularLanguageTransformer(RegularConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model