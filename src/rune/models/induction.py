"""Synthetic induction-head transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class InductionConfig:
    vocab_size: int = 32
    sequence_length: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


class AttentionOnlyBlock(nn.Module):
    """Transformer block with attention and residual normalization but no MLP."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        normalized = self.norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        return hidden + attended


class InductionTransformer(nn.Module):
    """Two-layer attention-only transformer for copy-after-previous-token tasks."""

    def __init__(self, config: InductionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Parameter(torch.empty(config.sequence_length, config.d_model))
        self.blocks = nn.ModuleList(
            AttentionOnlyBlock(config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.config.sequence_length:
            expected = self.config.sequence_length
            raise ValueError(f"InductionTransformer expects shape (batch, {expected})")
        hidden = self.token_embedding(tokens) + self.position_embedding.unsqueeze(0)
        attention_mask = torch.triu(
            torch.full(
                (self.config.sequence_length, self.config.sequence_length),
                float("-inf"),
                device=tokens.device,
            ),
            diagonal=1,
        )
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return self.unembed(self.final_norm(hidden[:, -1]))


def sample_induction_data(
    num_examples: int,
    *,
    vocab_size: int = 32,
    sequence_length: int = 16,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample contexts where the final token should copy the value after its prior match."""

    if vocab_size < 4:
        raise ValueError("vocab_size must be at least 4")
    if sequence_length < 4:
        raise ValueError("sequence_length must be at least 4")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, vocab_size, (num_examples, sequence_length), generator=generator)
    targets = torch.empty(num_examples, dtype=torch.long)

    for row in range(num_examples):
        key = int(torch.randint(0, vocab_size, (), generator=generator).item())
        value = int(torch.randint(0, vocab_size - 1, (), generator=generator).item())
        if value >= key:
            value += 1
        first_position = int(torch.randint(0, sequence_length - 2, (), generator=generator).item())
        filler = torch.randint(0, vocab_size - 1, (sequence_length,), generator=generator)
        filler = filler + (filler >= key).long()
        tokens[row] = filler
        tokens[row, first_position] = key
        tokens[row, first_position + 1] = value
        tokens[row, -1] = key
        targets[row] = value

    return tokens.to(device=device), targets.to(device=device)


@torch.inference_mode()
def induction_accuracy(model: nn.Module, tokens: Tensor, targets: Tensor) -> float:
    model.eval()
    predictions = model(tokens).argmax(dim=-1)
    return float((predictions == targets).float().mean().item())


def save_induction_checkpoint(
    path: str | Path,
    model: InductionTransformer,
    *,
    accuracy: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "format": "rune.induction_checkpoint",
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "accuracy": accuracy,
        "metadata": metadata or {},
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_induction_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> InductionTransformer:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "rune.induction_checkpoint":
        raise ValueError("Not a Rune induction checkpoint")
    model = InductionTransformer(InductionConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model