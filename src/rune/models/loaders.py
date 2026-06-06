"""Forward-pass checkpoint loaders for Rune model families."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from rune.models.helix_add import load_helix_add_checkpoint
from rune.models.induction import load_induction_checkpoint
from rune.models.modadd import load_modadd_checkpoint
from rune.models.modadd_redundant import load_redundant_modadd_checkpoint
from rune.models.random_control import load_random_control_checkpoint
from rune.models.regular import load_regular_checkpoint

CHECKPOINT_LOADERS = {
    "rune.helix_add_checkpoint": load_helix_add_checkpoint,
    "rune.modadd_checkpoint": load_modadd_checkpoint,
    "rune.redundant_modadd_checkpoint": load_redundant_modadd_checkpoint,
    "rune.regular_checkpoint": load_regular_checkpoint,
    "rune.random_control_checkpoint": load_random_control_checkpoint,
    "rune.induction_checkpoint": load_induction_checkpoint,
}


def checkpoint_format(path: str | Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        return checkpoint["format"]
    except KeyError as error:
        raise ValueError("Checkpoint does not declare a Rune format") from error


def load_model_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    """Load any supported Rune checkpoint as a forward-pass module."""

    format_name = checkpoint_format(path)
    try:
        loader = CHECKPOINT_LOADERS[format_name]
    except KeyError as error:
        raise ValueError(f"Unsupported Rune checkpoint format: {format_name}") from error
    return loader(path, map_location=map_location)
