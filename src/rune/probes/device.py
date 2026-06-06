"""Device selection utilities for Rune probes.

Probes that run forward passes benefit from accelerator throughput
(MPS on Apple Silicon, CUDA elsewhere).  This module provides a
single ``auto_device`` helper so device logic is not duplicated across
probe scripts.
"""

from __future__ import annotations

import os

import torch


def auto_device() -> torch.device:
    """Return MPS if available, else CUDA, else CPU.

    Used by probes that benefit from accelerator throughput (forward
    passes, SAE training).

    Override with env var ``RUNE_PROBE_DEVICE=cpu|mps|cuda`` for
    debugging or to force a specific backend.
    """
    forced = os.environ.get("RUNE_PROBE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
