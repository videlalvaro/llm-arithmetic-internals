"""HuggingFace-backed Pythia (GPTNeoX) model loading utilities."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import device as torch_device
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from rune.models.llama import default_hf_cache_dir  # shared cache-dir logic

DEFAULT_PYTHIA_MODEL_ID = "EleutherAI/pythia-160m"


def resolve_pythia_source(source: str | os.PathLike[str] | None = None) -> str | Path:
    """Resolve a Pythia model source from explicit input or environment."""

    if source is not None:
        source_path = Path(source).expanduser()
        return source_path if source_path.exists() else str(source)

    if configured_path := os.getenv("RUNE_PYTHIA_MODEL_PATH"):
        return Path(configured_path).expanduser()
    if configured_id := os.getenv("RUNE_PYTHIA_MODEL_ID"):
        return configured_id
    return DEFAULT_PYTHIA_MODEL_ID


def resolve_pythia_device(device: str | torch_device = "cpu") -> torch_device:
    """Return a usable runtime device for Pythia inference.

    Environment requests can be optimistic; on some macOS setups `mps` reports
    available but still rejects tensor placement. This helper probes the backend
    and falls back to CPU when the requested device is not actually usable.
    """

    requested = torch.device(device)
    if requested.type == "cpu":
        return requested
    if requested.type == "cuda":
        return requested if torch.cuda.is_available() else torch.device("cpu")
    if requested.type == "mps":
        if not torch.backends.mps.is_available():
            return torch.device("cpu")
        try:
            torch.zeros(1).to(requested)
        except RuntimeError:
            return torch.device("cpu")
        return requested
    return requested


def load_pythia_causal_lm(
    source: str | os.PathLike[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    device: str | torch_device = "cpu",
    local_files_only: bool | None = None,
) -> PreTrainedModel:
    """Load a Pythia causal language model through HuggingFace transformers."""

    resolved_source = resolve_pythia_source(source)
    resolved_device = resolve_pythia_device(device)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else default_hf_cache_dir()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    local_only = isinstance(resolved_source, Path) if local_files_only is None else local_files_only
    model = AutoModelForCausalLM.from_pretrained(
        resolved_source,
        cache_dir=str(resolved_cache_dir),
        local_files_only=local_only,
    )
    model.eval()
    if resolved_device.type != "cpu":
        model.to(resolved_device)
    return model


def load_pythia_tokenizer(
    source: str | os.PathLike[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool | None = None,
) -> PreTrainedTokenizer:
    """Load the tokenizer for the configured Pythia source."""

    resolved_source = resolve_pythia_source(source)
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else default_hf_cache_dir()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    local_only = isinstance(resolved_source, Path) if local_files_only is None else local_files_only
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source,
        cache_dir=str(resolved_cache_dir),
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
