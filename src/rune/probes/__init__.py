"""Rune probe utilities — device selection, model adapters, prompt generation,
and extraction caching.

Public API
----------
::

    from rune.probes.device import auto_device
    from rune.probes.model_adapter import detect_adapter, ADAPTER_PYTHIA
    from rune.probes.prompts import few_shot_repl_prompts
    from rune.probes.cache import load_extraction, save_extraction
"""

from rune.probes.cache import cache_path, load_extraction, save_extraction
from rune.probes.device import auto_device
from rune.probes.model_adapter import (
    ADAPTER_HELIX_ADD,
    ADAPTER_LLAMA,
    ADAPTER_PYTHIA,
    ModelAdapter,
    detect_adapter,
)
from rune.probes.prompts import few_shot_repl_prompts

__all__ = [
    "auto_device",
    "ModelAdapter",
    "ADAPTER_HELIX_ADD",
    "ADAPTER_PYTHIA",
    "ADAPTER_LLAMA",
    "detect_adapter",
    "few_shot_repl_prompts",
    "cache_path",
    "save_extraction",
    "load_extraction",
]
