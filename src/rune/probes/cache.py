"""ClockExtraction disk cache for cross-probe reuse.

A wide-period extraction on Pythia-2.8B takes ~5 minutes.  If multiple
probes use the same ``(model_id, periods)`` pair, this module lets the
second run skip extraction by loading from ``.cache/extractions/``.

Cache files are ``torch.save`` / ``torch.load`` bundles of the
``ClockExtraction`` dataclass.  They are gitignored by default (add
``.cache/`` to ``.gitignore``).

Usage
-----
::

    from rune.probes.cache import load_extraction, save_extraction

    ext = load_extraction(model_id, periods)
    if ext is None:
        ext = extract_clock_arithmetic(...)
        save_extraction(ext, model_id, periods)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def cache_path(
    model_id: str,
    periods: tuple[int, ...],
    *,
    suffix: str = "extraction",
) -> Path:
    """Compute the cache filename for a ``(model_id, periods)`` pair.

    The path is relative to the current working directory so it works
    whether the scripts are run from the repo root or elsewhere.

    Example
    -------
    ``cache_path("EleutherAI/pythia-2.8b", (2, 5, 10))``
    → ``.cache/extractions/EleutherAI_pythia-2.8b__2_5_10__extraction.pt``
    """
    safe_model = model_id.replace("/", "_")
    safe_periods = "_".join(str(p) for p in sorted(periods))
    return (
        Path(".cache") / "extractions" / f"{safe_model}__{safe_periods}__{suffix}.pt"
    )


def save_extraction(
    extraction: Any,
    model_id: str,
    periods: tuple[int, ...],
) -> Path:
    """Save a ``ClockExtraction`` to disk for cross-probe reuse.

    Parameters
    ----------
    extraction
        The ``ClockExtraction`` object returned by
        ``extract_clock_arithmetic``.
    model_id
        HuggingFace model ID string (e.g. ``"EleutherAI/pythia-2.8b"``).
    periods
        Period tuple used for this extraction (e.g. ``(2, 5, 10)``).

    Returns
    -------
    Path
        The path the file was written to.
    """
    path = cache_path(model_id, periods)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extraction, path)
    return path


def load_extraction(
    model_id: str,
    periods: tuple[int, ...],
) -> Any | None:
    """Load a previously-saved ``ClockExtraction`` or return ``None``.

    Returns ``None`` (not an exception) if the cache file does not exist,
    so callers can do::

        ext = load_extraction(model_id, periods) or extract_clock_arithmetic(...)
    """
    path = cache_path(model_id, periods)
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)
