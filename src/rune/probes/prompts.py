"""3-shot REPL prompt generation utilities for Rune probes.

All Pythia probes use the same ``>>> a + b`` few-shot template.
This module centralises prompt generation with a fixed ``seed=0``
default so cross-probe results remain comparable.
"""

from __future__ import annotations

import torch


def few_shot_repl_prompts(
    n_pairs: int = 200,
    max_operand: int = 49,
    seed: int = 0,
) -> tuple[list[str], list[tuple[int, int]], list[int]]:
    """Generate the 3-shot REPL prompt suite used across all Pythia probes.

    Each prompt ends with ``>>> {a} + {b}\\n`` where ``a, b ∈ [0,
    max_operand]``.  With ``max_operand=49`` the answer ``a + b`` stays
    in ``[0, 98]``, all single-token under GPT-NeoX BPE per
    ``docs/pythia_tokenization_audit.md``.

    Parameters
    ----------
    n_pairs
        Number of ``(a, b)`` pairs to sample.  Default 200 matches the
        probe baseline from ``scripts/pythia_scale_sweep.py`` and
        ``scripts/pythia_mdl_probe.py``.
    max_operand
        Inclusive upper bound for operand sampling.
    seed
        Torch generator seed.  Leave at 0 for cross-probe comparability.

    Returns
    -------
    prompts
        List of ``n_pairs`` prompt strings.
    operand_pairs
        List of ``(a, b)`` integer tuples in the same order.
    true_answers
        List of ``a + b`` integer values in the same order.
    """
    rng = torch.Generator().manual_seed(seed)
    a_vals = torch.randint(0, max_operand + 1, (n_pairs,), generator=rng).tolist()
    b_vals = torch.randint(0, max_operand + 1, (n_pairs,), generator=rng).tolist()

    prompts: list[str] = []
    operand_pairs: list[tuple[int, int]] = []
    true_answers: list[int] = []

    for a, b in zip(a_vals, b_vals, strict=True):
        prompt = (
            ">>> 1 + 1\n2\n"
            ">>> 2 + 2\n4\n"
            ">>> 3 + 3\n6\n"
            f">>> {a} + {b}\n"
        )
        prompts.append(prompt)
        operand_pairs.append((int(a), int(b)))
        true_answers.append(int(a) + int(b))

    return prompts, operand_pairs, true_answers
