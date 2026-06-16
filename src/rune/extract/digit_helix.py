"""Readout-side digit-positional decoder.

Companion to :mod:`rune.extract.token_helix`.  Where ``TokenHelixDecoder``
fits an abelian-character basis ``(cos(2πc/T), sin(2πc/T))`` to the rows of
``lm_head.weight`` for integer tokens, this module fits a **one-hot
digit-positional basis** to the same rows.

The motivation comes from ``docs/pythia_2.8b_sae_discovery.md`` (75% of the
top-causal SAE features at Pythia-2.8B's answer position are
*digit-positional*: "ones digit is 7", "tens digit is 4", etc.) and the
Llama modular-battery finding that the helix basis explains only ~10%
of integer-token lm_head row variance (`R²_global = 0.0983`).  The other
~90% is presumed to live in digit-positional features.

Output: a :class:`DigitDecoder` that gives, for each candidate integer
``c``, a 30-dimensional one-hot indicator over ``{ones, tens, hundreds}
× {0..9}`` and a regression matrix ``W_digit`` mapping that indicator to
``lm_head.weight`` rows.

Compose with :class:`TokenHelixDecoder` via concatenation::

    B_combined = [B_helix | B_digit]   # shape (V_int, 2K+1 + 30)
    W_combined = pinv(B_combined) @ lm_head.weight[token_ids, :]
    target = pinv(W_combined) @ B_combined[n, :]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

_PINV_RTOL = 1e-3


def _digit_basis_matrix(integers: Sequence[int], n_positions: int = 3) -> Tensor:
    """One-hot per-digit-position basis matrix.

    For each integer ``c`` and digit position ``p in [0, n_positions)``,
    produce an indicator column at ``p * 10 + digit_p(c)``.

    Returns shape ``(len(integers), 10 * n_positions)``.

    Example
    -------
    For ``integers=[47]`` and ``n_positions=3``, row is::

        [0]*7 + [1] + [0]*2     # ones=7
        + [0]*4 + [1] + [0]*5   # tens=4
        + [1] + [0]*9           # hundreds=0
    """
    V_int = len(integers)
    B = torch.zeros(V_int, 10 * n_positions, dtype=torch.float32)
    for i, c in enumerate(integers):
        for pos in range(n_positions):
            d = (c // (10 ** pos)) % 10
            B[i, pos * 10 + d] = 1.0
    return B


@dataclass(frozen=True)
class DigitDecoder:
    """Per-digit-position one-hot decoder fitted to ``lm_head.weight``.

    Attributes
    ----------
    integer_token_ids
        BPE token IDs in order of ``[lo, hi]``.
    answer_value_range
        Inclusive ``(lo, hi)`` integer range that was fit.
    n_positions
        Number of digit positions used (default 3 → ones, tens, hundreds).
    B_digit
        ``(V_int, 10 * n_positions)`` one-hot indicator matrix.
    W_digit
        ``(10 * n_positions, d_model)`` regression solution.
    R2_global
        Fraction of variance explained on the holdout fold.
    """

    integer_token_ids: Tensor
    answer_value_range: tuple[int, int]
    n_positions: int
    B_digit: Tensor
    W_digit: Tensor
    R2_global: float


def extract_digit_decoder(
    lm_head: nn.Module,
    integer_token_ids: Sequence[int],
    answer_value_range: tuple[int, int],
    *,
    n_positions: int = 3,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> DigitDecoder:
    """Fit a one-hot digit-positional basis to lm_head.weight rows.

    Parameters
    ----------
    lm_head
        The model's unembedding ``nn.Linear`` (e.g. ``model.lm_head``).
    integer_token_ids
        BPE token IDs for integers ``[lo, hi]`` in order.
    answer_value_range
        ``(lo, hi)`` inclusive integer range covered by ``integer_token_ids``.
    n_positions
        Number of digit positions to encode (default 3 = ones/tens/hundreds).
    holdout_frac
        Fraction of integers held out for R² evaluation.
    seed
        RNG seed for the holdout split.
    """
    lo, hi = answer_value_range
    integers = list(range(lo, hi + 1))
    if len(integers) != len(integer_token_ids):
        raise ValueError(
            f"extract_digit_decoder: integer_token_ids length "
            f"{len(integer_token_ids)} != range size {len(integers)}"
        )

    # lm_head.weight is (vocab, d_model).  Select the integer rows.
    W_lm: Tensor = lm_head.weight.detach().float().cpu()
    tok_ids_t = torch.tensor(list(integer_token_ids), dtype=torch.long)
    W_int = W_lm[tok_ids_t]  # (V_int, d_model)

    # One-hot digit basis.
    B = _digit_basis_matrix(integers, n_positions=n_positions)

    # Train/holdout split.
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(integers), generator=rng)
    n_holdout = max(1, int(len(integers) * holdout_frac))
    holdout_idx = perm[:n_holdout]
    fit_idx = perm[n_holdout:]

    B_fit = B[fit_idx]
    W_lm_fit = W_int[fit_idx]
    B_hold = B[holdout_idx]
    W_lm_hold = W_int[holdout_idx]

    # Regularized pinv regression: W_digit = pinv(B_fit) @ W_lm_fit
    W_digit = torch.linalg.pinv(B_fit, rtol=_PINV_RTOL) @ W_lm_fit

    # Holdout R²
    W_lm_pred = B_hold @ W_digit
    ss_res = ((W_lm_hold - W_lm_pred) ** 2).sum().item()
    mean = W_lm_hold.mean(dim=0, keepdim=True)
    ss_tot = ((W_lm_hold - mean) ** 2).sum().item()
    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return DigitDecoder(
        integer_token_ids=tok_ids_t,
        answer_value_range=(lo, hi),
        n_positions=n_positions,
        B_digit=B,
        W_digit=W_digit,
        R2_global=float(R2),
    )


def build_combined_basis(
    helix_B: Tensor,
    digit_B: Tensor,
    lm_head: nn.Module,
    integer_token_ids: Sequence[int],
) -> tuple[Tensor, Tensor, float]:
    """Concatenate helix + digit bases; refit jointly against lm_head rows.

    Returns
    -------
    B_combined : (V_int, K_h + K_d) — concatenated basis
    W_combined : (K_h + K_d, d_model) — joint regression solution
    R2_global  : holdout R² of B_combined @ W_combined vs lm_head.weight[token_ids]
    """
    W_lm: Tensor = lm_head.weight.detach().float().cpu()
    tok_ids_t = torch.tensor(list(integer_token_ids), dtype=torch.long)
    W_int = W_lm[tok_ids_t]
    if helix_B.shape[0] != digit_B.shape[0]:
        raise ValueError(
            f"build_combined_basis: shape mismatch helix_B={helix_B.shape} "
            f"digit_B={digit_B.shape}"
        )
    B_combined = torch.cat([helix_B, digit_B], dim=1)

    # Use 80/20 holdout for the joint R² estimate.
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(B_combined.shape[0], generator=rng)
    n_holdout = max(1, int(B_combined.shape[0] * 0.2))
    hold = perm[:n_holdout]
    fit = perm[n_holdout:]

    W_combined = torch.linalg.pinv(B_combined[fit], rtol=_PINV_RTOL) @ W_int[fit]
    pred = B_combined[hold] @ W_combined
    ss_res = ((W_int[hold] - pred) ** 2).sum().item()
    mean = W_int[hold].mean(dim=0, keepdim=True)
    ss_tot = ((W_int[hold] - mean) ** 2).sum().item()
    R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return B_combined, W_combined, float(R2)


def make_combined_write_fn(
    B_combined: Tensor,
    W_combined: Tensor,
    ans_lo: int,
    n_candidates: int,
) -> Any:
    """Build a projection-style combined write_fn for use in JIT eval.

    Strategy: ``h_target[n] = pinv(W_combined) @ B_combined[n, :]``
    (full-replacement, mirrors ``readout-pinv`` from two_sided.py).

    Parameters
    ----------
    B_combined : (V_int, K_h + K_d) — combined basis
    W_combined : (K_h + K_d, d_model) — joint regression
    ans_lo : int — minimum integer in the answer range
    n_candidates : int — number of integers in the basis

    Returns a callable ``(n: int, h_current: Tensor) -> Tensor``.
    """
    W_combined_pinv = torch.linalg.pinv(W_combined, rtol=_PINV_RTOL)

    def write_combined_pinv(n: int, _h_current: Tensor) -> Tensor:
        c = n - ans_lo
        c_clamped = max(0, min(n_candidates - 1, c))
        b_n = B_combined[c_clamped]  # (K_h + K_d,)
        h_target = b_n @ W_combined  # (d_model,)
        return h_target.float()

    # The "projection" variant: preserve orthogonal complement of the combined
    # subspace, only overwrite the combined-subspace component.
    P_combined = W_combined_pinv @ W_combined  # (d_model, d_model), rank K_h + K_d
    I_d = torch.eye(P_combined.shape[0], dtype=P_combined.dtype)
    P_orth = I_d - P_combined

    def write_combined_projection(n: int, h_current: Tensor) -> Tensor:
        c = n - ans_lo
        c_clamped = max(0, min(n_candidates - 1, c))
        b_n = B_combined[c_clamped]
        h_chars = b_n @ W_combined
        h_orth = (P_orth @ h_current.float().cpu()).float()
        return (h_orth + h_chars).float()

    return {
        "combined-pinv": write_combined_pinv,
        "combined-projection": write_combined_projection,
    }


__all__ = [
    "DigitDecoder",
    "extract_digit_decoder",
    "build_combined_basis",
    "make_combined_write_fn",
]
