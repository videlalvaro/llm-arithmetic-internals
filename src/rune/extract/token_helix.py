"""Readout-side helix extractor — Lane 2.E two-sided extension.

Extracts a token-character basis from ``lm_head.weight`` (or ``embed_out``
in GPTNeoX terminology) to produce a ``TokenHelixDecoder`` that captures the
phase convention and period structure the readout actually uses when decoding
arithmetic answers.

Cheat-audit note (relaxed from the standard black-box discipline):
  - This module READS ``lm_head.weight[token_ids, :]`` directly.  This is
    intentional: the readout-side extraction is precisely about reading the
    structure the unembedding matrix uses to decode integer tokens.  This is
    the *source of structure*, not a back-channel shortcut.
  - See ``cheat_audit_relaxation_rationale`` in
    ``scripts/pythia_2.8b_two_sided.py`` for the explicit justification.
  - All other black-box disciplines remain: no model.config reads, no
    forward pre-hooks, no embed_in.weight reads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rune.extract.clock import _PINV_RTOL

# ─── Public result type ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenHelixDecoder:
    """Readout-side helix extracted from lm_head.weight.

    All tensors live on CPU (float32).

    Attributes
    ----------
    periods
        Helix periods fitted to the token rows of lm_head.weight.
    integer_token_ids
        BPE token IDs for integers 0..V_int-1 (the rows extracted from
        lm_head.weight).
    answer_value_range
        ``(lo, hi)`` — the range of integer answers fitted.
    B_token
        Character matrix ``(V_int, 2K+1)`` with rows
        ``[1, cos(2πc/T_1), sin(2πc/T_1), ..., cos(2πc/T_K), sin(2πc/T_K)]``
        for c ∈ [0, V_int).
    W_token
        Regression matrix ``(2K+1, d_model)`` such that
        ``B_token @ W_token ≈ lm_head.weight[token_ids, :]``.
        Fitted via regularised least squares on the fit fold.
    R2_per_period
        Held-out R² per period (2D block of cos/sin columns vs the actual
        lm_head rows, projected onto the 2D character subspace for that T).
    R2_global
        Overall held-out R²: how well ``B_token @ W_token`` reconstructs
        the full lm_head rows on the holdout fold.
    A_per_period
        Nanda Eq. (1) amplitude (mean over holdout tokens) for each period.
    phi_per_period
        Nanda Eq. (1) phase offset for each period (radians), estimated from
        the angle of the (W_token[cos_col], W_token[sin_col]) 2-vector
        averaged over d_model.
    """

    periods: tuple[int, ...]
    integer_token_ids: Tensor
    answer_value_range: tuple[int, int]
    B_token: Tensor
    W_token: Tensor
    R2_per_period: dict[int, float]
    R2_global: float
    A_per_period: dict[int, float]
    phi_per_period: dict[int, float]


# ─── Main public function ──────────────────────────────────────────────────────


def extract_token_helix(
    lm_head: nn.Module,
    integer_token_ids: Sequence[int],
    answer_value_range: tuple[int, int],
    periods: tuple[int, ...],
    *,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> TokenHelixDecoder:
    """Extract a readout-side helix basis from ``lm_head.weight``.

    Parameters
    ----------
    lm_head
        The unembedding module.  Must expose a ``weight`` attribute of shape
        ``(vocab_size, d_model)`` — standard for nn.Linear.
    integer_token_ids
        BPE token IDs corresponding to integers ``lo, lo+1, ..., hi``.
        Length must equal ``hi - lo + 1``.
    answer_value_range
        ``(lo, hi)`` inclusive range of integers to fit.
    periods
        Helix periods to include in the character basis.
    holdout_frac
        Fraction of integers [lo, hi] to hold out for evaluation.
    seed
        Random seed for the fit/holdout split.

    Returns
    -------
    TokenHelixDecoder
        The fitted readout-side basis and associated fit metrics.

    Notes
    -----
    Cheat audit: this function READS ``lm_head.weight`` directly.  This is
    intentional for the readout-side extraction.  The prior black-box audit's
    "no lm_head.weight reads" was for residual extraction discipline; here,
    reading lm_head.weight is the entire point.
    """
    # ── 1. Read lm_head weights ───────────────────────────────────────────────
    W_lm: Tensor = lm_head.weight.detach().float().cpu()
    # W_lm shape: (vocab_size, d_model)

    tok_ids = torch.tensor(list(integer_token_ids), dtype=torch.long)
    lo, hi = answer_value_range
    v_int = hi - lo + 1

    if len(integer_token_ids) != v_int:
        raise ValueError(
            f"integer_token_ids length {len(integer_token_ids)} != "
            f"hi - lo + 1 = {v_int} for answer_value_range=({lo}, {hi})."
        )

    W_tok = W_lm[tok_ids]  # (V_int, d_model) — lm_head rows for integer tokens

    # ── 2. Build character matrix B_token ─────────────────────────────────────
    c_vals = torch.arange(v_int, dtype=torch.float32) + lo  # c in [lo, hi]
    B_token = _build_character_matrix(c_vals, periods)  # (V_int, 2K+1)

    # ── 3. Fit / holdout split ────────────────────────────────────────────────
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(v_int, generator=rng)
    n_holdout = max(1, int(holdout_frac * v_int))
    n_fit = v_int - n_holdout
    fit_idx = perm[:n_fit]
    holdout_idx = perm[n_fit:]

    B_fit = B_token[fit_idx]  # (n_fit, 2K+1)
    W_fit = W_tok[fit_idx]    # (n_fit, d_model)

    B_ho = B_token[holdout_idx]   # (n_holdout, 2K+1)
    W_ho = W_tok[holdout_idx]     # (n_holdout, d_model)

    # ── 4. Regularised least-squares fit: W_token = pinv(B_fit) @ W_fit ──────
    B_fit_pinv = torch.linalg.pinv(B_fit, rtol=_PINV_RTOL)  # (2K+1, n_fit)
    W_token = B_fit_pinv @ W_fit  # (2K+1, d_model)

    # ── 5. Evaluate on holdout ────────────────────────────────────────────────
    W_ho_pred = B_ho @ W_token  # (n_holdout, d_model)

    # Global R²
    W_ho_mean = W_ho.mean(dim=0, keepdim=True)
    ss_tot = float(((W_ho - W_ho_mean) ** 2).sum().item())
    ss_res = float(((W_ho - W_ho_pred) ** 2).sum().item())
    R2_global = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Per-period R²: project onto the 2D subspace for each period
    basis_dim = 1 + 2 * len(periods)
    R2_per_period: dict[int, float] = {}
    A_per_period: dict[int, float] = {}
    phi_per_period: dict[int, float] = {}

    for T_idx, T in enumerate(periods):
        cos_col = 1 + 2 * T_idx
        sin_col = 2 + 2 * T_idx
        if sin_col >= basis_dim:
            R2_per_period[T] = 0.0
            A_per_period[T] = 0.0
            phi_per_period[T] = 0.0
            continue

        # Predicted from this period's 2D block only
        B_period = B_ho[:, [0, cos_col, sin_col]]  # (n_holdout, 3) w/ bias
        W_period = W_token[[0, cos_col, sin_col], :]  # (3, d_model)
        W_ho_period_pred = B_period @ W_period  # (n_holdout, d_model)

        ss_tot_p = float(((W_ho - W_ho_mean) ** 2).sum().item())
        ss_res_p = float(((W_ho - W_ho_period_pred) ** 2).sum().item())
        R2_per_period[T] = max(0.0, 1.0 - ss_res_p / ss_tot_p) if ss_tot_p > 1e-12 else 0.0

        # Nanda Eq. (1) amplitude and phase: the weight vector for cos and sin
        # components of period T tells us how this period is read by the logit.
        # W_token[cos_col, :] is the d_model-vector applied to cos(2πc/T);
        # its norm averaged over d_model gives amplitude.
        w_cos = W_token[cos_col, :]  # (d_model,)
        w_sin = W_token[sin_col, :]  # (d_model,)
        amp = float(torch.sqrt(w_cos ** 2 + w_sin ** 2).mean().item())
        A_per_period[T] = amp
        # Phase: angle of the mean (w_cos, w_sin) 2-vector
        phi = float(torch.atan2(w_sin.mean(), w_cos.mean()).item())
        phi_per_period[T] = phi

    return TokenHelixDecoder(
        periods=periods,
        integer_token_ids=tok_ids,
        answer_value_range=answer_value_range,
        B_token=B_token,
        W_token=W_token,
        R2_per_period=R2_per_period,
        R2_global=float(R2_global),
        A_per_period=A_per_period,
        phi_per_period=phi_per_period,
    )


# ─── Private helpers ───────────────────────────────────────────────────────────


def _build_character_matrix(
    c_vals: Tensor,
    periods: tuple[int, ...],
    *,
    affine: bool = True,
) -> Tensor:
    """Build character matrix B with columns
    [1, cos(2πc/T_1), sin(2πc/T_1), ..., cos(2πc/T_K), sin(2πc/T_K)].

    Parameters
    ----------
    c_vals
        (V,) float tensor of integer values c.
    periods
        Helix periods.
    affine
        If True (default), prepend a constant column of 1s.

    Returns
    -------
    Tensor of shape (V, 1 + 2*K) if affine else (V, 2*K).
    """
    cols = []
    if affine:
        cols.append(torch.ones(len(c_vals), 1, dtype=torch.float32))
    for T in periods:
        angle = 2.0 * math.pi * c_vals / T
        cols.append(torch.cos(angle).unsqueeze(1))
        cols.append(torch.sin(angle).unsqueeze(1))
    return torch.cat(cols, dim=1)


__all__ = [
    "TokenHelixDecoder",
    "extract_token_helix",
]
