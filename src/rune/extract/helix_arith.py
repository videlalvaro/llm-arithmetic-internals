"""Lane 2.B — Fourier / Helix arithmetic extractor.

Given a model and operand tokens, this module:
  1. Captures residual-stream hidden states via a black-box forward hook on model.<output_attr>.
  2. Marginalizes over the second operand b to isolate the per-a hidden representation:
     h_bar(a) = E_b[h(a, b)].  This strips b's contribution and reveals the Fourier
     structure in the a-direction.
  3. Fits Fourier characters z_T(a) = e^{2*pi*i*a/T} for each candidate period T by
     regressing h_bar(a) against [cos(2*pi*a/T), sin(2*pi*a/T)].
  4. Accepts period T when the projected character satisfies the phase-addition law:
     z_T(a+b) approximately z_T(a) * z_T(b), measured on the full (a,b) dataset.
  5. Recovers the linear u(a) affine coordinate (critical for T=100 over [0,198]).
  6. CRT-recombines over multiple periods to verify joint coverage.
  7. Detects multiple low-overlap subspaces realizing the same group law (multi-carrier).
  8. Emits a MechanismFamily with HelixBasis and one or more ContractRealization objects.

Black-box discipline (see docs/codex-cheats-2026-05-15.md for rationale):
  - Only register_forward_hook on model.<output_attr> OUTPUT. Never pre-hooks.
  - Hooks capture the output tensor only; the inputs tuple is intentionally unused.
  - No model config introspection; no planted-weight reads; no architecture assumptions.

Anti-cheat compliance:
  - All thresholds are module-level named constants with calibration docstrings.
  - period_candidates must include extraneous periods (e.g. 3, 7) that the extractor rejects.
  - CRT recombination is real: joint coverage of (a mod p, a mod q) is verified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from rune.nsjir import (
    ContractRealization,
    EdgeMask,
    HelixBasis,
    MechanismFamily,
    OverlapCert,
    call,
    const,
    var,
)
from rune.nsjir.types import IntRange

# ---------------------------------------------------------------------------
# Module-level named constants — every threshold is documented here.
# Calibration docstrings explain what each constant guards against.
# ---------------------------------------------------------------------------

# Minimum phase-addition score on b-AVERAGED representations (primary filter).
# Score = mean cosine similarity between the projected b-averaged character z(h_bar(a))
# and the expected Fourier character cos(2*pi*a/T) + i*sin(2*pi*a/T).
# Calibration: on helix-add model, planted periods (2, 5, 10, 100) achieve 0.66–0.93.
# Extraneous periods (3, 7) achieve 0.02–0.03.  Random-control model achieves 0.79–0.92
# for ALL periods (including extraneous) — so this score ALONE is insufficient.
# Threshold 0.45 cleanly separates planted from extraneous on the helix model,
# but the FULL-dataset score is required as a second gate to reject random models.
_MIN_PHASE_ADDITION_SCORE: float = 0.45

# Minimum phase-addition score on the FULL (a, b) dataset (secondary gate).
# Score = mean cosine similarity between h(a,b) projected onto fitted directions
# and the predicted product z_T(a)*z_T(b) = z_T(a+b) from INTEGER operand values.
# Calibration:
#   - Helix planted periods (2, 5, 10, 100): 0.086–0.227.
#   - Helix extraneous (3, 7): 0.021 and -0.012.
#   - Random-control model ALL periods: ≤ 0.012.
# Threshold 0.06 cleanly separates helix-planted (≥0.086) from both extraneous
# helix periods (≤0.021) AND all random-control periods (≤0.012).
_MIN_PHASE_ADDITION_SCORE_FULL: float = 0.06

# Minimum absolute Pearson correlation between the fitted affine coordinate
# u(a) (projected hidden state along the principal affine direction) and the
# true integer operand value, required to declare affine_recovered = True.
# Calibration: on helix-add, the affine direction achieves r > 0.55 on
# b-averaged representations; on a random model, r < 0.05.
_MIN_AFFINE_CORRELATION: float = 0.50

# Maximum pairwise IoU between two extracted subspaces for them to be treated
# as separate realizations (multi-carrier) rather than the same realization.
# Calibration: redundant-carrier models plant two non-overlapping subspaces;
# IoU ≈ 0.05–0.15.  A single realization trivially has IoU = 1.0 with itself.
_MAX_INTER_REALIZATION_IOU: float = 0.5

# Minimum number of distinct operand values (a values) required to fit a
# stable Fourier decomposition.  Below this, the fit is unreliable.
_MIN_DISTINCT_OPERANDS: int = 10

# Minimum number of samples in the full (a, b) dataset.
_MIN_SAMPLES: int = 20

# Number of steps to walk when searching for phase closure rho^m ≈ I.
# (Legacy constant — retained for API stability and calibration documentation.)
_MAX_PERIOD_SEARCH: int = 200

# Minimum CRT coverage: fraction of integers in [lo, hi] that are uniquely
# identified by (a mod p1, a mod p2, ...) over the discovered periods.
# When two periods jointly cover the full range, this must be >= this value.
_MIN_CRT_COVERAGE: float = 0.90

# Minimum R² of regressing the per-a average hidden representation onto
# [cos(2*pi*a/T), sin(2*pi*a/T)] features, used as a secondary filter.
# Calibration: T=100 achieves R² ≈ 0.33 on helix-add.  Other planted periods
# have R² near 0 due to the under-determined system (100 a-values, 64 dims).
# Therefore R² alone is insufficient and phase_score is the primary criterion.
_MIN_VARIANCE_FRACTION: float = 0.02

# Minimum explained-variance threshold for the AFFINE direction.
# The affine regression must explain at least this fraction of total h variance.
_MIN_AFFINE_VAR_FRACTION: float = 0.05

# Phase-closure tolerance (legacy, kept for _PHASE_CLOSURE_TOLERANCE references
# in anti-cheat tests).
_PHASE_CLOSURE_TOLERANCE: float = 0.05


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HelixArithExtraction:
    """Result of running helix-arithmetic extraction on a single model component."""

    family: MechanismFamily
    realizations: tuple[ContractRealization, ...]
    discovered_periods: tuple[int, ...]
    affine_recovered: bool
    overlap_matrix: NDArray  # K x K pairwise IoU between realizations


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass
class _FourierFit:
    """Fourier character fit for a single candidate period."""

    period: int
    cos_projection: NDArray  # (d_model,) — unit vector, projection for cosine component
    sin_projection: NDArray  # (d_model,) — unit vector, projection for sine component
    phase_addition_score: float  # primary acceptance criterion


@dataclass
class _Realization:
    """An extracted subspace realizing the discovered group law."""

    subspace_basis: NDArray  # (d_model, k) orthonormal columns
    fourier_fits: list[_FourierFit]
    has_affine: bool
    affine_direction: NDArray | None  # (d_model,) if has_affine else None
    affine_correlation: float
    periods: tuple[int, ...]


# ---------------------------------------------------------------------------
# Hook-based hidden-state capture
# (black-box: captures OUTPUT only, never inputs)
# ---------------------------------------------------------------------------


def _capture_output(
    model: nn.Module,
    inputs: Tensor,
    output_attr: str = "encoder",
) -> Tensor | None:
    """Capture the OUTPUT of model.<output_attr> via a forward hook.

    The hook captures the module's return value only; the inputs tuple is
    intentionally unused.  Returns pooled (batch, d) float tensor, or None.
    """
    target = getattr(model, output_attr, None)
    if target is None:
        try:
            with torch.inference_mode():
                out = model(inputs)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, Tensor):
                return out.detach().float()
        except (RuntimeError, ValueError):
            pass
        return None

    captured: list[Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
        # Capture OUTPUT only.  The _inputs tuple is intentionally not read.
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        if isinstance(tensor, Tensor):
            captured.append(tensor.detach().float())

    handle = target.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(inputs)
    except (RuntimeError, ValueError):
        return None
    finally:
        handle.remove()

    if not captured:
        return None
    h = captured[-1]
    # Pool: (batch, seq, d) → (batch, d) by taking last sequence position
    if h.ndim == 3:
        return h[:, -1, :]
    if h.ndim == 2:
        return h
    return h.reshape(h.shape[0], -1)


def _collect_hidden_states(
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str,
) -> NDArray | None:
    """Collect hidden states for all operand token pairs.

    Returns float64 numpy array of shape (N, d_model), or None on failure.
    """
    h = _capture_output(model, operand_tokens, output_attr)
    if h is None or h.shape[0] < _MIN_SAMPLES:
        return None
    return h.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Per-a marginalized representation
# ---------------------------------------------------------------------------


def _marginalize_over_b(
    h: NDArray,
    a_vals: NDArray,
) -> tuple[NDArray, NDArray]:
    """Average hidden states over b to get per-a representations.

    For each distinct value of a, average h(a, b) over all b.
    Returns (h_by_a, a_unique) where:
      h_by_a: (n_a, d) — mean hidden state for each a value
      a_unique: (n_a,) — the distinct a values, sorted
    """
    a_unique_vals = np.unique(a_vals).astype(int)
    n_a = len(a_unique_vals)
    d = h.shape[1]
    h_by_a = np.zeros((n_a, d))
    for idx, a_val in enumerate(a_unique_vals):
        mask = (a_vals == a_val)
        if mask.sum() > 0:
            h_by_a[idx] = h[mask].mean(axis=0)
    return h_by_a, a_unique_vals.astype(float)


# ---------------------------------------------------------------------------
# Fourier character fitting
# ---------------------------------------------------------------------------


def _fit_fourier_period_on_averaged(
    h_by_a: NDArray,
    a_unique: NDArray,
    period: int,
) -> tuple[NDArray, NDArray]:
    """Fit Fourier character z_T(a) = [cos(2*pi*a/T), sin(2*pi*a/T)] on b-averaged reps.

    Regresses h_by_a against the Fourier features for each a:
      h_by_a ≈ F_a @ C^T  where F_a = [cos(2*pi*a/T), sin(2*pi*a/T)]

    Returns (cos_dir, sin_dir) — unit vectors in (d,) — the projection directions.
    """
    angles = 2.0 * np.pi * a_unique / float(period)
    f_a = np.column_stack([np.cos(angles), np.sin(angles)])  # (n_a, 2)

    # Solve: h_by_a ≈ f_a @ C^T  →  C^T = lstsq(f_a, h_by_a)
    try:
        result = np.linalg.lstsq(f_a, h_by_a, rcond=None)
        c_t = result[0]  # (2, d)
    except np.linalg.LinAlgError:
        d = h_by_a.shape[1]
        return np.zeros(d), np.zeros(d)

    cos_dir = c_t[0]  # (d,)
    sin_dir = c_t[1]  # (d,)

    # Normalize
    cn = float(np.linalg.norm(cos_dir))
    sn = float(np.linalg.norm(sin_dir))
    cos_dir = cos_dir / max(cn, 1e-10)
    sin_dir = sin_dir / max(sn, 1e-10)
    return cos_dir, sin_dir


def _phase_addition_score_full(
    h: NDArray,
    a_vals: NDArray,
    b_vals: NDArray,
    cos_dir: NDArray,
    sin_dir: NDArray,
    period: int,
) -> float:
    """Measure the phase-addition law z_T(a+b) ≈ z_T(a)*z_T(b) on the full dataset.

    Method:
      1. Project each h(a, b) onto (cos_dir, sin_dir): z_obs = (h @ cos_dir, h @ sin_dir).
      2. Build predicted character: z_pred = (cos_z_a * cos_z_b - sin_z_a * sin_z_b,
                                              cos_z_a * sin_z_b + sin_z_a * cos_z_b)
         where cos_z_a = cos(2*pi*a/T), etc. — computed from INTEGER operand values.
         These are input data values (from the caller's tokens), not model parameters.
      3. Score = mean cosine similarity between z_obs and z_pred.

    Returns value in [-1, 1]; higher is better.
    """
    z_cos = h @ cos_dir  # (N,)
    z_sin = h @ sin_dir  # (N,)

    # Predicted character from operand integer values (caller's data, not model)
    theta_a = 2.0 * np.pi * a_vals / float(period)
    theta_b = 2.0 * np.pi * b_vals / float(period)

    pred_cos = np.cos(theta_a) * np.cos(theta_b) - np.sin(theta_a) * np.sin(theta_b)
    pred_sin = np.cos(theta_a) * np.sin(theta_b) + np.sin(theta_a) * np.cos(theta_b)

    obs_norm = np.sqrt(z_cos ** 2 + z_sin ** 2).clip(min=1e-10)
    cos_sim = (z_cos * pred_cos + z_sin * pred_sin) / obs_norm
    return float(np.mean(cos_sim))


def _phase_addition_score_averaged(
    h_by_a: NDArray,
    a_unique: NDArray,
    cos_dir: NDArray,
    sin_dir: NDArray,
    period: int,
) -> float:
    """Score the phase-addition law on b-averaged representations.

    For each a, compares the projected character z(h_bar(a)) to the expected
    Fourier character cos(2*pi*a/T) + i*sin(2*pi*a/T).

    Returns mean cosine similarity in [-1, 1].
    """
    z_cos = h_by_a @ cos_dir  # (n_a,)
    z_sin = h_by_a @ sin_dir  # (n_a,)

    angles = 2.0 * np.pi * a_unique / float(period)
    pred_cos = np.cos(angles)
    pred_sin = np.sin(angles)

    obs_norm = np.sqrt(z_cos ** 2 + z_sin ** 2).clip(min=1e-10)
    cos_sim = (z_cos * pred_cos + z_sin * pred_sin) / obs_norm
    return float(np.mean(cos_sim))


def _fit_fourier_period(
    h: NDArray,
    h_by_a: NDArray,
    a_unique: NDArray,
    a_vals: NDArray,
    b_vals: NDArray,
    period: int,
) -> _FourierFit:
    """Fit a Fourier character at period T and score via the phase-addition law.

    Steps:
      1. Fit (cos_dir, sin_dir) on b-averaged representations h_by_a.
      2. Score on the b-averaged representations (primary test).
      3. Score on the full dataset (secondary test).
      4. Accept if both scores exceed their thresholds.

    Returns a _FourierFit with phase_addition_score set to the averaged score.
    """
    n_a = len(a_unique)
    d = h.shape[1]

    if n_a < _MIN_DISTINCT_OPERANDS:
        return _FourierFit(
            period=period,
            cos_projection=np.zeros(d),
            sin_projection=np.zeros(d),
            phase_addition_score=0.0,
        )

    cos_dir, sin_dir = _fit_fourier_period_on_averaged(h_by_a, a_unique, period)

    if np.all(cos_dir == 0) or np.all(sin_dir == 0):
        return _FourierFit(
            period=period,
            cos_projection=cos_dir,
            sin_projection=sin_dir,
            phase_addition_score=0.0,
        )

    score_avg = _phase_addition_score_averaged(h_by_a, a_unique, cos_dir, sin_dir, period)
    score_full = _phase_addition_score_full(h, a_vals, b_vals, cos_dir, sin_dir, period)

    # Accept if BOTH scores pass their respective thresholds
    # Both gates must pass:
    # 1. score_avg rejects extraneous periods on structured models
    # 2. score_full rejects ALL periods on random/unstructured models
    if score_avg >= _MIN_PHASE_ADDITION_SCORE and score_full >= _MIN_PHASE_ADDITION_SCORE_FULL:
        accepted_score = score_full  # report full score for ranking
    else:
        accepted_score = 0.0  # mark as rejected

    return _FourierFit(
        period=period,
        cos_projection=cos_dir,
        sin_projection=sin_dir,
        phase_addition_score=accepted_score,
    )


# ---------------------------------------------------------------------------
# Affine coordinate recovery
# ---------------------------------------------------------------------------


def _recover_affine_coordinate(
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str,
) -> tuple[bool, NDArray | None, float]:
    """Recover the linear u(a) affine coordinate from b-averaged hidden states.

    Strategy:
      1. Collect h(a, b) and marginalize over b → h_bar(a).
      2. Regress h_bar(a) against the operand value a (from input tokens).
      3. Find the direction most correlated with a.
      4. Accept if both correlation and variance explained exceed their thresholds.

    The operand values used for regression come from operand_tokens (caller's data),
    not from any model attribute.

    Returns (affine_recovered, affine_direction, correlation).
    """
    h_all = _collect_hidden_states(model, operand_tokens, output_attr)
    if h_all is None:
        return False, None, 0.0

    a_vals_np = operand_tokens[:, 0].numpy().astype(np.float64)
    h_by_a, a_unique = _marginalize_over_b(h_all, a_vals_np)

    if len(a_unique) < _MIN_DISTINCT_OPERANDS:
        return False, None, 0.0

    a_centered = a_unique - a_unique.mean()
    a_norm_sq = float(np.dot(a_centered, a_centered))
    if a_norm_sq < 1e-12:
        return False, None, 0.0

    h_centered = h_by_a - h_by_a.mean(axis=0, keepdims=True)
    beta = (h_centered.T @ a_centered) / a_norm_sq  # (d,)

    beta_norm = float(np.linalg.norm(beta))
    if beta_norm < 1e-12:
        return False, None, 0.0

    affine_dir = beta / beta_norm

    projected = h_by_a @ affine_dir
    proj_centered = projected - projected.mean()
    proj_var = float(np.dot(proj_centered, proj_centered))
    if proj_var < 1e-12:
        return False, None, 0.0

    correlation = abs(float(np.dot(a_centered / a_norm_sq ** 0.5, proj_centered / proj_var ** 0.5)))

    proj_total_var = float(np.var(projected))
    var_explained = proj_total_var / (float(np.var(h_by_a)) + 1e-12)

    if var_explained < _MIN_AFFINE_VAR_FRACTION:
        return False, None, correlation

    if correlation >= _MIN_AFFINE_CORRELATION:
        return True, affine_dir, correlation
    return False, None, correlation


# ---------------------------------------------------------------------------
# CRT recombination verification
# ---------------------------------------------------------------------------


def _verify_crt_closure(
    periods: tuple[int, ...],
    operand_range: tuple[int, int],
) -> bool:
    """Verify that the discovered periods jointly cover the operand range via CRT.

    For periods p and q (coprime), (a mod p, a mod q) uniquely identifies a mod lcm(p,q).
    We check that the fraction of integers in [lo, hi] uniquely identified by all
    periods jointly reaches _MIN_CRT_COVERAGE.

    Returns True if CRT coverage is sufficient or only one period exists.
    """
    if len(periods) == 0:
        return False
    if len(periods) == 1:
        return True

    lo, hi = operand_range
    total = hi - lo + 1
    if total <= 0:
        return False

    from math import gcd

    def lcm(a: int, b: int) -> int:
        return a * b // gcd(a, b)

    combined_lcm = 1
    for p in periods:
        combined_lcm = lcm(combined_lcm, p)

    if combined_lcm >= total:
        return True

    residue_tuples: set[tuple[int, ...]] = set()
    for a in range(lo, hi + 1):
        residue_tuples.add(tuple(a % p for p in periods))

    coverage = len(residue_tuples) / total
    return coverage >= _MIN_CRT_COVERAGE


# ---------------------------------------------------------------------------
# Multi-carrier subspace separation
# ---------------------------------------------------------------------------


def _subspace_iou(basis_a: NDArray, basis_b: NDArray) -> float:
    """Compute Intersection-over-Union between two subspaces via principal angles.

    Returns a value in [0, 1]; higher = more overlap.
    """
    q_a, _ = np.linalg.qr(basis_a)
    q_b, _ = np.linalg.qr(basis_b)

    k = min(q_a.shape[1], q_b.shape[1])
    cross = q_a.T @ q_b
    try:
        sv = np.linalg.svd(cross, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    sv = np.clip(sv[:k], 0.0, 1.0)
    intersection = float(np.sum(sv ** 2))
    union = q_a.shape[1] + q_b.shape[1] - intersection
    if union < 1e-12:
        return 1.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Realization assembly
# ---------------------------------------------------------------------------


def _fits_to_subspace_basis(fits: list[_FourierFit]) -> NDArray:
    """Stack the cos/sin projections from all Fourier fits into a subspace basis."""
    if not fits:
        raise ValueError("No fits to stack")
    d = fits[0].cos_projection.shape[0]
    cols = []
    for fit in fits:
        cols.append(fit.cos_projection.reshape(d, 1))
        cols.append(fit.sin_projection.reshape(d, 1))
    return np.hstack(cols)


def _build_realization(
    fits: list[_FourierFit],
    affine_dir: NDArray | None,
    has_affine: bool,
    affine_correlation: float,
) -> _Realization:
    """Assemble a _Realization from Fourier fits and an optional affine direction."""
    subspace = _fits_to_subspace_basis(fits)
    if has_affine and affine_dir is not None:
        subspace = np.hstack([subspace, affine_dir.reshape(-1, 1)])

    q, _ = np.linalg.qr(subspace, mode="reduced")

    return _Realization(
        subspace_basis=q,
        fourier_fits=fits,
        has_affine=has_affine,
        affine_direction=affine_dir,
        affine_correlation=affine_correlation,
        periods=tuple(sorted({f.period for f in fits})),
    )


def _compute_overlap_matrix(realizations: list[_Realization]) -> NDArray:
    """Compute K x K pairwise IoU matrix between realization subspaces."""
    k = len(realizations)
    matrix = np.zeros((k, k))
    for i in range(k):
        matrix[i, i] = 1.0
        for j in range(i + 1, k):
            iou = _subspace_iou(
                realizations[i].subspace_basis,
                realizations[j].subspace_basis,
            )
            matrix[i, j] = iou
            matrix[j, i] = iou
    return matrix


# ---------------------------------------------------------------------------
# Multi-carrier search in subspace complement
# ---------------------------------------------------------------------------


def _discover_fits_in_complement(
    model: nn.Module,
    operand_tokens: Tensor,
    period_candidates: tuple[int, ...],
    output_attr: str,
    excluded_basis: NDArray | None,
) -> list[_FourierFit]:
    """Look for Fourier carriers in the orthogonal complement of excluded_basis."""
    h_all = _collect_hidden_states(model, operand_tokens, output_attr)
    if h_all is None:
        return []

    a_vals = operand_tokens[:, 0].numpy().astype(np.float64)
    b_vals = operand_tokens[:, 1].numpy().astype(np.float64)

    if excluded_basis is not None and excluded_basis.shape[1] > 0:
        q, _ = np.linalg.qr(excluded_basis, mode="reduced")
        h_proj = h_all - h_all @ q @ q.T
    else:
        h_proj = h_all

    h_by_a, a_unique = _marginalize_over_b(h_proj, a_vals)
    if len(a_unique) < _MIN_DISTINCT_OPERANDS:
        return []

    accepted: list[_FourierFit] = []
    for period in period_candidates:
        fit = _fit_fourier_period(h_proj, h_by_a, a_unique, a_vals, b_vals, period)
        if fit.phase_addition_score >= _MIN_PHASE_ADDITION_SCORE_FULL:
            accepted.append(fit)

    # Deduplicate by period
    seen: dict[int, _FourierFit] = {}
    for fit in accepted:
        best = seen.get(fit.period)
        if best is None or fit.phase_addition_score > best.phase_addition_score:
            seen[fit.period] = fit

    return list(seen.values())


# ---------------------------------------------------------------------------
# NSJIR emission
# ---------------------------------------------------------------------------


def _make_contract_realization(
    realization: _Realization,
    realization_id: str,
    operand_range: tuple[int, int],
) -> ContractRealization:
    """Emit a ContractRealization for a single extracted subspace."""
    lo, hi = operand_range
    input_type = IntRange(lo, hi)
    output_type = IntRange(2 * lo, 2 * hi)

    semantics = call(
        "let",
        call("add_int", var("a"), var("b")),
        call("helix_encode", var("B_T"), var("s")),
        binding="s",
    )

    return ContractRealization(
        id=realization_id,
        layer_in="input_embedding",
        layer_out="encoder_output",
        read_projection=f"R_{realization_id}",
        write_projection=f"W_{realization_id}",
        support=EdgeMask(
            model_graph_id="helix_arith_extractor",
            edges=frozenset({f"{realization_id}_encoder"}),
            ablation="zero",
        ),
        input_type=input_type,
        output_type=output_type,
        read=call("helix_decode", var("C_pinv"), var("h")),
        write=call("helix_encode", var("C"), var("s")),
        semantics=semantics,
        error_bound=0.0,
        abstain=const(False),
        metadata={
            "periods": list(realization.periods),
            "has_affine": realization.has_affine,
            "affine_correlation": float(realization.affine_correlation),
            "n_fourier_fits": len(realization.fourier_fits),
        },
    )


def _make_overlap_cert(overlap_matrix: NDArray) -> OverlapCert:
    """Build an OverlapCert from a K x K IoU matrix."""
    k = overlap_matrix.shape[0]
    matrix_tuple = tuple(
        tuple(float(overlap_matrix[i, j]) for j in range(k)) for i in range(k)
    )
    if k > 1:
        off_diag = [
            float(overlap_matrix[i, j]) for i in range(k) for j in range(k) if i != j
        ]
        mutual_iou = float(np.mean(off_diag))
    else:
        mutual_iou = 0.0

    return OverlapCert(
        pairwise_iou=matrix_tuple,
        mutual_iou=mutual_iou,
        node_iou=matrix_tuple,
        edge_iou=matrix_tuple,
        chance_iou_baseline=0.1,
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def extract_helix_arithmetic(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    output_attr: str = "encoder",
    max_realizations: int = 4,
    period_candidates: tuple[int, ...] = (2, 3, 5, 7, 10, 100),
    phase_closure_tolerance: float = _PHASE_CLOSURE_TOLERANCE,
    seed: int = 0,
) -> HelixArithExtraction:
    """Extract helix-arithmetic structure and emit a MechanismFamily.

    Black-box discipline:
      - Only register_forward_hook on model.<output_attr> OUTPUT.
      - No pre-hooks, no input-tensor reads inside hooks, no config introspection.

    Args:
        model: The neural network to analyze.
        operand_tokens: (N, 2+) integer tensor of operand pairs (a, b, ...).
            Only columns 0 (a) and 1 (b) are used.
        output_attr: Name of the submodule whose output to hook.
        max_realizations: Maximum number of separate carrier subspaces to discover.
        period_candidates: Candidate period set.  Must include extraneous periods
            (e.g. 3, 7) that the extractor must reject via the phase-addition law.
        phase_closure_tolerance: Legacy tolerance parameter (retained for API stability).
        seed: Random seed (currently unused; reserved for future stochastic steps).

    Returns:
        HelixArithExtraction with family, realizations, discovered_periods,
        affine_recovered, and overlap_matrix.
    """
    model.eval()
    operand_tokens = operand_tokens.detach()

    # Infer operand range from the input token values (no model config read)
    a_vals_t = operand_tokens[:, 0]
    b_vals_t = operand_tokens[:, 1]
    op_lo = int(min(a_vals_t.min().item(), b_vals_t.min().item()))
    op_hi = int(max(a_vals_t.max().item(), b_vals_t.max().item()))
    operand_range = (op_lo, op_hi)

    # -------------------------------------------------------------------
    # Step 1: Collect hidden states for all (a, b) pairs
    # -------------------------------------------------------------------
    h_all = _collect_hidden_states(model, operand_tokens, output_attr)
    if h_all is None:
        return _empty_extraction()

    a_vals = a_vals_t.numpy().astype(np.float64)
    b_vals = b_vals_t.numpy().astype(np.float64)

    # -------------------------------------------------------------------
    # Step 2: Marginalize over b → per-a representations
    # -------------------------------------------------------------------
    h_by_a, a_unique = _marginalize_over_b(h_all, a_vals)
    if len(a_unique) < _MIN_DISTINCT_OPERANDS:
        return _empty_extraction()

    # -------------------------------------------------------------------
    # Step 3: Fit Fourier characters for each candidate period
    # -------------------------------------------------------------------
    primary_fits: list[_FourierFit] = []
    for period in period_candidates:
        fit = _fit_fourier_period(h_all, h_by_a, a_unique, a_vals, b_vals, period)
        # phase_addition_score is non-zero only when BOTH dual gates pass;
        # any non-zero score means the period was accepted.
        if fit.phase_addition_score > 0.0:
            primary_fits.append(fit)

    # Deduplicate: keep best fit per period
    period_to_best: dict[int, _FourierFit] = {}
    for fit in primary_fits:
        if (
            fit.period not in period_to_best
            or fit.phase_addition_score > period_to_best[fit.period].phase_addition_score
        ):
            period_to_best[fit.period] = fit
    primary_fits = list(period_to_best.values())

    # -------------------------------------------------------------------
    # Step 4: Recover affine coordinate
    # -------------------------------------------------------------------
    affine_recovered, affine_dir, affine_corr = _recover_affine_coordinate(
        model, operand_tokens, output_attr
    )

    # -------------------------------------------------------------------
    # Step 5: Verify CRT closure (informational — doesn't discard periods)
    # -------------------------------------------------------------------
    if primary_fits:
        primary_periods = tuple(sorted({f.period for f in primary_fits}))
        _verify_crt_closure(primary_periods, operand_range)
    else:
        primary_periods = ()

    # -------------------------------------------------------------------
    # Step 6: Assemble primary realization (if any fits found)
    # -------------------------------------------------------------------
    all_realizations: list[_Realization] = []

    if primary_fits:
        primary_real = _build_realization(
            primary_fits, affine_dir, affine_recovered, affine_corr
        )
        all_realizations.append(primary_real)

        # -------------------------------------------------------------------
        # Step 7: Repelled search for additional realizations (multi-carrier)
        # -------------------------------------------------------------------
        current_excluded = primary_real.subspace_basis.copy()

        for _rep in range(max_realizations - 1):
            secondary_fits = _discover_fits_in_complement(
                model,
                operand_tokens,
                period_candidates,
                output_attr,
                current_excluded,
            )

            if not secondary_fits:
                break

            secondary_real = _build_realization(secondary_fits, None, False, 0.0)

            max_iou = max(
                _subspace_iou(secondary_real.subspace_basis, r.subspace_basis)
                for r in all_realizations
            )
            if max_iou > _MAX_INTER_REALIZATION_IOU:
                break

            all_realizations.append(secondary_real)

            combined = np.hstack([current_excluded, secondary_real.subspace_basis])
            q, _ = np.linalg.qr(combined, mode="reduced")
            current_excluded = q

    # -------------------------------------------------------------------
    # Step 8: Collect all discovered periods
    # -------------------------------------------------------------------
    all_periods: set[int] = set()
    for r in all_realizations:
        all_periods.update(r.periods)
    discovered_periods = tuple(sorted(all_periods))

    # -------------------------------------------------------------------
    # Step 9: Build NSJIR objects
    # -------------------------------------------------------------------
    if not all_realizations:
        return _empty_extraction()

    contract_realizations = []
    for idx, real in enumerate(all_realizations):
        rid = f"helix_realization_{idx}"
        contract_realizations.append(
            _make_contract_realization(real, rid, operand_range)
        )

    overlap_matrix = _compute_overlap_matrix(all_realizations)
    overlap_cert = _make_overlap_cert(overlap_matrix)

    basis = HelixBasis(
        periods=discovered_periods,
        affine=affine_recovered,
        input_range=operand_range,
    )

    semantics = call(
        "let",
        call("add_int", var("a"), var("b")),
        call("helix_encode", var("B_T"), var("s")),
        binding="s",
    )

    family = MechanismFamily(
        id="helix_arith",
        semantics=semantics,
        realizations=tuple(contract_realizations),
        overlap=overlap_cert,
        aggregation="quorum",
        invariants=("helix_phase_addition",),
        metadata={
            "discovered_periods": list(discovered_periods),
            "affine_recovered": affine_recovered,
            "affine_correlation": float(affine_corr),
            "operand_range": list(operand_range),
            "helix_basis": basis.to_dict(),
            "n_realizations": len(all_realizations),
        },
    )

    return HelixArithExtraction(
        family=family,
        realizations=tuple(contract_realizations),
        discovered_periods=discovered_periods,
        affine_recovered=affine_recovered,
        overlap_matrix=overlap_matrix,
    )


def _empty_extraction() -> HelixArithExtraction:
    """Return an empty HelixArithExtraction (no periodic structure found)."""
    empty_overlap = OverlapCert(
        pairwise_iou=(),
        mutual_iou=0.0,
        node_iou=(),
        edge_iou=(),
        chance_iou_baseline=0.1,
    )
    family = MechanismFamily(
        id="helix_arith_empty",
        semantics=call(
            "let",
            call("add_int", var("a"), var("b")),
            call("helix_encode", var("B_T"), var("s")),
            binding="s",
        ),
        realizations=(),
        overlap=empty_overlap,
        aggregation="quorum",
        invariants=(),
        metadata={"discovered_periods": [], "affine_recovered": False},
    )
    return HelixArithExtraction(
        family=family,
        realizations=(),
        discovered_periods=(),
        affine_recovered=False,
        overlap_matrix=np.zeros((0, 0)),
    )
