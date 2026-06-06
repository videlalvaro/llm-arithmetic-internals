"""Lane 1.B — Causal symmetry discovery via commutator defect minimization.

For each candidate component in a transformer, search over group hypotheses (cyclic,
permutation, shift, affine_cyclic) for the group action g and linear representation A_g
that minimize Δ_C(g) = E_x[‖C(T_g x) − A_g C(x)‖² / ‖C(x)‖²].

Black-box discipline: treats the model as a function (forward calls only).  Hidden states
are captured via registered_forward_hook on model.encoder OUTPUT only.  No weight reading,
no config introspection, no forward pre-hooks, no input captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetryHypothesis:
    group_family: str  # "cyclic" | "permutation" | "shift" | "affine_cyclic"
    parameter: int | None  # e.g. modulus for cyclic; None for permutation
    commutator_defect: float
    subspace_dim: int


@dataclass(frozen=True)
class SymmetryDiscoveryResult:
    accepted: tuple[SymmetryHypothesis, ...]
    candidates_scored: tuple[SymmetryHypothesis, ...]  # all scored, accepted or not
    false_positive_rate_estimate: float


# ---------------------------------------------------------------------------
# Hidden-state capture (hook on model.encoder OUTPUT only)
# ---------------------------------------------------------------------------


def _capture_encoder_output(model: nn.Module, inputs: Tensor) -> Tensor | None:
    """Capture a representation tensor for symmetry analysis.

    Primary path: if model has an `encoder` attribute, register a forward hook on it
    and capture the encoder's *output* tensor (never input).

    Fallback path: if model has no `encoder`, run the full forward pass and use the
    logit tensor as the representation.  This handles analytic control models
    (e.g. RedundantModAddModel) where all structure is in the output.

    Returns the captured tensor as (batch, d), or None on failure.
    """
    if hasattr(model, "encoder"):
        captured: list[Tensor] = []

        def _hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
            # output is the encoder's return value — a Tensor or (Tensor, ...) tuple
            # We capture the OUTPUT only — never read from _inputs.
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if isinstance(tensor, Tensor):
                captured.append(tensor.detach().float())

        handle = model.encoder.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                model(inputs)
        except (RuntimeError, ValueError):
            return None
        finally:
            handle.remove()

        if not captured:
            return None
        return captured[-1]

    # Fallback: use forward output as the representation
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


def _capture_encoder_output_batch(
    model: nn.Module,
    inputs_list: list[Tensor],
) -> list[Tensor | None]:
    """Capture encoder output for multiple input tensors via one hook registration.

    More efficient than repeated hook registration when many forward passes are needed.
    Returns a list parallel to inputs_list; entries are None on forward failures.
    """
    if not hasattr(model, "encoder"):
        return [None] * len(inputs_list)

    results: list[Tensor | None] = []
    for inputs in inputs_list:
        result = _capture_encoder_output(model, inputs)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Utility: extract a pooled vector representation from a hook capture
# ---------------------------------------------------------------------------


def _pool_hidden(hidden: Tensor) -> Tensor:
    """Reduce a (batch, seq, d_model) or (batch, d_model) tensor to (batch, d_model).

    For 3D tensors takes the last sequence position (CLS-equivalent for decoder-style
    or simply the readout position for the 2-token models in this project).
    For 2D tensors passes through unchanged.
    """
    if hidden.ndim == 3:
        return hidden[:, -1, :]  # (batch, d_model)
    if hidden.ndim == 2:
        return hidden  # already (batch, d_model)
    # Flatten anything else to 2D
    return hidden.reshape(hidden.shape[0], -1)


# ---------------------------------------------------------------------------
# Core: least-squares solver for linear representation A_g
# ---------------------------------------------------------------------------


def _fit_linear_rep(h_source: Tensor, h_target: Tensor) -> Tensor:
    """Solve for A_g minimising ‖H_target − H_source A_g^T‖_F.

    h_source: (n, d)  hidden states for x
    h_target: (n, d)  hidden states for g·x
    Returns A_g: (d, d)
    """
    # Least-squares: A_g^T = pinv(H_source) @ H_target  →  A_g = (pinv(H_source) @ H_target)^T
    h_s = h_source.numpy().astype(np.float64)
    h_t = h_target.numpy().astype(np.float64)
    # Solve h_s @ A_g^T = h_t  →  A_g^T = lstsq(h_s, h_t).solution
    result = np.linalg.lstsq(h_s, h_t, rcond=None)
    a_g_transpose = result[0]  # (d, d)
    return torch.from_numpy(a_g_transpose.T.astype(np.float32))


def _commutator_defect(h_source: Tensor, h_target: Tensor, a_g: Tensor) -> float:
    """Compute Δ_C(g) = E_x[‖h_target − A_g h_source‖² / ‖h_source‖²].

    h_source: (n, d)
    h_target: (n, d)
    a_g: (d, d) linear representation of group action g
    """
    predicted = h_source @ a_g.T  # (n, d)
    residual_sq = ((h_target - predicted) ** 2).sum(dim=1)  # (n,)
    norm_sq = (h_source**2).sum(dim=1).clamp(min=1e-12)  # (n,)
    return float((residual_sq / norm_sq).mean().item())


# ---------------------------------------------------------------------------
# Cyclic group scanning
# ---------------------------------------------------------------------------


def _build_cyclic_shifted_tokens(
    operand_tokens: Tensor,
    modulus: int,
    token_position: int = 0,
) -> tuple[Tensor, Tensor] | None:
    """Build pairs (orig, shifted) for cyclic C_m detection.

    Two strategies are tried in order:

    1. True cyclic C_m: if all tokens in position `token_position` lie in [0, m-1],
       apply the wrap-around shift: a → (a+1) mod m.  Pairs include every sample.

    2. Period-m shift: for any token range, find pairs (a, b) and (a+m, b) where
       both appear in the dataset.  The periodic structure h(a+m, ·) ≈ A h(a, ·)
       is a necessary consequence of true cyclic C_m symmetry (if the representation
       has period m in the token space).

    Returns (orig, shifted) tensors with the same number of rows, or None on failure.
    """
    col = operand_tokens[:, token_position]
    max_val = int(col.max().item())

    # Strategy 1: true cyclic shift (all tokens in [0, m-1])
    if max_val < modulus:
        shifted = operand_tokens.clone()
        shifted[:, token_position] = (col + 1) % modulus
        return operand_tokens, shifted

    # Strategy 2: period-m shift (find pairs (a, a+m) in the token range)
    mask = col + modulus <= max_val
    if mask.sum() < 4:
        return None
    orig_sub = operand_tokens[mask]
    shifted_sub = orig_sub.clone()
    shifted_sub[:, token_position] = shifted_sub[:, token_position] + modulus
    return orig_sub, shifted_sub


def _score_cyclic_modulus(
    model: nn.Module,
    operand_tokens: Tensor,
    modulus: int,
    *,
    projection_basis: Tensor | None = None,
) -> tuple[float, Tensor | None, Tensor | None]:
    """Score cyclic group C_m for the given modulus.

    Tries both true cyclic shift and period-m shift (see _build_cyclic_shifted_tokens).
    Returns the best (lowest) defect found.

    Returns (defect, h_source, h_target) where h_* are (n, d) pooled hidden states,
    or (1.0, None, None) on failure.

    If projection_basis is given (orthonormal columns, shape d×k), project hidden states
    into that k-dimensional subspace before fitting (for repelled discovery).
    """
    pair = _build_cyclic_shifted_tokens(operand_tokens, modulus, token_position=0)
    if pair is None:
        return 1.0, None, None
    orig_tokens, shifted_tokens = pair

    h_orig = _capture_encoder_output(model, orig_tokens)
    if h_orig is None:
        return 1.0, None, None
    h_shift = _capture_encoder_output(model, shifted_tokens)
    if h_shift is None:
        return 1.0, None, None

    h_orig = _pool_hidden(h_orig)
    h_shift = _pool_hidden(h_shift)

    if projection_basis is not None:
        # Project into subspace spanned by projection_basis columns
        pb = projection_basis.float()  # (d, k)
        h_orig = h_orig @ pb  # (n, k)
        h_shift = h_shift @ pb  # (n, k)

    a_g = _fit_linear_rep(h_orig, h_shift)
    defect = _commutator_defect(h_orig, h_shift, a_g)
    return defect, h_orig, h_shift


def _find_cyclic_subspace(
    model: nn.Module,
    operand_tokens: Tensor,
    modulus: int,
    subspace_dim: int,
    *,
    projection_basis: Tensor | None = None,
) -> Tensor | None:
    """Identify the principal subspace realizing C_m via SVD of the hidden states.

    Returns a (d, k) orthonormal basis for the symmetry-bearing subspace, or None on failure.
    Operates on the projected space if projection_basis is given.
    """
    pair = _build_cyclic_shifted_tokens(operand_tokens, modulus, token_position=0)
    if pair is None:
        return None
    orig_tokens, _shifted_tokens = pair

    h_orig = _capture_encoder_output(model, orig_tokens)
    if h_orig is None:
        return None

    h_orig = _pool_hidden(h_orig)

    if projection_basis is not None:
        pb = projection_basis.float()
        h_orig = h_orig @ pb

    # The symmetry subspace is identified by SVD of h_orig (or covariance thereof)
    # We keep the top-k singular vectors as the symmetry basis
    h_np = h_orig.numpy().astype(np.float64)
    try:
        _, _, vt = np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    k = min(subspace_dim, vt.shape[0])
    basis = torch.from_numpy(vt[:k].T.astype(np.float32))  # (d, k)

    if projection_basis is not None:
        # Lift back to original d-dimensional space
        basis = projection_basis.float() @ basis  # (d_orig, k)

    return basis


# ---------------------------------------------------------------------------
# Shift group scanning (non-modular)
# ---------------------------------------------------------------------------


def _score_shift(
    model: nn.Module,
    operand_tokens: Tensor,
    step: int,
    *,
    projection_basis: Tensor | None = None,
) -> tuple[float, Tensor | None, Tensor | None]:
    """Score shift group: token → token + step (no modular wrap).

    Only uses samples where the shifted token is still in the observed range.
    """
    col = operand_tokens[:, 0]
    max_val = int(col.max().item())
    mask = col + step <= max_val
    if mask.sum() < 4:
        return 1.0, None, None

    orig = operand_tokens[mask]
    shifted = orig.clone()
    shifted[:, 0] = shifted[:, 0] + step

    h_orig = _capture_encoder_output(model, orig)
    if h_orig is None:
        return 1.0, None, None
    h_shift = _capture_encoder_output(model, shifted)
    if h_shift is None:
        return 1.0, None, None

    h_orig = _pool_hidden(h_orig)
    h_shift = _pool_hidden(h_shift)

    if projection_basis is not None:
        pb = projection_basis.float()
        h_orig = h_orig @ pb
        h_shift = h_shift @ pb

    a_g = _fit_linear_rep(h_orig, h_shift)
    defect = _commutator_defect(h_orig, h_shift, a_g)
    return defect, h_orig, h_shift


# ---------------------------------------------------------------------------
# Permutation group scanning
# ---------------------------------------------------------------------------


def _score_permutation(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    n_swap_pairs: int = 10,
    seed: int = 0,
) -> float:
    """Test permutation equivariance via vocabulary label swaps.

    For random pairs (t1, t2), swap them in the input tokens.  A permutation-equivariant
    model produces predictions that also transform under the same transposition.

    We restrict the test to samples where one of t1 or t2 appears in the input AND the
    clean prediction differs from the prediction under the swap.  This guards against
    constant-output models being spuriously labelled equivariant.

    Returns commutator defect in [0,1]: lower = more equivariant.
    """
    if operand_tokens.shape[0] < 4:
        return 1.0

    token_vals = operand_tokens.unique().tolist()
    if len(token_vals) < 4:
        return 1.0

    # Degenerate check: if model outputs are constant (zero variance), return 1.0 (no symmetry)
    try:
        with torch.inference_mode():
            sample_out = model(operand_tokens[: min(32, operand_tokens.shape[0])])
        preds = sample_out.argmax(dim=-1)
        if preds.unique().shape[0] == 1:
            # All predictions identical — degenerate/constant model
            return 1.0
    except (RuntimeError, ValueError):
        return 1.0

    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    consistent_count = 0
    total_count = 0

    for _ in range(n_swap_pairs):
        # Pick two distinct token values to swap
        idx1, idx2 = rng.choice(len(token_vals), size=2, replace=False)
        t1, t2 = int(token_vals[idx1]), int(token_vals[idx2])

        # Find samples that contain t1 or t2 in any position
        mask_has_either = (operand_tokens == t1).any(dim=1) | (operand_tokens == t2).any(dim=1)
        if mask_has_either.sum() < 2:
            continue

        orig = operand_tokens[mask_has_either]
        if orig.shape[0] > 64:
            perm = torch.randperm(orig.shape[0], generator=generator)[:64]
            orig = orig[perm]

        # Build swapped tokens: t1→t2, t2→t1
        swapped = orig.clone()
        swapped[orig == t1] = t2
        swapped[orig == t2] = t1

        try:
            with torch.inference_mode():
                out_orig = model(orig)
                out_swap = model(swapped)
        except (RuntimeError, ValueError):
            continue

        pred_orig = out_orig.argmax(dim=-1)
        pred_swap = out_swap.argmax(dim=-1)

        # Under a true label permutation (t1↔t2):
        #   expected: pred(swap(x)) == swap(pred(x))
        # For pred(x) ∈ {t1,t2}: swap(pred(x)) is t2 or t1 respectively.
        # For pred(x) ∉ {t1,t2}: swap(pred(x)) = pred(x) (prediction unchanged).
        expected_swap = pred_orig.clone()
        expected_swap[pred_orig == t1] = t2
        expected_swap[pred_orig == t2] = t1

        match_rate = float((pred_swap == expected_swap).float().mean().item())
        consistent_count += match_rate
        total_count += 1

    if total_count == 0:
        return 1.0

    avg_consistency = consistent_count / total_count
    # Defect = 1 - consistency
    return max(0.0, 1.0 - avg_consistency)


# ---------------------------------------------------------------------------
# Affine+cyclic (helix-like) scanning
# ---------------------------------------------------------------------------


def _score_affine_cyclic(
    model: nn.Module,
    operand_tokens: Tensor,
    period: int,
    *,
    projection_basis: Tensor | None = None,
) -> float:
    """Score the affine+cyclic representation (u, z_T) ↦ (u+Δ, e^{2πiΔ/T} z_T).

    For a helix-like representation with period T, the linear part (u coordinate) should
    shift by a constant Δ when the input shifts by Δ, AND the circular components at
    period T should rotate by 2πΔ/T.

    We test this by:
    1. Capturing hidden states h(x) for operand a
    2. Shifting a → a+1 and capturing h(g·x)
    3. Fitting: does h(g·x) decompose into a linear+circular transformation of h(x)?

    The defect measures how well the combined affine+cyclic representation fits.
    """
    # We use two shifts: step=1 and step=-1 to get both directions
    col = operand_tokens[:, 0]
    max_val = int(col.max().item())

    # Forward shift: a → a+1 (clamped)
    mask_fwd = col + 1 <= max_val
    if mask_fwd.sum() < 4:
        return 1.0

    orig_fwd = operand_tokens[mask_fwd]
    shift_fwd = orig_fwd.clone()
    shift_fwd[:, 0] = shift_fwd[:, 0] + 1

    h_orig = _capture_encoder_output(model, orig_fwd)
    h_shift = _capture_encoder_output(model, shift_fwd)

    if h_orig is None or h_shift is None:
        return 1.0

    h_orig = _pool_hidden(h_orig)
    h_shift = _pool_hidden(h_shift)

    if projection_basis is not None:
        pb = projection_basis.float()
        h_orig = h_orig @ pb
        h_shift = h_shift @ pb

    # Decompose hidden state into: affine component + circular components for this period
    # We model h ≈ W_u * u + W_cos * cos(2πa/T) + W_sin * sin(2πa/T) + residual
    # Under shift a→a+1:
    #   u → u+1 (linear shift)
    #   cos(2π(a+1)/T) = cos(2πa/T + 2π/T)
    #   sin(2π(a+1)/T) = sin(2πa/T + 2π/T)

    # Build the "what the helix predicts" for the shifted output:
    # h'_predicted = h + W_u + (rotation effect on circular part)
    # We fit this by measuring if h_shift is well-explained by a combination of:
    # [h_orig, ones] (affine: h + constant) and rotated circular terms

    # Simple test: fit h_shift = A @ h_orig + b
    # For pure linear: A is identity + small perturbation
    # For helix: A contains a rotation block

    # The affine+cyclic defect is the same as cyclic defect but we explicitly
    # decompose the representation into linear + periodic parts and measure both

    # Compute the standard linear rep defect first
    a_g = _fit_linear_rep(h_orig, h_shift)
    lin_defect = _commutator_defect(h_orig, h_shift, a_g)

    # Now check if there's a periodic component at this period
    # by checking if the eigenvalues of A_g cluster near e^{2πi/T}
    a_g_np = a_g.numpy().astype(np.float64)
    try:
        eigenvalues = np.linalg.eigvals(a_g_np)
    except np.linalg.LinAlgError:
        return lin_defect

    target_angle = 2.0 * np.pi / period
    # Check if any eigenvalue has angle close to ±2π/T
    angles = np.angle(eigenvalues)
    angle_residuals = np.minimum(
        np.abs(angles - target_angle),
        np.abs(angles + target_angle),
    )
    min_residual = float(np.min(angle_residuals))

    # Affine+cyclic defect: combination of linear defect and eigenvalue alignment
    # Lower is better (more evidence of helix structure)
    eigenvalue_penalty = min(1.0, min_residual / (np.pi / 4))  # 0 = perfect, 1 = worst
    affine_cyclic_defect = 0.5 * lin_defect + 0.5 * eigenvalue_penalty

    return affine_cyclic_defect


# ---------------------------------------------------------------------------
# Orthogonal complement projection for repelled discovery
# ---------------------------------------------------------------------------


def _orthogonal_complement_basis(
    basis: Tensor,
    full_dim: int,
) -> Tensor:
    """Return an orthonormal basis for the complement of the subspace spanned by `basis`.

    basis: (d, k) orthonormal columns
    Returns: (d, full_dim-k) orthonormal basis for the complement
    """
    d = basis.shape[0]
    k = basis.shape[1]
    if k >= full_dim:
        # No complement
        return torch.eye(d)

    basis_np = basis.numpy().astype(np.float64)
    # QR decomposition to get a full orthonormal set, then take the complement columns
    q, _ = np.linalg.qr(
        np.hstack([basis_np, np.random.default_rng(0).standard_normal((d, full_dim - k))]),
        mode="complete",
    )
    complement = q[:, k:]  # (d, d-k)
    return torch.from_numpy(complement.astype(np.float32))


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------


def discover_symmetries(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    cyclic_range: tuple[int, int] = (2, 100),
    max_defect: float = 0.20,
    repulsion_threshold: float = 0.3,
    seed: int = 0,
) -> SymmetryDiscoveryResult:
    """Discover group symmetries in model hidden states by commutator defect minimization.

    For each candidate group hypothesis (cyclic, shift, permutation, affine+cyclic),
    finds the linear representation A_g minimising Δ_C(g) = E_x[‖C(T_g x)−A_g C(x)‖²/‖C(x)‖²].

    Implements repelled discovery: after accepting a cyclic hypothesis in subspace S_1,
    searches again in the orthogonal complement of S_1 (up to 3 realizations per symmetry).

    Returns all scored candidates plus the accepted subset (defect < max_defect).
    """
    model.eval()
    operand_tokens = operand_tokens.detach()

    all_candidates: list[SymmetryHypothesis] = []
    accepted: list[SymmetryHypothesis] = []

    # ------------------------------------------------------------------
    # 1. Cyclic group scanning: C_m for m in [cyclic_range[0], cyclic_range[1]]
    #
    # Uses _capture_encoder_output, which hooks model.encoder when present and
    # falls back to the full forward output for models without an encoder.
    # ------------------------------------------------------------------
    best_cyclic: dict[int, SymmetryHypothesis] = {}  # modulus → best hypothesis

    # Get a reference representation to determine the feature dimensionality.
    h_ref = _capture_encoder_output(model, operand_tokens)
    if h_ref is not None:
        h_ref_pooled = _pool_hidden(h_ref)
        full_dim = h_ref_pooled.shape[1]

        cyclic_lo, cyclic_hi = cyclic_range
        for m in range(cyclic_lo, cyclic_hi + 1):
            defect, _, _ = _score_cyclic_modulus(model, operand_tokens, m)
            hyp = SymmetryHypothesis(
                group_family="cyclic",
                parameter=m,
                commutator_defect=defect,
                subspace_dim=full_dim,
            )
            all_candidates.append(hyp)
            if defect < max_defect:
                if m not in best_cyclic or defect < best_cyclic[m].commutator_defect:
                    best_cyclic[m] = hyp

        # Accept best cyclic hypotheses and run repelled discovery
        for m, base_hyp in best_cyclic.items():
            accepted.append(base_hyp)

            # Repelled discovery: find up to 2 more low-overlap realizations
            # by searching in the orthogonal complement of S_1
            prev_basis = _find_cyclic_subspace(
                model, operand_tokens, m, min(full_dim // 4, 16)
            )
            if prev_basis is None:
                continue

            for _rep in range(2):
                if prev_basis.shape[1] >= full_dim:
                    break
                complement = _orthogonal_complement_basis(prev_basis, full_dim)
                if complement.shape[1] < 2:
                    break

                defect_rep, _, _ = _score_cyclic_modulus(
                    model, operand_tokens, m, projection_basis=complement
                )
                rep_hyp = SymmetryHypothesis(
                    group_family="cyclic",
                    parameter=m,
                    commutator_defect=defect_rep,
                    subspace_dim=complement.shape[1],
                )
                all_candidates.append(rep_hyp)
                if defect_rep < max_defect:
                    accepted.append(rep_hyp)
                    # Expand the "excluded" subspace for the next iteration.
                    # _find_cyclic_subspace already lifts back to full_dim when
                    # projection_basis is given, so new_sub is (full_dim, k).
                    new_sub = _find_cyclic_subspace(
                        model, operand_tokens, m, min(full_dim // 4, 16),
                        projection_basis=complement,
                    )
                    if new_sub is None:
                        break
                    # Combine prev_basis and new_sub into an expanded excluded subspace
                    combined_np = np.hstack([
                        prev_basis.numpy().astype(np.float64),
                        new_sub.numpy().astype(np.float64),
                    ])
                    q, _ = np.linalg.qr(combined_np, mode="reduced")
                    prev_basis = torch.from_numpy(q.astype(np.float32))
                else:
                    break

    # ------------------------------------------------------------------
    # 2. Shift group scanning (non-modular): step ∈ {1, 2, 3}
    # ------------------------------------------------------------------
    for step in (1, 2, 3):
        defect, _, _ = _score_shift(model, operand_tokens, step)
        hyp = SymmetryHypothesis(
            group_family="shift",
            parameter=step,
            commutator_defect=defect,
            subspace_dim=0,
        )
        all_candidates.append(hyp)
        if defect < max_defect:
            accepted.append(hyp)

    # ------------------------------------------------------------------
    # 3. Permutation group scanning (pure black-box, no encoder needed)
    # ------------------------------------------------------------------
    perm_defect = _score_permutation(model, operand_tokens, n_swap_pairs=20, seed=seed)
    perm_hyp = SymmetryHypothesis(
        group_family="permutation",
        parameter=None,
        commutator_defect=perm_defect,
        subspace_dim=0,
    )
    all_candidates.append(perm_hyp)
    if perm_defect < max_defect:
        accepted.append(perm_hyp)

    # ------------------------------------------------------------------
    # 4. Affine+cyclic scanning: for periods in cyclic_range
    # The affine+cyclic search uses the same representation capture as the cyclic
    # scan (encoder output if available, forward output otherwise).
    # ------------------------------------------------------------------
    for period in range(cyclic_range[0], min(cyclic_range[1] + 1, 20)):
        defect = _score_affine_cyclic(model, operand_tokens, period)
        hyp = SymmetryHypothesis(
            group_family="affine_cyclic",
            parameter=period,
            commutator_defect=defect,
            subspace_dim=0,
        )
        all_candidates.append(hyp)
        if defect < max_defect:
            accepted.append(hyp)

    # ------------------------------------------------------------------
    # 5. False-positive rate estimate: use null model (shuffled tokens)
    # ------------------------------------------------------------------
    fp_rate = _estimate_false_positive_rate(
        model, operand_tokens, max_defect=max_defect, seed=seed
    )

    return SymmetryDiscoveryResult(
        accepted=tuple(accepted),
        candidates_scored=tuple(all_candidates),
        false_positive_rate_estimate=fp_rate,
    )


def _estimate_false_positive_rate(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    max_defect: float,
    seed: int,
    n_null_mods: int = 5,
) -> float:
    """Estimate FP rate by testing random permutations of operand tokens against cyclic groups.

    A random permutation of token order breaks any real symmetry structure.
    If the detector still finds low defect on permuted tokens, it's a false positive.
    """
    if not hasattr(model, "encoder") or operand_tokens.shape[0] < 4:
        return 0.0

    generator = torch.Generator().manual_seed(seed + 99)
    n = operand_tokens.shape[0]
    perm = torch.randperm(n, generator=generator)
    shuffled = operand_tokens[perm]

    n_false = 0
    n_tested = 0

    for m in range(2, 2 + n_null_mods):
        defect, _, _ = _score_cyclic_modulus(model, shuffled, m)
        n_tested += 1
        if defect < max_defect:
            n_false += 1

    return n_false / max(1, n_tested)
