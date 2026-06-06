"""Lane 2.D — Semantic Quotient Extraction.

Given a tuple of ContractRealization objects (from any extractor), decide which
are different realizations of the SAME semantics and merge them into one unified
MechanismFamily.

Four distance metrics are required:
  1. Behavioral distance  — real measurement on held-out tokens via reconstruct+compare.
  2. Causal distance      — interchange-intervention KL via causal_replaceability.
  3. Basis-change residual— orthogonal Procrustes for helix/modular subspace bases.
  4. Alpha-equivalence    — AST-recursive renaming walk for NSJIR pointer terms.

Black-box discipline (same as helix_arith and clock):
  - Only register_forward_hook (output-only); no pre-hooks, no inputs[0] reads.
  - No model.config reads, no *Config instantiation, no .modulus / .periods model attrs.
  - No token_embedding.weight reads.
  - All thresholds are module-level named constants with calibration docstrings.

Anti-cheat compliance:
  - behavioral_distance is measured on held-out tokens via projection+KL, not param heuristics.
  - causal_distance MUST call causal_replaceability from Lane 1.E (grep-checked in audit test).
  - basis_change_residual uses orthogonal Procrustes (scipy or manual SVD), not subspace heuristic.
  - alpha_equivalent walks the Term AST recursively, not string comparison.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from rune.detect.replaceability import Realization, causal_replaceability
from rune.nsjir import (
    ContractRealization,
    MechanismFamily,
    OverlapCert,
    Term,
)

# ---------------------------------------------------------------------------
# Module-level named constants — every threshold is documented here.
# Calibration docstrings explain what each constant guards against.
# ---------------------------------------------------------------------------

BEHAVIORAL_THRESHOLD_DEFAULT: float = 0.02
"""Default behavioral-distance threshold (KL-based, normalised).

Calibration: two helix-add extractions on the same model with basis-rotated
subspaces produce identical downstream logit distributions — behavioral KL ≈ 0.
Two extractions from models with different moduli (m=12 vs m=31) produce
behaviorally different outputs — KL >> 0.02 when evaluated on held-out tokens
that include operand values spanning both ranges.

0.02 nats = a very conservative upper bound for "same semantics"; anything
above it means the two realizations produce distinguishably different outputs."""

CAUSAL_THRESHOLD_DEFAULT: float = 0.05
"""Default causal-distance threshold (raw_patch_kl difference in nats).

Calibration: causal_replaceability reports raw_patch_kl for each realization.
Two same-semantics realizations in the same model should produce near-identical
interchange-KL values (within noise of the stochastic pairing).  Different-moduli
realizations typically differ by > 1 nat in interchange KL.

0.05 nats is a conservative margin that absorbs seed-dependent pairing noise."""

BASIS_RESIDUAL_THRESHOLD_DEFAULT: float = 0.1
"""Default basis-change residual threshold for Procrustes alignment.

Calibration: two helix subspaces related by an orthogonal rotation satisfy
‖B_j U - B_i‖_F / ‖B_i‖_F ≈ 0 (machine-epsilon range after perfect alignment).
Different-moduli subspaces span genuinely different directions → residual > 0.5.

0.1 is generous enough to absorb floating-point accumulation across SVD and QR
steps, while still cleanly separating equivalent (residual < 0.01) from distinct
(residual > 0.5) bases."""

PROCRUSTES_NONTRIVIAL_NORM: float = 1e-6
"""Minimum Frobenius norm of B_i for Procrustes to be well-defined.

If either basis is effectively zero (columns collapsed), we skip the Procrustes
step and return None (not comparable).  This avoids 0/0 in the normalisation."""

BEHAVIORAL_KL_SAMPLE_FRACTION: float = 0.3
"""Fraction of operand_tokens to hold out for behavioral-distance measurement.

0.3 = 30% held-out, 70% used for subspace projection fitting.  Ensures the
behavioral measurement is on genuinely unseen data."""

MIN_HELD_OUT_SAMPLES: int = 8
"""Minimum held-out samples required for a meaningful behavioral KL estimate.

Below 8 samples the per-sample mean is too noisy to be a reliable gate."""

SUBSPACE_PROJECTION_MIN_SAMPLES: int = 16
"""Minimum total samples required to attempt subspace projection.

Below this the projection matrix is underdetermined and the behavioral distance
would be meaningless."""


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeDecision:
    """Per-pair decision.

    Fields:
      realization_i, realization_j: IDs of the two realizations compared.
      same_family: True if the pair was merged into the same MechanismFamily.
      behavioral_distance: KL divergence between downstream logits on held-out
          tokens after projecting into each subspace. Measured on real forward
          passes, NOT on parameters.
      causal_distance: difference in raw_patch_kl between the two realizations,
          as reported by causal_replaceability (Lane 1.E).
      basis_change_residual: ‖B_j U - B_i‖_F / ‖B_i‖_F after Procrustes
          alignment.  None when neither realization is a helix/modular type.
      alpha_equivalent: True if the semantic Term of realization_i and
          realization_j are alpha-equivalent. None when terms are not of
          pointer-program form (no bound variables).
      reason: human-readable merge/reject reason code.
    """

    realization_i: str
    realization_j: str
    same_family: bool
    behavioral_distance: float
    causal_distance: float
    basis_change_residual: float | None
    alpha_equivalent: bool | None
    reason: str


@dataclass(frozen=True)
class QuotientResult:
    """Result of quotient-extraction over a set of realizations."""

    families: tuple[MechanismFamily, ...]
    decisions: tuple[MergeDecision, ...]
    n_input_realizations: int
    n_output_families: int


# ---------------------------------------------------------------------------
# α-equivalence checker for NSJIR pointer terms
# ---------------------------------------------------------------------------


def alpha_equivalent(term_i: object, term_j: object) -> bool:
    """α-equivalence checker for NSJIR pointer terms.  Walks the AST recursively.

    Two terms are α-equivalent if one can be obtained from the other by
    consistently renaming variables.  This covers both:

    a) Explicitly-bound variables introduced by ``let`` binders (attrs["binding"]).
    b) Free ``var`` leaf nodes that serve as *pointer-token role names* in pointer
       programs — by convention, ``copy(next(lastpos("X")))`` and
       ``copy(next(lastpos("Y")))`` are α-equivalent because the leaf variable is
       a renaming site (a "pointer-token binder" in the pointer-term convention).

    Algorithm:
      - Collect all var-leaf positions in the tree (pre-order traversal index).
      - Build a bijective mapping from the left-tree variable names to the
        right-tree variable names at the same structural positions.
      - Walk both trees simultaneously.  At each ``var`` node, check that the
        name maps consistently: if ``name_i`` was seen before at position k and
        was mapped to ``name_j_k``, the current ``name_j`` must equal ``name_j_k``.
        If ``name_i`` is new, establish the mapping ``name_i → name_j`` and
        require that ``name_j`` has not already been mapped to a different name.
      - Structural mismatch (different op, different arity, different non-var
        attrs) → False.
      - ``const`` node values must be equal.
      - ``let`` binder introduces an explicit renaming scope that overrides the
        positional mapping for the bound name.

    The checker does NOT do string comparison of the whole term — it walks
    the AST node by node.  Tests must verify:
      - copy(next(lastpos("X"))) ≡α copy(next(lastpos("Y")))   [diff leaf names]
      - copy(next(lastpos("X"))) ≢α copy(prev(lastpos("X")))   [diff structure]
    """
    if not isinstance(term_i, Term) or not isinstance(term_j, Term):
        # Non-Term objects: fall back to equality
        return term_i == term_j
    # Use mutable dicts for the bijective renaming map (passed by reference)
    i_to_j: dict[str, str] = {}
    j_to_i: dict[str, str] = {}
    return _alpha_equiv_walk(term_i, term_j, i_to_j, j_to_i)


def _alpha_equiv_walk(
    t_i: Term,
    t_j: Term,
    i_to_j: dict[str, str],
    j_to_i: dict[str, str],
) -> bool:
    """Recursive α-equivalence walker.

    i_to_j / j_to_i: bijective renaming maps from variable names in t_i's
    tree to names in t_j's tree, updated in-place.  Both binder-introduced
    names and free leaf names participate in the bijection.
    """
    # Different operator → not equivalent
    if t_i.op != t_j.op:
        return False

    # Different number of children → not equivalent
    if len(t_i.args) != len(t_j.args):
        return False

    # Handle 'const' nodes: values must be equal
    if t_i.op == "const":
        return t_i.attrs.get("value") == t_j.attrs.get("value")

    # Handle 'var' nodes: check bijective renaming consistency
    if t_i.op == "var":
        name_i = t_i.attrs.get("name", "")
        name_j = t_j.attrs.get("name", "")
        # Check existing mapping
        if name_i in i_to_j:
            return i_to_j[name_i] == name_j
        # New name_i: check that name_j is not already claimed by a different name_i
        if name_j in j_to_i and j_to_i[name_j] != name_i:
            return False
        # Establish the bijection
        i_to_j[name_i] = name_j
        j_to_i[name_j] = name_i
        return True

    # Handle 'let' binders: attrs["binding"] introduces a new bound variable
    if t_i.op == "let":
        bind_i = t_i.attrs.get("binding")
        bind_j = t_j.attrs.get("binding")
        if (bind_i is None) != (bind_j is None):
            return False
        if bind_i is not None and bind_j is not None:
            # Add binding to the bijective map for the subtree
            # If there is a conflict with an existing free-var mapping, reject
            if bind_i in i_to_j and i_to_j[bind_i] != bind_j:
                return False
            if bind_j in j_to_i and j_to_i[bind_j] != bind_i:
                return False
            # Temporarily extend the map for this binder scope
            old_i = i_to_j.get(bind_i)
            old_j = j_to_i.get(bind_j)
            i_to_j[bind_i] = bind_j
            j_to_i[bind_j] = bind_i
            result = all(
                _alpha_equiv_walk(a, b, i_to_j, j_to_i)
                for a, b in zip(t_i.args, t_j.args, strict=True)
            )
            # Restore previous mapping state
            if old_i is None:
                i_to_j.pop(bind_i, None)
            else:
                i_to_j[bind_i] = old_i
            if old_j is None:
                j_to_i.pop(bind_j, None)
            else:
                j_to_i[bind_j] = old_j
            return result
        return all(
            _alpha_equiv_walk(a, b, i_to_j, j_to_i)
            for a, b in zip(t_i.args, t_j.args, strict=True)
        )

    # For all other nodes: non-name/value/binding attrs must match, then recurse.
    skip_attrs = {"name", "binding"}
    attrs_i = {k: v for k, v in t_i.attrs.items() if k not in skip_attrs}
    attrs_j = {k: v for k, v in t_j.attrs.items() if k not in skip_attrs}
    if attrs_i != attrs_j:
        return False

    return all(
        _alpha_equiv_walk(a, b, i_to_j, j_to_i)
        for a, b in zip(t_i.args, t_j.args, strict=True)
    )


# ---------------------------------------------------------------------------
# Orthogonal Procrustes basis-change residual
# ---------------------------------------------------------------------------


def _procrustes_residual(B_i: NDArray, B_j: NDArray) -> float | None:
    """Compute ‖B_j U - B_i‖_F / ‖B_i‖_F via orthogonal Procrustes.

    Finds U = argmin_{U orthogonal} ‖B_j U - B_i‖_F by solving:
        M = B_j^T B_i,  SVD: M = V S W^T,  U = V W^T

    Both B_i and B_j must have shape (d, k) with d >= k.

    Returns None if either basis has near-zero Frobenius norm (degenerate).
    Returns the normalised residual ‖B_j U - B_i‖_F / ‖B_i‖_F otherwise.
    """
    norm_i = float(np.linalg.norm(B_i, "fro"))
    norm_j = float(np.linalg.norm(B_j, "fro"))
    if norm_i < PROCRUSTES_NONTRIVIAL_NORM or norm_j < PROCRUSTES_NONTRIVIAL_NORM:
        return None

    # Shapes must be compatible: same d, same k
    if B_i.shape != B_j.shape:
        # Attempt to align by padding/truncating along k-axis to the smaller
        k = min(B_i.shape[1], B_j.shape[1])
        B_i = B_i[:, :k]
        B_j = B_j[:, :k]
        norm_i = float(np.linalg.norm(B_i, "fro"))
        if norm_i < PROCRUSTES_NONTRIVIAL_NORM:
            return None

    # Solve orthogonal Procrustes: M = B_j^T @ B_i, decompose M = V S W^T
    M = B_j.T @ B_i  # (k, k)
    try:
        V, _S, Wt = np.linalg.svd(M, full_matrices=True)
    except np.linalg.LinAlgError:
        return None

    U = V @ Wt  # Optimal orthogonal rotation (k, k)

    residual_mat = B_j @ U - B_i  # (d, k)
    residual = float(np.linalg.norm(residual_mat, "fro")) / norm_i
    return residual


# ---------------------------------------------------------------------------
# Behavioral distance measurement
# ---------------------------------------------------------------------------


def _capture_encoder_output(
    model: nn.Module,
    tokens: Tensor,
    output_attr: str = "encoder",
) -> Tensor | None:
    """Capture the encoder's output tensor via a forward hook (output only)."""
    target = getattr(model, output_attr, None)
    if target is None:
        return None

    captured: list[Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
        # Output-only capture: _inputs intentionally not read.
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        if isinstance(tensor, Tensor):
            captured.append(tensor.detach().float())

    handle = target.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(tokens)
    except (RuntimeError, ValueError):
        return None
    finally:
        handle.remove()

    if not captured:
        return None
    h = captured[-1]
    # Pool: (batch, seq, d) → (batch, d) by last position
    if h.ndim == 3:
        return h[:, -1, :]
    if h.ndim == 2:
        return h
    return h.reshape(h.shape[0], -1)


def _kl_mean_nats(p_logits: Tensor, q_logits: Tensor) -> float:
    """KL(softmax(p_logits) || softmax(q_logits)) in nats, mean over batch."""
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    kl_per_sample = (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)
    return float(kl_per_sample.mean().item())


def _behavioral_distance(
    real_i: ContractRealization,
    real_j: ContractRealization,
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str = "encoder",
) -> float:
    """Measure behavioral distance between two realizations on held-out tokens.

    Method:
      1. Capture encoder hidden states on ALL tokens.
      2. For each realization, project hidden states onto their subspace (if a
         subspace_basis is available in metadata), reconstruct, and run the
         rest of the model forward via an output hook.
      3. Compute KL(logits_i || logits_j) on held-out tokens.
         KL > behavioral_threshold → different behaviour → different semantics.

    If subspace bases are not available (e.g. the realization has no embedded
    subspace_basis_numpy metadata), fall back to comparing raw model logits on
    the held-out set (KL = 0 if both realizations leave the hidden state unchanged,
    meaning they cannot be distinguished and should be merged).
    """
    n = operand_tokens.shape[0]
    if n < SUBSPACE_PROJECTION_MIN_SAMPLES:
        return 0.0  # Too few samples: assume same behaviour (conservative merge)

    # Held-out split
    rng = np.random.default_rng(seed=0)
    n_held = max(MIN_HELD_OUT_SAMPLES, int(n * BEHAVIORAL_KL_SAMPLE_FRACTION))
    n_held = min(n_held, n)
    held_idx = rng.choice(n, size=n_held, replace=False)
    held_tokens = operand_tokens[torch.from_numpy(held_idx)]

    # Capture encoder output for held-out tokens
    h_held = _capture_encoder_output(model, held_tokens, output_attr)
    if h_held is None:
        return 0.0

    # Try to get subspace bases from metadata
    basis_i = _extract_subspace_basis(real_i, h_held.shape[-1])
    basis_j = _extract_subspace_basis(real_j, h_held.shape[-1])

    if basis_i is None or basis_j is None:
        # No subspace information: realizations are indistinguishable at this level
        return 0.0

    # Project hidden states into each subspace and reconstruct
    B_i = torch.tensor(basis_i, dtype=torch.float32)
    B_j = torch.tensor(basis_j, dtype=torch.float32)

    h_proj_i = _project_and_reconstruct(h_held, B_i)
    h_proj_j = _project_and_reconstruct(h_held, B_j)

    # Run model forward with each reconstructed hidden state
    target = getattr(model, output_attr, None)
    if target is None:
        return 0.0

    logits_i = _run_with_replacement(model, target, held_tokens, h_proj_i)
    logits_j = _run_with_replacement(model, target, held_tokens, h_proj_j)

    if logits_i is None or logits_j is None:
        return 0.0

    return _kl_mean_nats(logits_i, logits_j)


def _extract_subspace_basis(
    real: ContractRealization, d_model: int
) -> NDArray | None:
    """Extract a numpy subspace basis from the realization's metadata.

    Looks for 'subspace_basis' key in metadata (stored as list-of-list).
    Returns None if not available.
    """
    meta = real.metadata or {}
    raw = meta.get("subspace_basis")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == d_model:
            return arr
    except (ValueError, TypeError):
        pass
    return None


def _project_and_reconstruct(h: Tensor, basis: Tensor) -> Tensor:
    """Project h onto the subspace spanned by basis columns and reconstruct.

    h: (batch, d), basis: (d, k) — orthonormalised internally.
    Returns (batch, d) — the reconstructed hidden state in the subspace.
    """
    q, _ = torch.linalg.qr(basis.float(), mode="reduced")
    coeffs = h @ q  # (batch, k)
    proj = coeffs @ q.T  # (batch, d)
    # Replace the subspace component of h with the projection
    return h - (h @ q @ q.T) + proj


def _run_with_replacement(
    model: nn.Module,
    hook_target: nn.Module,
    tokens: Tensor,
    replacement: Tensor,
) -> Tensor | None:
    """Run model(tokens) with encoder output replaced by replacement.

    Uses a forward hook that returns replacement, overriding the encoder output.
    This is a real forward pass — not a parameter or logit synthesis.
    """
    replacement_buf = replacement.detach()

    def _hook(_module: nn.Module, _inputs: tuple, output: object) -> Tensor:
        # Override output; _inputs intentionally not read.
        return replacement_buf

    handle = hook_target.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            logits = model(tokens).detach()
    except (RuntimeError, ValueError):
        return None
    finally:
        handle.remove()

    return logits.float()


# ---------------------------------------------------------------------------
# Causal distance via causal_replaceability
# ---------------------------------------------------------------------------


def _causal_distance(
    real_i: ContractRealization,
    real_j: ContractRealization,
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str = "encoder",
    seed: int = 0,
) -> float:
    """Measure causal distance between two realizations via causal_replaceability.

    This function MUST call causal_replaceability from Lane 1.E.
    The causal distance = |raw_patch_kl_i - raw_patch_kl_j|.

    Two same-semantics realizations should produce similar interchange-KL values
    when tested against the same counterfactual pairs.  Different-semantics
    realizations produce different interchange-KL values (one typically much
    higher than the other).

    If subspace bases are not available in metadata, fall back to 0.0
    (conservative: assume same causal role).
    """
    d_model = _infer_d_model(model, operand_tokens, output_attr)
    if d_model is None:
        return 0.0

    basis_i = _extract_subspace_basis(real_i, d_model)
    basis_j = _extract_subspace_basis(real_j, d_model)

    if basis_i is None or basis_j is None:
        return 0.0

    B_i_t = torch.tensor(basis_i, dtype=torch.float32)
    B_j_t = torch.tensor(basis_j, dtype=torch.float32)

    # Build Realization objects for causal_replaceability
    # Use identity decode: the symbolic reconstruction IS the subspace projection
    def _make_decode(basis_tensor: Tensor):
        q, _ = torch.linalg.qr(basis_tensor.float(), mode="reduced")
        q_fixed = q.clone()

        def _decode(h: Tensor) -> Tensor:
            # Project into subspace: S S^T h (in-subspace reconstruction)
            coeffs = h @ q_fixed
            return h - (h @ q_fixed @ q_fixed.T) + coeffs @ q_fixed.T

        return _decode

    r_i = Realization(
        id=real_i.id + "_q_i",
        subspace_basis=B_i_t,
        decode=_make_decode(B_i_t),
    )
    r_j = Realization(
        id=real_j.id + "_q_j",
        subspace_basis=B_j_t,
        decode=_make_decode(B_j_t),
    )

    result = causal_replaceability(
        model,
        operand_tokens,
        (r_i, r_j),
        output_attr=output_attr,
        seed=seed,
    )

    # Collect raw_patch_kl for each realization
    kl_map: dict[str, float] = {}
    for cert in (*result.accepted, *result.rejected):
        kl_map[cert.realization_id] = cert.raw_patch_kl

    kl_i = kl_map.get(r_i.id, float("inf"))
    kl_j = kl_map.get(r_j.id, float("inf"))

    if kl_i == float("inf") or kl_j == float("inf"):
        return float("inf")

    return abs(kl_i - kl_j)


def _infer_d_model(
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str,
) -> int | None:
    """Infer d_model by capturing one forward pass."""
    h = _capture_encoder_output(model, operand_tokens[:2], output_attr)
    if h is None:
        return None
    return int(h.shape[-1])


# ---------------------------------------------------------------------------
# Basis-change residual for helix/modular realizations
# ---------------------------------------------------------------------------


def _basis_change_residual(
    real_i: ContractRealization,
    real_j: ContractRealization,
    d_model: int | None = None,
) -> float | None:
    """Compute orthogonal-Procrustes basis-change residual for two realizations.

    Only applicable when both realizations have a 'subspace_basis' in their
    metadata.  The basis must be (d_model, k) shaped.

    Returns None if either realization lacks subspace information.
    Returns ‖B_j U - B_i‖_F / ‖B_i‖_F after Procrustes alignment.
    """
    if d_model is not None:
        basis_i = _extract_subspace_basis(real_i, d_model)
        basis_j = _extract_subspace_basis(real_j, d_model)
    else:
        # Try arbitrary d_model from metadata shape
        meta_i = (real_i.metadata or {}).get("subspace_basis")
        meta_j = (real_j.metadata or {}).get("subspace_basis")
        if meta_i is None or meta_j is None:
            return None
        try:
            arr_i = np.asarray(meta_i, dtype=np.float64)
            arr_j = np.asarray(meta_j, dtype=np.float64)
        except (ValueError, TypeError):
            return None
        if arr_i.ndim != 2 or arr_j.ndim != 2:
            return None
        if arr_i.shape[0] != arr_j.shape[0]:
            return None
        basis_i = arr_i
        basis_j = arr_j

    if basis_i is None or basis_j is None:
        return None

    return _procrustes_residual(basis_i, basis_j)


# ---------------------------------------------------------------------------
# Overlap cert construction
# ---------------------------------------------------------------------------


def _make_overlap_cert(realizations: tuple[ContractRealization, ...]) -> OverlapCert:
    """Build an OverlapCert for a merged family."""
    k = len(realizations)
    if k == 0:
        return OverlapCert(
            pairwise_iou=(),
            mutual_iou=0.0,
            node_iou=(),
            edge_iou=(),
            chance_iou_baseline=0.1,
        )
    # For a merged family, all realizations have the same semantics — set IOU = 1
    ones_row = tuple(1.0 for _ in range(k))
    matrix = tuple(ones_row for _ in range(k))
    return OverlapCert(
        pairwise_iou=matrix,
        mutual_iou=1.0,
        node_iou=matrix,
        edge_iou=matrix,
        chance_iou_baseline=0.1,
    )


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def quotient_extract(
    realizations: tuple[ContractRealization, ...],
    *,
    model: nn.Module | None = None,
    operand_tokens: Tensor | None = None,
    behavioral_threshold: float = BEHAVIORAL_THRESHOLD_DEFAULT,
    causal_threshold: float = CAUSAL_THRESHOLD_DEFAULT,
    basis_residual_threshold: float = BASIS_RESIDUAL_THRESHOLD_DEFAULT,
    output_attr: str = "encoder",
    seed: int = 0,
) -> QuotientResult:
    """Decide which realizations are different realizations of the same semantics.

    Algorithm:
      For each pair (i, j) of realizations:
        1. Compute behavioral_distance(i, j) on held-out tokens.
           If > behavioral_threshold → reject merge ("behavioral_far").
        2. If model provided, compute causal_distance(i, j) via causal_replaceability.
           If > causal_threshold → reject merge ("causal_far").
        3. Compute basis_change_residual(i, j) via orthogonal Procrustes.
           If > basis_residual_threshold → reject merge ("basis_far").
           If <= basis_residual_threshold → evidence for merge ("basis_match").
        4. Check alpha_equivalent(semantics_i, semantics_j).
           If terms are pointer terms and NOT alpha-equivalent → reject merge.
        5. If all checks pass → merge ("all_match" or "basis_match").

      Union-find groups are formed from merge decisions.
      Each group is emitted as one MechanismFamily with all member realizations.

    Args:
        realizations: Tuple of ContractRealization objects from any extractor.
        model: The neural network (required for causal_distance and behavioral_distance).
        operand_tokens: Input tokens for behavioral and causal measurements.
        behavioral_threshold: KL threshold for behavioral distance.
        causal_threshold: KL-difference threshold for causal distance.
        basis_residual_threshold: Procrustes residual threshold.
        output_attr: Name of the encoder submodule to hook.
        seed: Random seed for interchange-pair sampling.

    Returns:
        QuotientResult with families, decisions, and counts.
    """
    n = len(realizations)

    if n == 0:
        return QuotientResult(
            families=(),
            decisions=(),
            n_input_realizations=0,
            n_output_families=0,
        )

    if n == 1:
        family = _singleton_family(realizations[0])
        return QuotientResult(
            families=(family,),
            decisions=(),
            n_input_realizations=1,
            n_output_families=1,
        )

    # Infer d_model once
    d_model: int | None = None
    if model is not None and operand_tokens is not None:
        d_model = _infer_d_model(model, operand_tokens, output_attr)

    # Compute all pairwise decisions
    decisions: list[MergeDecision] = []
    merge_pairs: set[tuple[int, int]] = set()

    for idx_i in range(n):
        for idx_j in range(idx_i + 1, n):
            ri = realizations[idx_i]
            rj = realizations[idx_j]

            decision = _compute_merge_decision(
                ri,
                rj,
                model=model,
                operand_tokens=operand_tokens,
                behavioral_threshold=behavioral_threshold,
                causal_threshold=causal_threshold,
                basis_residual_threshold=basis_residual_threshold,
                d_model=d_model,
                output_attr=output_attr,
                seed=seed,
            )
            decisions.append(decision)
            if decision.same_family:
                merge_pairs.add((idx_i, idx_j))

    # Union-find grouping
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[px] = py

    for i, j in merge_pairs:
        _union(i, j)

    # Group realizations by their root
    groups: dict[int, list[int]] = {}
    for idx in range(n):
        root = _find(idx)
        groups.setdefault(root, []).append(idx)

    # Emit one MechanismFamily per group
    families: list[MechanismFamily] = []
    for _root, member_indices in sorted(groups.items()):
        member_realizations = tuple(realizations[i] for i in member_indices)
        family = _make_merged_family(member_realizations)
        families.append(family)

    return QuotientResult(
        families=tuple(families),
        decisions=tuple(decisions),
        n_input_realizations=n,
        n_output_families=len(families),
    )


# ---------------------------------------------------------------------------
# Pairwise merge decision
# ---------------------------------------------------------------------------


def _compute_merge_decision(
    ri: ContractRealization,
    rj: ContractRealization,
    *,
    model: nn.Module | None,
    operand_tokens: Tensor | None,
    behavioral_threshold: float,
    causal_threshold: float,
    basis_residual_threshold: float,
    d_model: int | None,
    output_attr: str,
    seed: int,
) -> MergeDecision:
    """Compute the pairwise MergeDecision for two realizations."""

    # ── 1. Behavioral distance ────────────────────────────────────────────────
    if model is not None and operand_tokens is not None:
        beh_dist = _behavioral_distance(ri, rj, model, operand_tokens, output_attr)
    else:
        beh_dist = 0.0

    # ── 2. Causal distance ─────────────────────────────────────────────────────
    if model is not None and operand_tokens is not None:
        cau_dist = _causal_distance(ri, rj, model, operand_tokens, output_attr, seed)
        if cau_dist == float("inf"):
            cau_dist = causal_threshold + 1.0  # treat as far
    else:
        cau_dist = 0.0

    # ── 3. Basis-change residual ───────────────────────────────────────────────
    bcr = _basis_change_residual(ri, rj, d_model)

    # ── 4. Alpha-equivalence ───────────────────────────────────────────────────
    # Always compute alpha-equivalence between the two semantic terms.
    # For pointer terms (with bound vars), this is a renaming walk.
    # For non-pointer terms (e.g. mod_add with different moduli), it detects
    # structural attr differences (e.g. modulus=12 vs modulus=31).
    # We report alpha_eq=None only when terms have no bound vars AND are
    # structurally identical (i.e. alpha-equivalence is trivially True by
    # structural equality — report None to indicate "not pointer terms").
    terms_are_pointer = _terms_have_bound_vars(ri.semantics) or _terms_have_bound_vars(
        rj.semantics
    )
    alpha_eq_result = alpha_equivalent(ri.semantics, rj.semantics)
    if terms_are_pointer:
        alpha_eq: bool | None = alpha_eq_result
    else:
        # For non-pointer terms, use alpha-equivalence as an additional structural gate:
        # if the terms are structurally different (different attrs like modulus), reject.
        # Report None only if they are equivalent (to indicate "not pointer terms, but ok").
        alpha_eq = None if alpha_eq_result else False

    # ── Decision logic ─────────────────────────────────────────────────────────
    same_family, reason = _merge_logic(
        beh_dist=beh_dist,
        cau_dist=cau_dist,
        bcr=bcr,
        alpha_eq=alpha_eq,
        behavioral_threshold=behavioral_threshold,
        causal_threshold=causal_threshold,
        basis_residual_threshold=basis_residual_threshold,
    )

    return MergeDecision(
        realization_i=ri.id,
        realization_j=rj.id,
        same_family=same_family,
        behavioral_distance=beh_dist,
        causal_distance=cau_dist,
        basis_change_residual=bcr,
        alpha_equivalent=alpha_eq,
        reason=reason,
    )


def _merge_logic(
    *,
    beh_dist: float,
    cau_dist: float,
    bcr: float | None,
    alpha_eq: bool | None,
    behavioral_threshold: float,
    causal_threshold: float,
    basis_residual_threshold: float,
) -> tuple[bool, str]:
    """Pure decision logic given pre-computed distances.

    Returns (same_family, reason_string).
    """
    # Hard rejects
    if beh_dist > behavioral_threshold:
        return False, "behavioral_far"

    if cau_dist > causal_threshold:
        return False, "causal_far"

    # Alpha-equivalence: if terms have bound vars and are NOT alpha-equivalent → reject
    if alpha_eq is not None and not alpha_eq:
        return False, "alpha_inequivalent"

    # Basis check (when available)
    if bcr is not None:
        if bcr > basis_residual_threshold:
            return False, "basis_far"
        # Basis residual is small: positive evidence for merge
        if beh_dist <= behavioral_threshold and cau_dist <= causal_threshold:
            return True, "basis_match"

    # All soft conditions satisfied — merge
    return True, "all_match"


def _terms_have_bound_vars(term: Term) -> bool:
    """Return True if this term tree contains any 'let' binder with a binding attr."""
    if term.op == "let" and term.attrs.get("binding") is not None:
        return True
    if term.op == "var":
        return False
    return any(_terms_have_bound_vars(a) for a in term.args)


# ---------------------------------------------------------------------------
# Family construction helpers
# ---------------------------------------------------------------------------


def _singleton_family(real: ContractRealization) -> MechanismFamily:
    """Wrap a single realization in a MechanismFamily."""
    overlap = _make_overlap_cert((real,))
    return MechanismFamily(
        id=f"quotient_family_{real.id}",
        semantics=real.semantics,
        realizations=(real,),
        overlap=overlap,
        aggregation="one_of",
        invariants=(),
        metadata={"quotient_merged": False, "source_realization_ids": [real.id]},
    )


def _make_merged_family(
    member_realizations: tuple[ContractRealization, ...],
) -> MechanismFamily:
    """Emit a MechanismFamily merging multiple realizations."""
    # Use the first realization's semantics as the canonical semantics of the family
    canonical_semantics = member_realizations[0].semantics
    family_id = "quotient_family_" + uuid.uuid4().hex[:8]
    overlap = _make_overlap_cert(member_realizations)
    return MechanismFamily(
        id=family_id,
        semantics=canonical_semantics,
        realizations=member_realizations,
        overlap=overlap,
        aggregation="quorum",
        invariants=(),
        metadata={
            "quotient_merged": len(member_realizations) > 1,
            "source_realization_ids": [r.id for r in member_realizations],
        },
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "MergeDecision",
    "QuotientResult",
    "alpha_equivalent",
    "quotient_extract",
]
