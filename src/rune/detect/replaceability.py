"""Lane 1.E — Causal Replaceability gatekeeper for detector claims.

Every other Phase-1 detector emits candidate ``Realization`` objects asserting
that a particular subspace ``S`` of the encoder's hidden state realises some
algebraic structure ``X`` (cyclic group, helix, lookup, etc.).  This module is
the gatekeeper that decides whether to accept those claims.

A candidate is accepted iff **all three** of the following KLs, each computed
from real model forward passes with a patched hidden state, fall below a named
threshold and the fitted manifold is not catastrophically less faithful than
the raw activation:

1. ``raw_patch_kl`` — **interchange-intervention KL**.  For each anchor sample
   ``x``, pick a counterfactual ``x'`` whose model prediction differs from
   ``x``'s, transplant ``S^T h_{x'}`` into ``h_x`` along the subspace, run a
   real forward, and measure ``KL(M(x') || M_patched(x))``.  If ``S`` truly
   *transports* the symbolic variable, the patched forward at ``x`` should
   behave like ``M(x')``: low KL.  If ``S`` is a random direction, patching
   does not move behaviour toward ``x'``: high KL.  This is the canonical
   Geiger-style interchange-intervention measurement (Geiger et al. 2021,
   arXiv:2106.02193).

2. ``fitted_manifold_kl`` — **reconstruction-preservation KL**.  For each
   anchor ``x``, replace ``S^T h_x`` with ``S^T decode(h_x)`` (the symbolic
   reconstruction), run a real forward, and measure
   ``KL(M(x) || M_patched(x))``.  The reconstruction is *patched into the
   model*, not measured against the raw activation by L2 norm — see
   ``_compute_fitted_manifold_kl``.  This closes the round-2 cheat where a
   PCA-style reconstruction looked faithful in L2 but did not preserve
   behaviour when actually substituted into the forward pass.

3. ``sibling_ablation_kl`` — **alternative-route replaceability**.  Mean-
   substitute every *other* realization's subspace from ``h_x`` (zeroing the
   per-sample information in those directions) AND transplant ``S^T h_{x'}``
   into the realization's subspace, then run a real forward and measure
   ``KL(M(x') || M_patched(x))``.  A genuine alternative-route realization
   preserves the interchange behaviour under sibling ablation; an
   epiphenomenal one does not.

The gatekeeper is strictly black-box:

  - only ``register_forward_hook`` (output-only) on ``model.<output_attr>``;
    never ``register_forward_pre_hook`` and never reading from a hook's
    ``inputs[0]`` argument;
  - no model config introspection — forbidden patterns listed in the audit
    test ``tests/detection/test_replaceability_anti_cheat_audit.py``;
  - no parameter, buffer, or embedding-weight reads;
  - all numeric thresholds are module-level named constants documented below.

Calibration notes for each named constant are inline at its definition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Named module-level constants
# ---------------------------------------------------------------------------

# Default KL ceiling (in nats) above which a realization is rejected.
# Calibration:
#   - On a 100%-accurate modadd-7 model, a Lane-1.B-discovered cyclic-7
#     subspace produces interchange KL < 1e-3 nats (empirically ~1e-7 on the
#     dev box used to wire this module).  A random orthogonal subspace of
#     the same dimensionality produces interchange KL > 10 nats.  0.01 is a
#     conservative pass threshold that comfortably separates the two.
#   - Overridable by callers via the ``kl_reject`` keyword.
_DEFAULT_KL_REJECT_NATS = 0.01

# Maximum ratio ``fitted_manifold_kl / max(raw_patch_kl, floor)``.  Calibration:
#   - PLAN.md kill criterion: "fitted-manifold patching falls below raw-patch
#     baseline by >2×, the symbolic object is epiphenomenal."  We
#     operationalise "by >2×" as fitted_vs_raw > 2.0.
#   - A constant-zero decode adversary produces fitted_manifold_kl on the
#     order of the model's mean-zero-h KL, which is >> 2× the interchange
#     baseline on any non-degenerate realization, so the ratio gate rejects
#     it.
_DEFAULT_FITTED_VS_RAW_RATIO_MAX = 2.0

# Numerical floor for the denominator of ``fitted_vs_raw_ratio`` to avoid
# 0/0.  Purely numerical; never a hand-tuned multiplier applied to a KL.
_RATIO_DENOMINATOR_FLOOR = 1.0e-12

# Maximum number of (anchor, counterfactual) pairs evaluated per realization.
# Calibration: 256 pairs keep the standard error of the mean KL below 5% of
# the mean on modadd-7; larger budgets are accepted but slower.  Smaller
# datasets (e.g. modadd-7 has 49 unique pairs) take whatever is available.
_DEFAULT_MAX_COUNTERFACTUAL_PAIRS = 256

# Minimum number of (anchor, counterfactual) pairs required before we report
# a finite KL.  If the model produces fewer than this many distinct
# predicted-class buckets (so no inter-class pair exists), we return
# +inf for the affected KL — the realization is rejected at the gate.
_MIN_VALID_PAIRS = 8

# Tolerance for the orthonormality check on a supplied ``subspace_basis``.
# A basis whose Gram matrix differs from I by more than this is silently
# re-orthonormalised by QR.  This is a basis-cleaning step, not a gate.
_ORTHONORMALITY_TOL = 1.0e-4


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Realization:
    """One detector's candidate claim about a subspace of the model's hidden state.

    Fields:
      - ``id``                — opaque identifier, must be unique within a
                                 ``causal_replaceability`` call; threaded
                                 through each ``ReplaceabilityCertificate``.
      - ``subspace_basis``    — Tensor of shape ``(d, k)`` whose columns span
                                 the candidate subspace.  Columns need not be
                                 orthonormal; the gatekeeper re-orthonormalises
                                 by QR before use.  ``d`` must equal the
                                 model's hidden_state dimension at the
                                 captured hook point.
      - ``decode``            — callable ``f(h) -> h_replacement`` where
                                 ``h`` is the encoder's output tensor as
                                 captured by the hook (e.g. ``(batch, seq, d)``)
                                 and the returned tensor has the *same shape*.
                                 ``decode`` represents the candidate's
                                 symbolic decode → ``C · B(â)`` reconstruction.
                                 A constant-zero decode is an adversary that
                                 must be rejected by the fitted-vs-raw ratio.
                                 The fitted-manifold path patches
                                 ``decode(h)`` into the model via a real
                                 forward, so a reconstruction-only "cheat"
                                 (high L2 fidelity, low behaviour fidelity)
                                 is exposed by this gate.
    """

    id: str
    subspace_basis: Tensor
    decode: Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ReplaceabilityCertificate:
    """Per-realization gatekeeper result.

    The dataclass is ``frozen=True`` so callers cannot post-hoc rewrite a
    failing certificate into a passing one.

    Field semantics (each KL is in nats):

    - ``raw_patch_kl`` — KL(M(x') || M_patched(x)), where M_patched is the
      model with ``S^T h_x`` replaced by ``S^T h_{x'}`` via a real forward
      pass.  ``x'`` is a counterfactual whose model prediction differs from
      ``x``'s.  Low value means the subspace transports the realised
      variable.  See ``_compute_raw_patch_kl``.

    - ``fitted_manifold_kl`` — KL(M(x) || M_patched(x)), where M_patched is
      the model with ``S^T h_x`` replaced by ``S^T decode(h_x)`` via a real
      forward pass.  Low value means the symbolic reconstruction preserves
      behaviour when substituted back in.  See ``_compute_fitted_manifold_kl``;
      every KL on this field is computed by an actual ``model(...)`` call.

    - ``sibling_ablation_kl`` — KL(M(x') || M_patched(x)) computed as in
      ``raw_patch_kl`` but additionally mean-substituting every other
      realization's subspace before the forward pass.  Tests alternative-
      route replaceability.  See ``_compute_sibling_ablation_kl``.

    - ``fitted_vs_raw_ratio`` — ``fitted_manifold_kl / max(raw_patch_kl,
      _RATIO_DENOMINATOR_FLOOR)``.  Gate against epiphenomenal symbolic
      objects (PLAN.md kill criterion).

    - ``passes_threshold`` — every KL <= ``kl_reject`` AND
      ``fitted_vs_raw_ratio <= fitted_vs_raw_ratio_max``.
    """

    realization_id: str
    raw_patch_kl: float
    fitted_manifold_kl: float
    sibling_ablation_kl: float
    fitted_vs_raw_ratio: float
    passes_threshold: bool


@dataclass(frozen=True)
class ReplaceabilityResult:
    accepted: tuple[ReplaceabilityCertificate, ...]
    rejected: tuple[ReplaceabilityCertificate, ...]
    threshold_kl: float
    threshold_fitted_vs_raw_ratio: float


# ---------------------------------------------------------------------------
# Hook utilities (output-only; never register_forward_pre_hook; never read
# inputs[0] inside a hook).
# ---------------------------------------------------------------------------


def _get_hook_target(model: nn.Module, output_attr: str) -> nn.Module:
    """Resolve ``model.<output_attr>``; raise if it is missing or non-Module."""
    if not hasattr(model, output_attr):
        raise AttributeError(
            f"model has no attribute {output_attr!r} to register an output hook on"
        )
    target = getattr(model, output_attr)
    if not isinstance(target, nn.Module):
        raise TypeError(
            f"model.{output_attr} is not an nn.Module (got {type(target).__name__}); "
            f"output hooks can only be registered on nn.Module instances"
        )
    return target


def _capture_encoder_outputs(
    model: nn.Module,
    hook_target: nn.Module,
    operand_tokens: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run ``model(operand_tokens)`` once, capturing encoder output and final logits.

    Uses ``register_forward_hook`` (output-only) on ``hook_target``.  Reads
    only the ``output`` argument; the ``inputs`` argument is deliberately
    ignored — see the anti-cheat audit test.
    """
    captured_output: list[Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple, output: object) -> None:
        # Output-only capture.  We deliberately ignore the _inputs argument
        # so that the hook cannot accidentally side-channel the planted
        # embedding rows (the round-2 cheat 2c failure mode).
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        if isinstance(tensor, Tensor):
            captured_output.append(tensor.detach())

    handle = hook_target.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            logits = model(operand_tokens).detach()
    finally:
        handle.remove()

    if not captured_output:
        raise RuntimeError(
            "encoder output hook captured nothing; check that model.<output_attr> "
            "is invoked during forward(operand_tokens)"
        )
    return captured_output[-1].float(), logits.float()


def _run_model_with_patched_encoder(
    model: nn.Module,
    hook_target: nn.Module,
    operand_tokens: Tensor,
    replacement: Tensor,
) -> Tensor:
    """Run ``model(operand_tokens)`` while overriding the encoder's output
    with ``replacement`` for the duration of that forward call.

    Implementation: register an output hook that *returns* ``replacement``,
    overriding the natural encoder output.  This is the canonical PyTorch
    pattern for activation patching and is the load-bearing path that lets
    every KL field in this module be derived from a real forward.

    Note: this function calls ``model(operand_tokens)``.  The audit test
    greps for this call to confirm that ``fitted_manifold_kl`` is computed
    by a real forward, not by a reconstruction norm.
    """
    if replacement.shape[0] != operand_tokens.shape[0]:
        raise ValueError(
            "replacement batch dimension does not match operand_tokens batch"
        )
    replacement_buf = replacement.detach()

    def _hook(_module: nn.Module, _inputs: tuple, output: object) -> Tensor:
        # We do not look at _output; we override it.
        # _inputs is also ignored — see hook discipline above.
        return replacement_buf

    handle = hook_target.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            patched_logits = model(operand_tokens).detach()
    finally:
        handle.remove()
    return patched_logits.float()


# ---------------------------------------------------------------------------
# Subspace projection utilities
# ---------------------------------------------------------------------------


def _orthonormalise_basis(basis: Tensor) -> Tensor:
    """Return an orthonormal basis spanning the same column space as ``basis``.

    Uses QR decomposition.  If the input is already orthonormal within
    ``_ORTHONORMALITY_TOL`` it is returned unchanged.
    """
    if basis.ndim != 2:
        raise ValueError(f"subspace_basis must be 2-D (d, k); got shape {tuple(basis.shape)}")
    if basis.shape[1] == 0:
        return basis
    # Quick check: B^T B should be ~I for orthonormal columns.
    gram = basis.T @ basis
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    if (gram - eye).abs().max().item() <= _ORTHONORMALITY_TOL:
        return basis
    q, _ = torch.linalg.qr(basis.float(), mode="reduced")
    return q


def _project_into_subspace(hidden: Tensor, basis: Tensor) -> Tensor:
    """Return ``S S^T h``: the component of ``hidden`` inside the subspace.

    ``hidden`` has shape ``(..., d)``; ``basis`` has shape ``(d, k)`` with
    orthonormal columns.  The projection acts on the last dimension and
    broadcasts across all leading dimensions.
    """
    coeffs = hidden @ basis  # (..., k)
    return coeffs @ basis.T  # (..., d)


def _patch_subspace(
    hidden: Tensor,
    basis: Tensor,
    replacement_full: Tensor,
) -> Tensor:
    """Replace the subspace component of ``hidden`` with the subspace
    component of ``replacement_full``.

    Result: ``hidden - S S^T hidden + S S^T replacement_full``.

    Shapes: ``hidden`` and ``replacement_full`` both ``(..., d)``;
            ``basis`` ``(d, k)`` orthonormal.
    """
    if hidden.shape != replacement_full.shape:
        raise ValueError(
            f"hidden {tuple(hidden.shape)} and replacement_full "
            f"{tuple(replacement_full.shape)} must have the same shape"
        )
    return hidden + _project_into_subspace(replacement_full - hidden, basis)


def _ablate_subspaces(
    hidden: Tensor,
    sibling_bases: tuple[Tensor, ...],
    mean_hidden: Tensor,
) -> Tensor:
    """Mean-substitute every sibling subspace from ``hidden``.

    For each ``basis_j`` the component of ``hidden`` along it is replaced
    with the corresponding component of ``mean_hidden``.  Mean substitution
    avoids catastrophic norm collapse while still erasing per-sample
    information in the sibling subspace.

    This is *not* a residual-norm computation; the returned tensor is fed
    through a real model forward in the caller (see
    ``_compute_sibling_ablation_kl``).
    """
    result = hidden
    mean_expanded = mean_hidden.expand_as(hidden)
    for basis_j in sibling_bases:
        result = _patch_subspace(result, basis_j, mean_expanded)
    return result


# ---------------------------------------------------------------------------
# Counterfactual pairing — different predicted argmax class (interchange)
# ---------------------------------------------------------------------------


def _build_interchange_pairs(
    logits: Tensor,
    *,
    max_pairs: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """For each anchor index i, find a counterfactual index j with a
    *different* predicted argmax class.

    Returns ``(anchor_idx, counterfactual_idx)`` tensors of equal length,
    capped at ``max_pairs``.

    Operational meaning: under interchange intervention, transplanting the
    realization's subspace from the counterfactual into the anchor should
    transport the counterfactual's behaviour into the anchor.  Comparing
    against ``M(x')`` therefore tests whether the subspace really carries
    that behaviour.
    """
    preds = logits.argmax(dim=-1).cpu().long()
    n = int(preds.shape[0])
    if n < 2:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # Index of "any sample with predicted class != c" per class c.
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(int(preds[i].item()), []).append(i)
    other_class_indices: dict[int, list[int]] = {}
    for c in buckets:
        others: list[int] = []
        for c2, idxs in buckets.items():
            if c2 == c:
                continue
            others.extend(idxs)
        other_class_indices[c] = others

    generator = torch.Generator().manual_seed(seed)
    anchors: list[int] = []
    counterfactuals: list[int] = []
    permuted = torch.randperm(n, generator=generator).tolist()
    for i in permuted:
        c_i = int(preds[i].item())
        candidates = other_class_indices.get(c_i, [])
        if not candidates:
            continue
        offset = int(torch.randint(0, len(candidates), (1,), generator=generator).item())
        j = candidates[offset]
        if j == i:
            continue
        anchors.append(i)
        counterfactuals.append(j)
        if len(anchors) >= max_pairs:
            break

    return (
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(counterfactuals, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# KL between two logit batches
# ---------------------------------------------------------------------------


def _kl_mean_nats(p_logits: Tensor, q_logits: Tensor) -> float:
    """Return mean over batch of KL(softmax(p_logits) || softmax(q_logits))
    in nats.

    Uses ``torch.log_softmax`` for numerical stability.  KL is asymmetric:
    we follow the contract notation ``KL(reference || patched)``, so
    ``p_logits`` is the reference and ``q_logits`` is the patched output.
    """
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    kl_per_sample = (p * (log_p - log_q)).sum(dim=-1)
    # KL is non-negative analytically; numerical noise may produce tiny
    # negatives, which we clamp to 0 before averaging.
    kl_per_sample = kl_per_sample.clamp_min(0.0)
    return float(kl_per_sample.mean().item())


# ---------------------------------------------------------------------------
# Core: compute the three KLs for one realization
# ---------------------------------------------------------------------------


def _compute_raw_patch_kl(
    *,
    model: nn.Module,
    hook_target: nn.Module,
    operand_tokens: Tensor,
    hidden_all: Tensor,
    clean_logits_all: Tensor,
    anchor_idx: Tensor,
    counterfactual_idx: Tensor,
    basis: Tensor,
) -> float:
    """Interchange-intervention KL for the raw-activation patch.

    Algorithm:
      1. Build patched_hidden[i] = h[a_i] with its S-component replaced by
         h[cf_i]'s S-component.
      2. Run ``model(operand_tokens[anchor_idx])`` with the encoder output
         replaced by patched_hidden (via output hook).
      3. KL(M(x_cf) || M_patched(x_anchor)) averaged over pairs.

    A subspace that transports the realised variable produces a patched
    forward that behaves like the counterfactual: low KL.  A random
    subspace fails to transport: high KL.
    """
    if anchor_idx.numel() < _MIN_VALID_PAIRS:
        return float("inf")
    anchor_tokens = operand_tokens[anchor_idx]
    anchor_hidden = hidden_all[anchor_idx]
    cf_hidden = hidden_all[counterfactual_idx]
    patched_hidden = _patch_subspace(anchor_hidden, basis, cf_hidden)
    # Real forward — see _run_model_with_patched_encoder for the model(...) call.
    patched_logits = _run_model_with_patched_encoder(
        model, hook_target, anchor_tokens, patched_hidden
    )
    # Reference = model(x_cf), so the test is "does the patch import x_cf
    # behaviour into x_anchor".
    cf_logits = clean_logits_all[counterfactual_idx]
    return _kl_mean_nats(cf_logits, patched_logits)


def _compute_fitted_manifold_kl(
    *,
    model: nn.Module,
    hook_target: nn.Module,
    operand_tokens: Tensor,
    hidden_all: Tensor,
    clean_logits_all: Tensor,
    anchor_idx: Tensor,
    basis: Tensor,
    decode: Callable[[Tensor], Tensor],
) -> float:
    """Reconstruction-preservation KL via real forward.

    Algorithm:
      1. Build patched_hidden[i] = h[a_i] with its S-component replaced by
         the S-component of decode(h[a_i]) (the symbolic reconstruction).
      2. Run ``model(operand_tokens[anchor_idx])`` with the encoder output
         replaced by patched_hidden.
      3. KL(M(x_anchor) || M_patched(x_anchor)) averaged over anchors.

    Closes the round-2 cheat: a decode that yields an L2-faithful
    reconstruction but does not preserve behaviour will produce a large KL
    on this path because the decoded value is *substituted into the model*
    (real forward), not compared against the raw h by L2 norm.
    """
    if anchor_idx.numel() < _MIN_VALID_PAIRS:
        return float("inf")
    anchor_tokens = operand_tokens[anchor_idx]
    anchor_hidden = hidden_all[anchor_idx]
    decoded = decode(anchor_hidden)
    if not isinstance(decoded, Tensor):
        raise TypeError(
            f"Realization.decode must return a Tensor; got {type(decoded).__name__}"
        )
    if decoded.shape != anchor_hidden.shape:
        raise ValueError(
            f"Realization.decode returned shape {tuple(decoded.shape)}; "
            f"expected {tuple(anchor_hidden.shape)}"
        )
    decoded = decoded.detach().float()
    patched_hidden = _patch_subspace(anchor_hidden, basis, decoded)
    # Real forward — see _run_model_with_patched_encoder for the model(...) call.
    patched_logits = _run_model_with_patched_encoder(
        model, hook_target, anchor_tokens, patched_hidden
    )
    anchor_clean = clean_logits_all[anchor_idx]
    return _kl_mean_nats(anchor_clean, patched_logits)


def _compute_sibling_ablation_kl(
    *,
    model: nn.Module,
    hook_target: nn.Module,
    operand_tokens: Tensor,
    hidden_all: Tensor,
    clean_logits_all: Tensor,
    anchor_idx: Tensor,
    counterfactual_idx: Tensor,
    basis: Tensor,
    sibling_bases: tuple[Tensor, ...],
) -> float:
    """Interchange KL with sibling subspaces mean-substituted.

    Algorithm:
      1. Patch h[a_i]'s S with h[cf_i]'s S (interchange).
      2. Mean-substitute every sibling subspace's component in the patched
         hidden state (alternative-route ablation).
      3. Run ``model(operand_tokens[anchor_idx])`` with the encoder output
         replaced by the doubly-modified hidden.
      4. KL(M(x_cf) || M_patched(x_anchor)) averaged over pairs.

    A genuine alternative-route realization preserves the interchange
    behaviour even when its siblings are erased.  An epiphenomenal
    realization (only "passing" because a sibling was carrying the work)
    will fail the gate after sibling ablation.
    """
    if anchor_idx.numel() < _MIN_VALID_PAIRS:
        return float("inf")
    anchor_tokens = operand_tokens[anchor_idx]
    anchor_hidden = hidden_all[anchor_idx]
    cf_hidden = hidden_all[counterfactual_idx]
    patched_hidden = _patch_subspace(anchor_hidden, basis, cf_hidden)

    if sibling_bases:
        # Per-position mean across the full dataset (shape (1, ..., d)),
        # then broadcast to the anchor batch.  This is the standard
        # ``mean`` ablation in the ActivationCache manifest (Lane 0.2).
        mean_hidden = hidden_all.mean(dim=0, keepdim=True)
        patched_hidden = _ablate_subspaces(
            patched_hidden, sibling_bases, mean_hidden
        )

    # Real forward — see _run_model_with_patched_encoder for the model(...) call.
    patched_logits = _run_model_with_patched_encoder(
        model, hook_target, anchor_tokens, patched_hidden
    )
    cf_logits = clean_logits_all[counterfactual_idx]
    return _kl_mean_nats(cf_logits, patched_logits)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def causal_replaceability(
    model: nn.Module,
    operand_tokens: Tensor,
    realizations: tuple[Realization, ...],
    *,
    output_attr: str = "encoder",
    kl_reject: float = _DEFAULT_KL_REJECT_NATS,
    fitted_vs_raw_ratio_max: float = _DEFAULT_FITTED_VS_RAW_RATIO_MAX,
    seed: int = 0,
) -> ReplaceabilityResult:
    """Gate each realization through three real-forward KL measurements.

    A realization is accepted iff:
        raw_patch_kl       <= kl_reject
        fitted_manifold_kl <= kl_reject
        sibling_ablation_kl<= kl_reject
        fitted_vs_raw_ratio<= fitted_vs_raw_ratio_max

    ``output_attr`` names the ``nn.Module`` attribute on ``model`` whose
    output is the hidden state under test (the canonical Phase-1 hook
    target).  Only ``register_forward_hook`` (output-only) is used; this
    module never registers ``register_forward_pre_hook`` and never reads
    ``inputs[0]`` inside a hook.

    See the module docstring for full semantics and the anti-cheat audit
    list.
    """
    if operand_tokens.ndim < 2:
        raise ValueError("operand_tokens must be at least 2-D (batch, ...)")
    if not realizations:
        return ReplaceabilityResult(
            accepted=(),
            rejected=(),
            threshold_kl=kl_reject,
            threshold_fitted_vs_raw_ratio=fitted_vs_raw_ratio_max,
        )

    seen_ids: set[str] = set()
    for r in realizations:
        if r.id in seen_ids:
            raise ValueError(f"duplicate Realization id {r.id!r}")
        seen_ids.add(r.id)

    model.eval()
    hook_target = _get_hook_target(model, output_attr)

    operand_tokens = operand_tokens.detach()
    hidden_all, clean_logits_all = _capture_encoder_outputs(
        model, hook_target, operand_tokens
    )
    d_model = int(hidden_all.shape[-1])

    # Orthonormalise each basis once and validate the leading dim.
    orthonormal_bases: dict[str, Tensor] = {}
    for r in realizations:
        if r.subspace_basis.ndim != 2 or r.subspace_basis.shape[0] != d_model:
            raise ValueError(
                f"Realization {r.id!r}: subspace_basis must be (d, k) with "
                f"d == hidden d_model ({d_model}); got "
                f"{tuple(r.subspace_basis.shape)}"
            )
        orthonormal_bases[r.id] = _orthonormalise_basis(
            r.subspace_basis.detach().float()
        )

    # Build interchange pair index once per call so all realizations are
    # scored on the same anchors / counterfactuals.
    anchor_idx, counterfactual_idx = _build_interchange_pairs(
        clean_logits_all,
        max_pairs=_DEFAULT_MAX_COUNTERFACTUAL_PAIRS,
        seed=seed,
    )

    accepted: list[ReplaceabilityCertificate] = []
    rejected: list[ReplaceabilityCertificate] = []

    for r in realizations:
        basis = orthonormal_bases[r.id]
        # Siblings: every OTHER realization's basis.  Pre-computed so the
        # sibling-ablation pass zeroes the orthogonal-complement
        # "alternative routes" while only this realization's subspace is
        # patched.
        sibling_bases = tuple(
            orthonormal_bases[other.id]
            for other in realizations
            if other.id != r.id
        )

        raw_kl = _compute_raw_patch_kl(
            model=model,
            hook_target=hook_target,
            operand_tokens=operand_tokens,
            hidden_all=hidden_all,
            clean_logits_all=clean_logits_all,
            anchor_idx=anchor_idx,
            counterfactual_idx=counterfactual_idx,
            basis=basis,
        )
        fitted_kl = _compute_fitted_manifold_kl(
            model=model,
            hook_target=hook_target,
            operand_tokens=operand_tokens,
            hidden_all=hidden_all,
            clean_logits_all=clean_logits_all,
            anchor_idx=anchor_idx,
            basis=basis,
            decode=r.decode,
        )
        sibling_kl = _compute_sibling_ablation_kl(
            model=model,
            hook_target=hook_target,
            operand_tokens=operand_tokens,
            hidden_all=hidden_all,
            clean_logits_all=clean_logits_all,
            anchor_idx=anchor_idx,
            counterfactual_idx=counterfactual_idx,
            basis=basis,
            sibling_bases=sibling_bases,
        )

        denominator = max(raw_kl, _RATIO_DENOMINATOR_FLOOR)
        ratio = float(fitted_kl / denominator)

        passes = (
            raw_kl <= kl_reject
            and fitted_kl <= kl_reject
            and sibling_kl <= kl_reject
            and ratio <= fitted_vs_raw_ratio_max
        )
        cert = ReplaceabilityCertificate(
            realization_id=r.id,
            raw_patch_kl=raw_kl,
            fitted_manifold_kl=fitted_kl,
            sibling_ablation_kl=sibling_kl,
            fitted_vs_raw_ratio=ratio,
            passes_threshold=passes,
        )
        if passes:
            accepted.append(cert)
        else:
            rejected.append(cert)

    return ReplaceabilityResult(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        threshold_kl=kl_reject,
        threshold_fitted_vs_raw_ratio=fitted_vs_raw_ratio_max,
    )


__all__ = [
    "Realization",
    "ReplaceabilityCertificate",
    "ReplaceabilityResult",
    "causal_replaceability",
]
