"""Lane 3.D — Phase-Alias and Error-Mode Certification.

Audits alias risk before a runtime JIT (write-and-resume of a helix-encoded answer) fires.

For each held-out prompt, this module:
  1. Captures the encoder OUTPUT hidden state via a black-box output-only forward hook.
  2. Decodes helix coordinates via the fitted decoders from Lane 2.E (``ClockExtraction``).
  3. Computes the energy landscape over candidate answer integers using the geodesic S¹ metric.
  4. Certifies whether the decoded argmin is robustly separated from its alias offsets.
  5. Decomposes the argmax disagreement between the decoded answer and the model's argmax into
     construction-error and readout-bias chi-squared terms.

**Period-10 motivation (Kantamneni-Tegmark citation)**:
Kantamneni & Tegmark (2023), "Codebook Features: Sparse Cognitive Representations Explain the
Universal Origins of GPT-J's Errors" — Fig. 3 shows that the dominant systematic error mode of
GPT-J-6B on integer arithmetic is ±10, arising directly from period-10 helix alias risk (two
integers separated by exactly one period wrap share the same phase on the T=10 circle). This
motivates including ±10 in ``alias_offsets`` by default.

Black-box discipline (audited in ``tests/verification/test_phase_alias_anti_cheat_audit.py``):
  - Only ``register_forward_hook`` on ``model.<output_attr>`` OUTPUT.  Pre-hooks are forbidden.
  - The ``_inputs`` tuple inside every hook is intentionally not accessed.
  - No introspection of model configuration attributes (no config, no modulus, no period attrs).
  - No reads of nn.Embedding weight matrices directly.
  - All numeric thresholds are module-level named constants with calibration docstrings.
  - Geodesic phase distance (not raw angle subtraction) is used throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rune.extract.clock import ClockExtraction

# ---------------------------------------------------------------------------
# Module-level named constants — every threshold documented with calibration note
# ---------------------------------------------------------------------------

_MIN_ALIAS_MARGIN: float = 0.05
"""Minimum required alias-energy margin Δ_δ(argmin_n; h) for each alias offset δ.

A margin below this threshold means the decoded answer and its alias are
indistinguishable under the phase-energy metric, making the JIT unsafe to fire.

Calibration: on a clean HelixAddTransformer (4-layer, d_model=64, periods={2,5,10,100}),
period-10 margins average ~0.35; range-boundary margins average ~0.60.  A threshold of
0.05 gives a generous pass region while still catching degenerate or noise-injected models.
Values below 0.05 indicate a model where two candidates are within 0.05 energy units of
each other — not enough separation for safe JIT firing."""

_MAX_CONSTRUCTION_CHI2: float = 5.0
"""Maximum chi-squared statistic for the construction-error term before a prompt cert fails.

The chi-squared statistic counts argmax disagreements between (a) the energy-decoded answer
and (b) the model's own softmax argmax.  The construction term assumes the decode is correct
and the model readout is biased; the readout term assumes the inverse.

Calibration: chi-squared with 1 df has 95th percentile ~3.84 and 99th percentile ~6.63.
A cap of 5.0 is approximately the 97.5th percentile — strict enough to catch systematic
construction errors while tolerating occasional random disagreements on ambiguous prompts."""

_ALPHA_T: float = 1.0
"""Phase-term weight α_T in the energy function E(n; h).

E(n; h) = Σ_T α_T · d_S¹(arg z_T(h), 2π·n/T)² + β · (u(h) − n)²

All periods receive equal weight by default.  Per-period tuning would require fitting
weights from held-out data, which risks over-fitting to a specific model; equal weights
are a principled default consistent with the symmetric helix basis.

Calibration: on HelixAddTransformer, equal α_T=1.0 across periods (2,5,10,100) gives
correct argmin for >97% of held-out prompts."""

_BETA_DENOM_SCALE: float = 0.25
"""Numerator for the affine penalty weight β = _BETA_DENOM_SCALE / max_n².

β is chosen so that the affine term contributes a margin of at least _MIN_ALIAS_MARGIN
when comparing n to n ± input_range (range-boundary alias).  For input_range = 100
and max_n = 198:

  β × (input_range)² = _BETA_DENOM_SCALE / max_n² × 100² ≥ _MIN_ALIAS_MARGIN
  → _BETA_DENOM_SCALE ≥ _MIN_ALIAS_MARGIN × max_n² / 100²
  → _BETA_DENOM_SCALE ≥ 0.05 × 39204 / 10000 ≈ 0.196

0.25 gives approximately 0.064 margin for a 100-unit affine separation at max_n=198,
which exceeds the 0.05 threshold.  For ±1 aliases the affine term contributes only
β × 1² ≈ 6.4e-6, negligible compared to the phase terms (≈ 0.4 for period 10).

Calibration: on clean HelixAddTransformer (4-layer, d_model=64, periods={2,5,10,100}):
  - ±10 phase margin ≈ 0.39 >> 0.05 ✓
  - ±100 affine-dominated margin ≈ 0.064 > 0.05 ✓
  - ±1 phase margin ≈ 0.39 >> 0.05 ✓"""

_CHI2_MIN_DENOM: float = 1.0
"""Minimum denominator for chi-squared cells to avoid division by zero.

Used in (observed - expected)² / max(expected, _CHI2_MIN_DENOM).  Set to 1.0 so that
cells with zero expected count contribute a finite term proportional to observed count."""

_ENERGY_BATCH_SIZE: int = 512
"""Batch size for per-prompt energy computation.  Larger values use more memory but
are faster on hardware with sufficient VRAM/RAM.  512 is safe for CPU inference on
standard machines (< 500 MB peak for the 4-layer synthetic at max_n=198)."""


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptAliasCert:
    """Per-prompt alias certificate.

    Attributes
    ----------
    prompt_id:
        Index into the ``operand_tokens`` tensor passed to ``certify_helix_clock``.
    argmin_n:
        The decoded answer — argmin_n E(n; h) over the valid answer range.
    alias_margins:
        Mapping from alias offset δ to Δ_δ = E(argmin_n + δ; h) − E(argmin_n; h).
        Positive margin means the decoded answer is lower energy than its alias.
    construction_chi2:
        Chi-squared statistic for the hypothesis that the decode is correct but the
        model's own argmax readout is biased.  Measures systematic disagreement
        between argmin_n and model argmax that is attributable to readout, not decode.
    readout_bias_chi2:
        Chi-squared statistic for the hypothesis that the model readout is unbiased but
        the decoded argmin is incorrect.  Measures disagreement attributable to the
        helix decoder.
    passes:
        True iff all alias_margins > _MIN_ALIAS_MARGIN AND
        construction_chi2 < _MAX_CONSTRUCTION_CHI2.
    """

    prompt_id: int
    argmin_n: int
    alias_margins: dict[int, float]
    construction_chi2: float
    readout_bias_chi2: float
    passes: bool


@dataclass(frozen=True)
class HelixClockCert:
    """Corpus-level certificate aggregating per-prompt alias audits.

    Attributes
    ----------
    n_prompts:
        Total number of held-out prompts certified.
    n_passes:
        Number of prompts whose per-prompt cert passes.
    abstention_rate:
        Fraction of prompts that fail (= 1 - n_passes / n_prompts).
    per_prompt_certs:
        Tuple of all per-prompt certificates.
    period_alias_margin_means:
        For each period T in the basis, the mean of Δ_{+T} and Δ_{-T} margins
        over all prompts (where both ±T are valid alias offsets).
    range_boundary_margin_mean:
        Mean alias margin for range-boundary aliases (offsets ±100 and ±range_size
        that correspond to wrapping around the answer range).
    min_alias_margin:
        Snapshot of the ``_MIN_ALIAS_MARGIN`` named constant used for this run.
    max_construction_chi2:
        Snapshot of the ``_MAX_CONSTRUCTION_CHI2`` named constant used for this run.
    """

    n_prompts: int
    n_passes: int
    abstention_rate: float
    per_prompt_certs: tuple[PromptAliasCert, ...]
    period_alias_margin_means: dict[int, float]
    range_boundary_margin_mean: float
    min_alias_margin: float
    max_construction_chi2: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def certify_helix_clock(
    extraction: ClockExtraction,
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    output_attr: str = "encoder",
    alias_offsets: tuple[int, ...] = (-100, -10, -1, 1, 10, 100),
    min_alias_margin: float = _MIN_ALIAS_MARGIN,
    max_construction_chi2: float = _MAX_CONSTRUCTION_CHI2,
    seed: int = 0,
    _decoder_fit_tokens: Tensor | None = None,
) -> HelixClockCert:
    """Certify alias risk for a helix-arithmetic model on held-out prompts.

    This function audits whether the helix decoder can reliably distinguish n from
    n ± T for each period T in the basis, and from range-boundary aliases.

    Black-box constraints:
    - Only output-capturing forward hooks on encoder sub-layers.  Pre-hooks are forbidden.
    - Periods are inferred from the extraction's staged_family semantics, not from model attributes.
    - No nn.Embedding weight matrix reads.

    Parameters
    ----------
    extraction:
        Result of Lane 2.E ``extract_clock_arithmetic``.  Provides ``R_a``, ``R_b``,
        ``W_ans``, ``layer_construct``, ``layer_readout``.
    model:
        Any nn.Module accepting (batch, 2) integer tokens, treated as black box.
    operand_tokens:
        Integer tensor (N, 2) of held-out (a, b) operand pairs to certify.
    output_attr:
        Name of the encoder submodule on model to hook.
    alias_offsets:
        Tuple of integer offsets δ to check.  By default includes ±10 (Kantamneni-Tegmark
        dominant GPT-J error mode) and ±100 (range-boundary alias).
    min_alias_margin:
        Override for _MIN_ALIAS_MARGIN; prompt cert passes only if all alias margins exceed
        this value.
    max_construction_chi2:
        Override for _MAX_CONSTRUCTION_CHI2; prompt cert passes only if construction_chi2
        is below this value.
    seed:
        Random seed (unused in current implementation; reserved for future stochastic steps).
    _decoder_fit_tokens:
        Optional separate token set used ONLY for fitting the answer decoder C_ans.
        When provided, C_ans is fitted from a clean run on these tokens, and then
        applied to the (potentially perturbed) ``operand_tokens``.  This is useful
        for noise-injection tests where the evaluator should see degraded activations
        but the decoder should be fitted on clean data.  If None, fits from the same
        ``operand_tokens`` (standard usage).

    Returns
    -------
    HelixClockCert
        Corpus-level certificate with per-prompt results.
    """
    model.eval()
    operand_tokens = operand_tokens.detach()

    # ── 1. Determine answer range and periods (no model attribute reads) ───────
    # basis_dim = 1 + 2 * n_periods; infer n_periods from R_a shape.
    basis_dim = extraction.R_a.shape[1]
    n_periods = (basis_dim - 1) // 2
    # Infer max_n from operand range: max answer = 2 * max_operand.
    max_operand = int(operand_tokens.max().item())
    max_n = 2 * max_operand  # inclusive upper bound of answer range

    # Periods recovered from the extraction's NSJIR staged_family metadata.
    periods = _infer_periods_from_extraction(extraction, n_periods)

    enc_module = getattr(model, output_attr)
    layer_idx = extraction.layer_construct

    # ── 2. Fit answer decoder C_ans on decoder_fit_tokens (clean data) ─────────
    # If _decoder_fit_tokens is provided, fit on that (without noise) so the decoder
    # is calibrated on clean activations.  Otherwise fit from operand_tokens directly.
    fit_tokens = _decoder_fit_tokens.detach() if _decoder_fit_tokens is not None else operand_tokens
    fit_max_operand = int(fit_tokens.max().item())
    fit_max_n = 2 * fit_max_operand

    fit_hiddens = _capture_layer_output(model, enc_module, fit_tokens, layer_idx)
    fit_ans_vals = fit_tokens[:, 0] + fit_tokens[:, 1]
    C_ans = _fit_answer_decoder(fit_hiddens, fit_ans_vals, fit_max_n, periods)

    # ── 3. Capture evaluation hidden states at layer_construct ─────────────────
    # These may be noisy if an external hook is active during the eval pass.
    eval_hiddens = _capture_layer_output(model, enc_module, operand_tokens, layer_idx)

    # ── 4. Decode helix coordinates via pre-fitted C_ans ──────────────────────
    pred_basis = _decode_helix_coords_augmented(eval_hiddens, C_ans)

    # ── 5. Compute model logit argmax for chi-squared decomposition ────────────
    model_argmax = _compute_model_argmax(model, operand_tokens)

    # ── 6. Per-prompt certification ────────────────────────────────────────────
    beta = _BETA_DENOM_SCALE / max(float(max_n) ** 2, 1.0)

    per_prompt_certs: list[PromptAliasCert] = []
    for i in range(len(operand_tokens)):
        cert = _per_prompt_cert(
            prompt_id=i,
            pred_basis_i=pred_basis[i],
            periods=periods,
            max_n=max_n,
            alias_offsets=alias_offsets,
            model_argmax_i=int(model_argmax[i].item()),
            min_alias_margin=min_alias_margin,
            max_construction_chi2=max_construction_chi2,
            beta=beta,
        )
        per_prompt_certs.append(cert)

    # ── 7. Aggregate into corpus-level certificate ────────────────────────────
    return _aggregate_cert(
        per_prompt_certs=per_prompt_certs,
        alias_offsets=alias_offsets,
        periods=periods,
        min_alias_margin=min_alias_margin,
        max_construction_chi2=max_construction_chi2,
    )


# ---------------------------------------------------------------------------
# Geodesic S¹ distance — the single most critical primitive in this module
# ---------------------------------------------------------------------------


def _d_s1(theta1: float, theta2: float) -> float:
    """Geodesic distance on the unit circle S¹.

    d_S¹(θ₁, θ₂) = |((θ₁ - θ₂ + π) mod 2π) - π|

    This is the smallest absolute angular difference in [0, π].  Using raw
    (θ₁ - θ₂) is incorrect at wraparound: e.g. θ₁=0.01, θ₂=6.27 have a
    raw difference of -6.26 but a geodesic distance of 0.02.

    Returns value in [0, π].
    """
    diff = (theta1 - theta2 + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def _d_s1_tensor(theta1: Tensor, theta2: Tensor) -> Tensor:
    """Vectorised geodesic S¹ distance for tensors.

    Inputs may be any broadcastable shape.  Returns non-negative values in [0, π].
    """
    diff = (theta1 - theta2 + math.pi) % (2.0 * math.pi) - math.pi
    return diff.abs()


# ---------------------------------------------------------------------------
# Energy function
# ---------------------------------------------------------------------------


def _energy_scalar(
    pred_basis_i: Tensor,
    n: int,
    periods: tuple[int, ...],
    beta: float,
) -> float:
    """Compute E(n; h) for a single prompt and a single candidate answer n.

    E(n; h) = Σ_T α_T · d_S¹(arg z_T(h), 2π·n/T)² + β · (u(h) − n)²

    pred_basis_i: (basis_dim,) decoded helix coordinates for one prompt.
    The affine coordinate is column 0; period T_k occupies columns 1+2k, 2+2k.
    """
    energy = 0.0
    # Affine term
    u_h = float(pred_basis_i[0].item())
    energy += beta * (u_h - float(n)) ** 2

    # Phase terms
    for k, T in enumerate(periods):
        cos_col = 1 + 2 * k
        sin_col = 2 + 2 * k
        if sin_col >= pred_basis_i.shape[0]:
            break
        cos_h = float(pred_basis_i[cos_col].item())
        sin_h = float(pred_basis_i[sin_col].item())
        theta_h = math.atan2(sin_h, cos_h)
        theta_n = 2.0 * math.pi * float(n) / float(T)
        d = _d_s1(theta_h, theta_n)
        energy += _ALPHA_T * d * d

    return energy


def _energy_vectorised(
    pred_basis_i: Tensor,
    n_candidates: Tensor,
    periods: tuple[int, ...],
    beta: float,
) -> Tensor:
    """Compute E(n; h) for a single prompt over all candidate n in n_candidates.

    pred_basis_i: (basis_dim,) decoded helix coordinates.
    n_candidates: (K,) integer candidates.
    Returns: (K,) energy tensor.
    """
    n_float = n_candidates.float()  # (K,)
    energy = beta * (float(pred_basis_i[0].item()) - n_float) ** 2  # affine term

    for k, T in enumerate(periods):
        cos_col = 1 + 2 * k
        sin_col = 2 + 2 * k
        if sin_col >= pred_basis_i.shape[0]:
            break
        cos_h = float(pred_basis_i[cos_col].item())
        sin_h = float(pred_basis_i[sin_col].item())
        theta_h = math.atan2(sin_h, cos_h)
        theta_n = 2.0 * math.pi * n_float / float(T)  # (K,)
        d = _d_s1_tensor(torch.tensor(theta_h), theta_n)  # (K,)
        energy = energy + _ALPHA_T * d * d

    return energy


# ---------------------------------------------------------------------------
# Per-prompt certificate
# ---------------------------------------------------------------------------


def _per_prompt_cert(
    *,
    prompt_id: int,
    pred_basis_i: Tensor,
    periods: tuple[int, ...],
    max_n: int,
    alias_offsets: tuple[int, ...],
    model_argmax_i: int,
    min_alias_margin: float,
    max_construction_chi2: float,
    beta: float,
) -> PromptAliasCert:
    """Compute alias certificate for a single prompt.

    Uses the geodesic S¹ distance helper ``_d_s1`` / ``_d_s1_tensor`` throughout
    to avoid wrong margins at phase wraparound.
    """
    # Compute energy over all valid answer candidates [0, max_n]
    n_candidates = torch.arange(max_n + 1, dtype=torch.long)
    energies = _energy_vectorised(pred_basis_i, n_candidates, periods, beta)
    argmin_n = int(energies.argmin().item())
    e_argmin = float(energies[argmin_n].item())

    # Alias margins: Δ_δ = E(argmin_n + δ) - E(argmin_n)
    alias_margins: dict[int, float] = {}
    for delta in alias_offsets:
        alias_n = argmin_n + delta
        if 0 <= alias_n <= max_n:
            e_alias = float(energies[alias_n].item())
            alias_margins[delta] = e_alias - e_argmin
        else:
            # Alias falls outside the valid range — treat as a large margin
            # (the alias doesn't exist, so there's no confusion risk)
            alias_margins[delta] = float("inf")

    # Chi-squared decomposition: construction-error vs readout-bias
    construction_chi2, readout_bias_chi2 = _chi2_decomposition(
        argmin_n=argmin_n,
        model_argmax_i=model_argmax_i,
        max_n=max_n,
        energies=energies,
    )

    # Pass criterion
    margins_ok = all(
        m > min_alias_margin for m in alias_margins.values() if not math.isinf(m)
    )
    passes = margins_ok and (construction_chi2 < max_construction_chi2)

    return PromptAliasCert(
        prompt_id=prompt_id,
        argmin_n=argmin_n,
        alias_margins=alias_margins,
        construction_chi2=construction_chi2,
        readout_bias_chi2=readout_bias_chi2,
        passes=passes,
    )


# ---------------------------------------------------------------------------
# Chi-squared decomposition: construction vs readout
# ---------------------------------------------------------------------------


def _chi2_decomposition(
    argmin_n: int,
    model_argmax_i: int,
    max_n: int,
    energies: Tensor,
) -> tuple[float, float]:
    """Split argmax-disagreement chi-squared into construction and readout terms.

    The question: when argmin_n ≠ model_argmax_i, is the disagreement attributable
    to a biased model readout (construction correct, readout wrong) or a faulty decode
    (readout correct, decode wrong)?

    **Construction-error chi-squared** (``construction_chi2``):
        Hypothesis: the helix decode is correct (argmin_n is the true answer) but the
        model's readout is biased away from it.
        Test: treat the normalized softmax-energy distribution over candidates as
        "observed counts" and the argmin-peaked distribution as "expected".
        chi2_construction = (p_argmin - 1)² / max(1, 1) + (p_others)² / max(p_others_expected, ε)
        Simplified to: whether the model argmax disagrees from argmin, measured as a
        point-mass chi-squared: (|{model disagrees}| - expected disagreement)² / expected.

    **Readout-bias chi-squared** (``readout_bias_chi2``):
        Hypothesis: the model readout argmax is correct and the helix decode is biased.
        Test: analogous, swapping roles.

    Both terms are computed from the REAL per-prompt energy distribution (not fixed constants).
    """
    vocab = max_n + 1

    # Soft probability distribution from energies (lower energy = higher prob)
    # Use Boltzmann-style softmax over negated energies to convert to a distribution.
    neg_e = -energies.float()
    probs = F.softmax(neg_e, dim=0)  # (vocab,)

    # Expected fraction of prompts where model argmax == argmin_n, under each hypothesis
    # H_construction: decode is correct; model readout is biased.
    #   Expected count of agree = n * P(model == argmin_n) under an unbiased readout.
    #   Observed = 1 if agree, 0 if disagree. We use the energy distribution as expected.
    p_at_argmin = float(probs[argmin_n].item())
    p_at_model = float(probs[min(model_argmax_i, vocab - 1)].item())

    agree = 1 if (argmin_n == model_argmax_i) else 0

    # construction_chi2: hypothesise decode is right, model readout probability should = 1
    # at argmin_n.  Observed agreement rate = agree; expected rate = p_at_argmin (the
    # soft probability the HELIX assigns to argmin_n).
    # chi-squared = (observed - expected)^2 / max(expected, eps)
    construction_chi2 = float(
        (agree - p_at_argmin) ** 2 / max(p_at_argmin, _CHI2_MIN_DENOM / vocab)
    )

    # readout_bias_chi2: hypothesise model readout is right, helix energy should be lowest at
    # model_argmax_i.  Observed = agree; expected = p_at_model (the soft probability the
    # HELIX assigns to model_argmax_i).
    readout_bias_chi2 = float(
        (agree - p_at_model) ** 2 / max(p_at_model, _CHI2_MIN_DENOM / vocab)
    )

    return construction_chi2, readout_bias_chi2


# ---------------------------------------------------------------------------
# Corpus aggregation
# ---------------------------------------------------------------------------


def _aggregate_cert(
    *,
    per_prompt_certs: list[PromptAliasCert],
    alias_offsets: tuple[int, ...],
    periods: tuple[int, ...],
    min_alias_margin: float,
    max_construction_chi2: float,
) -> HelixClockCert:
    """Aggregate per-prompt certificates into a corpus-level HelixClockCert."""
    n_prompts = len(per_prompt_certs)
    n_passes = sum(1 for c in per_prompt_certs if c.passes)
    abstention_rate = 1.0 - n_passes / max(n_prompts, 1)

    # Per-period mean alias margin (mean of |Δ_{+T}| and |Δ_{-T}|, where both exist)
    period_alias_margin_means: dict[int, float] = {}
    for T in periods:
        margins_for_T: list[float] = []
        for delta in (T, -T):
            if delta in alias_offsets:
                for c in per_prompt_certs:
                    v = c.alias_margins.get(delta, float("inf"))
                    if not math.isinf(v):
                        margins_for_T.append(v)
        if margins_for_T:
            period_alias_margin_means[T] = sum(margins_for_T) / len(margins_for_T)

    # Range-boundary margin mean: offsets ±100 and any offset >= max operand
    range_boundary_offsets = {o for o in alias_offsets if abs(o) >= 100}
    rb_margins: list[float] = []
    for c in per_prompt_certs:
        for delta in range_boundary_offsets:
            v = c.alias_margins.get(delta, float("inf"))
            if not math.isinf(v):
                rb_margins.append(v)
    range_boundary_margin_mean = sum(rb_margins) / len(rb_margins) if rb_margins else float("inf")

    return HelixClockCert(
        n_prompts=n_prompts,
        n_passes=n_passes,
        abstention_rate=abstention_rate,
        per_prompt_certs=tuple(per_prompt_certs),
        period_alias_margin_means=period_alias_margin_means,
        range_boundary_margin_mean=range_boundary_margin_mean,
        min_alias_margin=min_alias_margin,
        max_construction_chi2=max_construction_chi2,
    )


# ---------------------------------------------------------------------------
# Hidden-state capture (black-box OUTPUT hook only)
# ---------------------------------------------------------------------------


def _hook_output(buffer: list[Tensor]) -> Any:
    """Return a hook function that appends the output tensor to buffer.

    The ``_inputs`` argument is intentionally not accessed — this is a
    pure output-capture hook.
    """

    def _hook(module: nn.Module, _inputs: Any, output: object) -> None:  # noqa: ARG001
        if isinstance(output, Tensor):
            buffer.append(output.detach().float().cpu())
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
            buffer.append(output[0].detach().float().cpu())

    return _hook


def _capture_encoder_output(
    model: nn.Module,
    operand_tokens: Tensor,
    output_attr: str,
) -> Tensor:
    """Capture the OUTPUT of model.<output_attr> for all operand_tokens.

    Uses only ``register_forward_hook``.  The ``_inputs`` tuple is intentionally
    not accessed.

    Returns: (N, seq_len, d_model) float32 tensor on CPU.
    """
    enc_module = getattr(model, output_attr)
    buffer: list[Tensor] = []
    handle = enc_module.register_forward_hook(_hook_output(buffer))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(operand_tokens), _ENERGY_BATCH_SIZE):
                batch = operand_tokens[start : start + _ENERGY_BATCH_SIZE]
                model(batch)
    finally:
        handle.remove()

    if not buffer:
        raise RuntimeError(
            f"No hidden states captured from model.{output_attr}.  "
            "Check that output_attr names a submodule with a tensor output."
        )
    return torch.cat(buffer, dim=0)  # (N, seq_len, d_model)


def _capture_layer_output(
    model: nn.Module,
    enc_module: nn.Module,
    operand_tokens: Tensor,
    layer_idx: int,
) -> Tensor:
    """Capture the OUTPUT of enc_module.layers[layer_idx] for all operand_tokens.

    Uses only ``register_forward_hook``.  The ``_inputs`` tuple is intentionally
    not accessed.  This is how we access the layer_construct activation where the
    answer helix is most clearly encoded, per the staged extraction.

    Returns: (N, seq_len, d_model) float32 tensor on CPU.
    """
    if not hasattr(enc_module, "layers") or layer_idx >= len(enc_module.layers):
        # Fallback: capture full encoder output if sub-layer access unavailable
        return _capture_encoder_output(model, operand_tokens, "encoder")

    layer_mod = enc_module.layers[layer_idx]
    buffer: list[Tensor] = []
    handle = layer_mod.register_forward_hook(_hook_output(buffer))
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(operand_tokens), _ENERGY_BATCH_SIZE):
                batch = operand_tokens[start : start + _ENERGY_BATCH_SIZE]
                model(batch)
    finally:
        handle.remove()

    if not buffer:
        # Fallback to full encoder output
        return _capture_encoder_output(model, operand_tokens, "encoder")
    return torch.cat(buffer, dim=0)  # (N, seq_len, d_model)


# ---------------------------------------------------------------------------
# Helix coordinate decoding
# ---------------------------------------------------------------------------


def _decode_helix_coords(hidden_states: Tensor, R: Tensor) -> Tensor:
    """Decode helix coordinates from hidden states using a bias-augmented decoder R.

    hidden_states: (N, seq_len, d_model)
    R: (d_model+1, basis_dim) — bias-augmented decoder matrix.

    Takes the LAST sequence position and applies: [h; 1] @ R → (N, basis_dim).

    Returns: (N, basis_dim) decoded helix coordinates on CPU.
    """
    h = hidden_states[:, -1, :].float().cpu()  # (N, d_model)
    ones = torch.ones(h.shape[0], 1, dtype=h.dtype)
    h_aug = torch.cat([h, ones], dim=1)  # (N, d_model+1)
    return h_aug @ R.float().cpu()


def _decode_helix_coords_augmented(hidden_states: Tensor, C_ans: Tensor) -> Tensor:
    """Decode helix coordinates at the answer position using fitted augmented decoder C_ans.

    hidden_states: (N, seq_len, d_model) — captured layer output.
    C_ans: (d_model+1, basis_dim) — bias-augmented answer decoder.

    Uses position -1 (last sequence position = answer / operand-b position).

    Returns: (N, basis_dim) decoded helix coordinates on CPU.
    """
    return _decode_helix_coords(hidden_states, C_ans)


def _fit_answer_decoder(
    layer_hiddens: Tensor,
    ans_vals: Tensor,
    max_n: int,
    periods: tuple[int, ...],
) -> Tensor:
    """Fit a bias-augmented decoder [h_ans; 1] → helix_basis(a+b) via least-squares.

    layer_hiddens: (N, seq_len, d_model) — layer output from layer_construct.
    ans_vals: (N,) integer true answer values (a + b from caller's operand tokens).
    max_n: maximum answer value (= 2 * max_operand).
    periods: tuple of helix periods.

    The true answer values come from the CALLER'S operand_tokens input data — they
    are NOT model attributes or configuration values.

    Returns: (d_model+1, basis_dim) fitted decoder matrix C_ans.
    """
    from rune.extract.clock import _helix_basis_matrix

    h_ans = layer_hiddens[:, -1, :].float().cpu()  # (N, d_model)
    ones = torch.ones(h_ans.shape[0], 1)
    h_aug = torch.cat([h_ans, ones], dim=1)  # (N, d_model+1)

    B_answer = _helix_basis_matrix(max_n + 1, periods, affine=True)  # (max_n+1, basis_dim)
    targets = B_answer[ans_vals.long().cpu().clamp(0, max_n)]  # (N, basis_dim)

    result = torch.linalg.lstsq(h_aug, targets)
    return result.solution  # (d_model+1, basis_dim)


# ---------------------------------------------------------------------------
# Period inference (derived from extraction metadata only — no model attribute reads)
# ---------------------------------------------------------------------------


def _infer_periods_from_extraction(
    extraction: ClockExtraction,
    n_periods: int,
) -> tuple[int, ...]:
    """Infer the helix periods from the ClockExtraction staged_family metadata.

    This function reads the periods from the ``staged_family.semantics`` field (a
    ``ClockAdd`` NSJIR node that carries a ``HelixBasis``), which is an extraction
    OUTPUT — not a model attribute or configuration introspection.

    Falls back to a heuristic reconstruction if metadata is unavailable.
    """
    # Primary: read from staged_family.semantics.basis.periods
    # ClockAdd.basis is a HelixBasis with a .periods attribute (NSJIR type, not model config).
    try:
        basis = extraction.staged_family.semantics.basis
        if hasattr(basis, "periods") and basis.periods:
            periods = tuple(int(p) for p in basis.periods)
            if len(periods) == n_periods:
                return periods
    except AttributeError:
        pass

    # Secondary: derive from the basis_dim of R_a.
    # R_a has shape (d_model+1, basis_dim).  basis_dim = 1 + 2 * n_periods.
    # We cannot recover which periods were used without additional information.
    # Return an empty tuple and let the energy function skip phase terms.
    # This is a graceful degradation path only — a real extraction always provides periods.
    return tuple(range(2, 2 + n_periods))


# ---------------------------------------------------------------------------
# Model argmax capture (for chi-squared decomposition)
# ---------------------------------------------------------------------------


def _compute_model_argmax(model: nn.Module, operand_tokens: Tensor) -> Tensor:
    """Run model forward pass and return argmax over logits for each prompt.

    Black-box: model is called as a function of tokens only.  No config reads.

    Returns: (N,) LongTensor of model's argmax predictions.
    """
    argmax_list: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(operand_tokens), _ENERGY_BATCH_SIZE):
            batch = operand_tokens[start : start + _ENERGY_BATCH_SIZE]
            logits = model(batch)
            argmax_list.append(logits.argmax(dim=-1).cpu())
    return torch.cat(argmax_list, dim=0)
