"""Lane 2.E — Clock Arithmetic Extractor.

Given a model with detected helix manifolds for operand tokens and answer state,
extracts a staged NSJIR program:

    transport:  gather_helix(pos_a), gather_helix(pos_b)
    construct:  s = add_int(a, b);  answer_state = helix_encode(B_T, s)
    readout:    resume_model_from(layer = l_readout)

Black-box constraints (enforced by design, audited in test_clock_extractor_anti_cheat_audit.py):
  - Only register_forward_hook (OUTPUT capture), never register_forward_pre_hook or _inputs[0].
  - No model.config reads, no *Config instantiation, no .modulus / .periods as model attribute.
  - No token_embedding.weight reads; the embedding hook captures OUTPUT, not weights.
  - Decoder R_a is a genuine least-squares fit, never a token-index shortcut.
  - Phase law is verified using DECODED helix coordinates from hidden states, never ground truth.
  - held_out_kl comes from a real write-and-resume forward pass, not synthesis of the unembed.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rune.nsjir import (
    ClockAdd,
    HelixBasis,
    MechanismStage,
    MechanismStageContract,
    StagedMechanismFamily,
    call,
    var,
)
from rune.nsjir.types import IntRange

# ─── Named calibration constants ──────────────────────────────────────────────
# All numeric thresholds appear here with docstrings; no anonymous magic numbers.

_PINV_RTOL: float = 1e-3
"""Relative tolerance for torch.linalg.pinv when computing W_ans = pinv(C_ans_linear).

Calibration note:
- At basis_dim ≤ 16 (e.g. DEFAULT_HELIX_PERIODS gives basis_dim=9), the
  unregularized pinv works fine — the effective singular-value cutoff sits well
  below 1e-4 and max round-trip reconstruction error is < 1e-4.
- At basis_dim ≥ 32 (e.g. WIDE_HELIX_PERIODS_30 gives basis_dim=61), the
  C_ans_linear matrix (d_model=2560, basis_dim=61) is tall but can have poorly
  conditioned small singular values from nearly-collinear character-pair columns.
  The wide-period re-extraction (docs/pythia_2.8b_wide_period_reextract.md)
  measured max reconstruction error 0.47 at basis_dim=61 with rtol=default
  (~1e-15).  rtol=1e-3 is the conservative choice: it truncates singular values
  smaller than 1e-3 * sigma_max, removing noise-dominated components and
  bringing max round-trip error below 1e-2.
- The operational identity h @ C_ans_linear @ pinv(C_ans_linear) @ C_ans_linear
  ≈ h @ C_ans_linear is preserved within the 1e-2 reconstruction tolerance."""

# ─── Named period-set constants ───────────────────────────────────────────────

DEFAULT_HELIX_PERIODS: tuple[int, ...] = (2, 5, 10, 100)
"""Synthetic-helix planted period set.  Matches the HelixAddTransformer
synthetic by construction.  Use for the helix-add testbench."""

WIDE_HELIX_PERIODS_30: tuple[int, ...] = (
    2, 9, 16, 22, 29, 36, 43, 50, 57, 63, 70, 77, 84, 91, 98,
    104, 111, 118, 125, 132, 139, 145, 152, 159, 166, 173, 180, 186, 193, 200,
)
"""30-period basis discovered by the abelian-character SAE at Pythia-2.8B
(`docs/pythia_2.8b_abelian_sae.md`).  Use for real-LM Lane 2.E extraction
on integer-arithmetic prompts.  basis_dim with affine = 61."""

WIDE_HELIX_PERIODS_TOP10: tuple[int, ...] = (2, 29, 36, 50, 91, 111, 139, 145, 186, 200)
"""Top-10 by mean amplitude in the abelian-SAE result.  Smaller basis
(basis_dim with affine = 21) when the 30-period basis is overkill or
when pinv conditioning at the full basis_dim is too poor."""

PHASE_LAW_TOLERANCE: float = 0.1
"""Maximum acceptable average ||z_T^ans - z_T^a * z_T^b||^2 over certified periods.
Values above this indicate the model does NOT implement Clock arithmetic reliably.
Derived from empirical calibration on HelixAddTransformer (4-layer, d_model=64,
periods={2,5,10,100}): clean law residual ~0.012 << tolerance."""

AFFINE_LAW_TOLERANCE: float = 0.5
"""Default tolerance for ||û^ans - (û^a + û^b)||^2 in raw-integer units.
NOTE: For 4-layer synthetic models the answer affine decoder has MSE ~12 because
the transformer mixes the answer coordinates.  The extractor records the residual
honestly; tests should pass a model-appropriate threshold.  The default 0.5 is
achievable only when an accurate answer-affine channel exists."""

STAGE_BOUNDARY_MIN_IMPROVEMENT: float = 0.05
"""Minimum phase-law-quality improvement needed for a layer to count as the
construct boundary.  Prevents spurious boundary detection from noise."""

HELD_OUT_SPLIT: float = 0.3
"""Fraction of operand_tokens reserved as held-out for KL and kill-criterion."""

PHASE_RESIDUAL_SCALE: float = 2.0
"""Expected maximum squared residual for a pure-random baseline on the unit circle.
||z1 - z2||^2 ≤ 4 always.  Random models score near 2.0 on average."""

KL_THRESHOLD: float = 0.05
"""kill_criterion threshold for KL(original || helix-write continuation) on held-out."""

MIN_DECODER_ACCURACY: float = 0.80
"""Minimum required fraction of correct integer predictions from R_a and R_b on
held-out data before the kills_kill_criterion can be True."""


# ─── Public result type ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClockExtraction:
    """Result of running Clock-arithmetic extraction on a helix-arithmetic model.

    All tensors live on CPU (the extractor always moves data off device before storing).
    """

    staged_family: StagedMechanismFamily
    """Staged NSJIR program: transport → construct → readout."""

    R_a: Tensor
    """Decoder matrix (d_model+1, basis_dim): maps [h_a; 1] → helix_basis(a)."""

    R_b: Tensor
    """Decoder matrix (d_model+1, basis_dim): maps [h_b; 1] → helix_basis(b)."""

    W_ans: Tensor
    """Encoder matrix (basis_dim, d_model): maps helix_basis(s) → Δh for writing."""

    C_ans_linear: Tensor
    """Decoder matrix (d_model, basis_dim): maps hidden state → helix_basis(s).
    The forward map matched to W_ans by W_ans = pinv(C_ans_linear).  Stored so
    that runtime JIT (src/rune/schedule/jit.py) can compute the residual write
    delta_h = (target_helix - h @ C_ans_linear) @ W_ans without re-fitting."""

    layer_construct: int
    """Index of the encoder layer whose output carries the answer helix most clearly."""

    layer_readout: int
    """Index of the encoder layer where the Clock write-and-resume is applied."""

    phase_law_residual: float
    """Mean ||ẑ_T^ans - ẑ_T^a · ẑ_T^b||^2 over certified periods, on held-out.
    Computed from decoded helix coordinates, not ground-truth integers."""

    affine_law_residual: float
    """Mean ||û^ans - (û^a + û^b)||^2 on held-out, in RAW INTEGER UNITS (squared).
    Computed from decoded affine coordinates, not ground-truth integers.
    Empirical baseline on the 4-layer HelixAddTransformer: ~12 (RMS ~3.5 units of
    a+b ∈ [0,198]).  The non-zero baseline reflects decoder noise + residual-stream
    mixing across layers, not absence of the affine channel: per-layer linear
    regression of hidden→(a+b) achieves Pearson r=0.996 at the best layer."""

    held_out_kl: float
    """KL(original_logits || helix_write_logits) on held-out, from a real write-and-resume."""

    fits_kill_criterion: bool
    """**Operational** gate: True iff the helix write-and-resume preserves model
    behavior on held-out, i.e., ``held_out_kl < KL_THRESHOLD`` AND the operand
    decoders meet ``MIN_DECODER_ACCURACY``.  This is the JIT-fire decision.

    Decoupled from ``mechanistic_advisory`` (phase_law + affine_law) because
    those gates were calibrated on the 99.75%-accurate HelixAddTransformer
    synthetic and reject behaviorally-faithful write-and-resume on real LMs
    where the helix mechanism is weaker but the operational fidelity is fine.
    See commit message of the refactor + ``docs/pythia_jit_operational_test.md``.

    For substrates that pass operationally but fail mechanistically (i.e.,
    fits_kill_criterion=True but mechanistic_advisory=False), the JIT can fire,
    but a downstream interpretability claim about HOW the model implements
    arithmetic should treat the helix-shape evidence as advisory only."""

    mechanistic_advisory: bool
    """**Advisory** signal: True iff the mechanistic residuals (phase_law,
    affine_law) are within their synthetic-calibrated thresholds.  When False
    on a substrate that passes ``fits_kill_criterion``, the helix mechanism is
    weak but the JIT write-and-resume still preserves behavior — interpret as
    "the model is implementing something near-helix, write-and-resume agrees,
    but the mechanistic structure is not as crisp as the planted synthetic."

    Computed as ``phase_law_residual < PHASE_LAW_TOLERANCE AND
    affine_law_residual < AFFINE_LAW_TOLERANCE``.  Both tolerances are calibrated
    on the HelixAdd synthetic and are intentionally tight."""


# ─── Main public function ──────────────────────────────────────────────────────


def extract_clock_arithmetic(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    output_attr: str = "encoder",
    operand_positions: tuple[int, int] = (0, 1),
    answer_position: int | None = None,
    operand_value_range: tuple[int, int] = (0, 99),
    answer_value_range: tuple[int, int] = (0, 198),
    periods_to_verify: tuple[int, ...] = (2, 5, 10, 100),
    phase_law_tolerance: float = PHASE_LAW_TOLERANCE,
    affine_law_tolerance: float = AFFINE_LAW_TOLERANCE,
    resume_fn: Callable[[Tensor, int], Tensor] | None = None,
    embedding_attr: str | None = None,
    seed: int = 0,
) -> ClockExtraction:
    """Extract a staged Clock-arithmetic NSJIR program from a helix-arithmetic model.

    Parameters
    ----------
    model:
        Any nn.Module that accepts (batch, 2) integer token tensors and returns
        logits (batch, vocab_size).  The extractor treats it as a black box:
        only forward hooks on OUTPUT tensors are used.
    operand_tokens:
        Integer tensor (N, 2) of operand pairs (a, b) to use for fitting.
    output_attr:
        Name of the attribute on *model* that contains the encoder module.
        Forward hooks are registered on ``model.<output_attr>`` and its
        sub-layers (encoder.layers[i]).  For HelixAddTransformer use ``"encoder"``;
        for Pythia (GPTNeoXForCausalLM) use ``"gpt_neox"``.  The token-embedding
        hook is registered on ``model.token_embedding`` (or on ``embedding_attr``
        when provided).
    operand_positions:
        (pos_a, pos_b) — sequence positions for operand a and operand b
        in the (batch, seq, d_model) hidden tensor.
    answer_position:
        Sequence position where the answer prediction lives.  Defaults to
        ``operand_positions[1]`` (HelixAdd behaviour: 2-token sequence with
        the answer at position 1).  For prompts like ``"a+b="`` use the
        position of the ``"="`` token (typically the last input token).
    operand_value_range:
        (lo, hi) integer range for operands a, b.  The B_operand basis is
        built for ``[lo, hi]`` (i.e. ``hi - lo + 1`` rows).  Default ``(0, 99)``.
    answer_value_range:
        (lo, hi) integer range for the sum a + b.  Default ``(0, 198)``.
    periods_to_verify:
        The set of helix periods to check in the phase-law verification.
    phase_law_tolerance:
        Residual threshold for accepting that the phase law holds.
    affine_law_tolerance:
        Residual threshold for accepting that the affine law holds.
    resume_fn:
        Optional callable ``(patched_hidden, from_layer) -> logits`` that
        runs encoder layers ``[from_layer, n_layers)`` followed by the
        model's final-norm and unembed.  When ``None`` (default), the
        extractor uses the built-in chain that assumes
        ``model.final_norm`` and ``model.unembed`` (HelixAddTransformer).
        Provide a custom callable for HuggingFace models such as Pythia
        (GPTNeoX) whose layer signatures require ``position_embeddings``.
    embedding_attr:
        Override for the embedding hook target.  ``None`` uses
        ``"token_embedding"`` if present (HelixAdd).  For Pythia, use
        ``"gpt_neox.embed_in"``.
    seed:
        Random seed for the fit / held-out split.

    Returns
    -------
    ClockExtraction
        The extracted staged program and all associated fit metadata.
    """
    model.eval()
    operand_tokens = operand_tokens.detach()

    pos_a, pos_b = operand_positions
    # Default answer position is the second operand position (HelixAdd behaviour).
    pos_ans_resolved = pos_b if answer_position is None else int(answer_position)

    # ── 1. Split into fit and held-out ────────────────────────────────────────
    rng = torch.Generator().manual_seed(seed)
    n_total = operand_tokens.shape[0]
    perm = torch.randperm(n_total, generator=rng)
    n_held = max(1, int(HELD_OUT_SPLIT * n_total))
    n_fit = n_total - n_held
    fit_idx = perm[:n_fit]
    test_idx = perm[n_fit:]

    # ── 2. Capture activations (black-box OUTPUT hooks only) ──────────────────
    enc_module = getattr(model, output_attr)
    n_layers = _count_encoder_layers(enc_module)

    # Capture per-layer residuals (encoder.layers[i] OUTPUT).
    # Some architectures (e.g. GPTNeoXLayer) return a tuple
    # ``(hidden_states, ...)`` from forward; we strip the tuple before
    # caching so downstream consumers always see a (B, S, D) Tensor.
    layer_buffers: dict[int, list[Tensor]] = {i: [] for i in range(n_layers)}
    handles: list[Any] = []
    for i in range(n_layers):
        layer_mod = enc_module.layers[i]

        def _make_layer_hook(idx: int):
            def _hook(module: nn.Module, inp: Any, output: Any) -> None:  # noqa: ARG001
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, Tensor):
                    # Cast to float32 so downstream lstsq / cdist / matmul work
                    # uniformly across bfloat16, float16, and float32 backends.
                    layer_buffers[idx].append(tensor.detach().float().cpu())

            return _hook

        handles.append(layer_mod.register_forward_hook(_make_layer_hook(i)))

    # Capture token-embedding OUTPUT when it exists (no weight read)
    emb_buffer: list[Tensor] = []
    emb_module = _resolve_embedding_module(model, embedding_attr)
    if emb_module is not None:
        def _emb_hook(module: nn.Module, inp: Any, output: Tensor) -> None:  # noqa: ARG001
            # Cast to float32; matches the layer hook for dtype consistency.
            emb_buffer.append(output.detach().float().cpu())

        handles.append(emb_module.register_forward_hook(_emb_hook))

    # Run all samples through the model in batches
    _run_batched(model, operand_tokens, batch_size=256)

    for h in handles:
        h.remove()

    # Concatenate captures
    layer_hiddens: dict[int, Tensor] = {}
    for i in range(n_layers):
        if layer_buffers[i]:
            cat = torch.cat(layer_buffers[i], dim=0)
            # Expected shape: (N, seq_len, d_model) — skip if unexpected
            if cat.ndim == 3:
                layer_hiddens[i] = cat
    emb_hidden: Tensor | None = None
    if emb_buffer:
        cat_emb = torch.cat(emb_buffer, dim=0)
        if cat_emb.ndim == 3:
            emb_hidden = cat_emb

    # ── 3. Fit operand decoders R_a, R_b ─────────────────────────────────────
    # Prefer the token-embedding output (highest signal before attention mixes).
    # Fall back to the earliest available encoder layer.
    operand_source = _pick_operand_source(emb_hidden, layer_hiddens)

    if operand_source is None:
        # Graceful degradation: no valid operand source (e.g. Identity encoder)
        d_fallback = 1
        basis_dim = 1 + 2 * len(periods_to_verify)
        R_a = torch.zeros(d_fallback + 1, basis_dim)
        R_b = torch.zeros(d_fallback + 1, basis_dim)
        W_ans = torch.zeros(basis_dim, d_fallback)
        return _make_degenerate_extraction(
            R_a, R_b, W_ans, periods_to_verify, n_layers
        )

    d_model = operand_source.shape[-1]
    basis_dim = 1 + 2 * len(periods_to_verify)
    op_lo, op_hi = operand_value_range
    ans_lo, ans_hi = answer_value_range
    B_operand = _helix_basis_matrix(op_hi + 1, periods_to_verify, affine=True)
    B_answer = _helix_basis_matrix(ans_hi + 1, periods_to_verify, affine=True)

    # Guard: if pos_a, pos_b, or pos_ans exceeds available positions, degenerate
    seq_len = operand_source.shape[1]
    if pos_a >= seq_len or pos_b >= seq_len or pos_ans_resolved >= seq_len:
        R_a = torch.zeros(d_model + 1, basis_dim)
        R_b = torch.zeros(d_model + 1, basis_dim)
        W_ans = torch.zeros(basis_dim, d_model)
        return _make_degenerate_extraction(R_a, R_b, W_ans, periods_to_verify, n_layers)

    # Fit R_a: [h_a; 1] -> B_operand(a)
    a_vals = operand_tokens[:, 0]
    b_vals = operand_tokens[:, 1]

    h_a_fit, h_a_fit_aug = _augmented(operand_source[fit_idx, pos_a, :])
    h_b_fit, h_b_fit_aug = _augmented(operand_source[fit_idx, pos_b, :])

    target_a = B_operand[a_vals[fit_idx]]  # (n_fit, basis_dim)
    target_b = B_operand[b_vals[fit_idx]]

    R_a = _lstsq(h_a_fit_aug, target_a)  # (d_model+1, basis_dim)
    R_b = _lstsq(h_b_fit_aug, target_b)

    # ── 4. Stage-boundary detector ────────────────────────────────────────────
    # Scan per-layer phase-law quality at pos_b (answer accumulates there)
    # to find l_construct (layer where answer helix is most clearly encoded)
    # and l_readout (layer after which we resume the model).
    layer_scores: dict[int, float] = {}
    layer_C_ans: dict[int, Tensor] = {}

    for i, h_layer in layer_hiddens.items():
        if h_layer.shape[1] <= max(pos_a, pos_b, pos_ans_resolved):
            continue
        h_ans = h_layer[:, pos_ans_resolved, :]  # (N, d_model) — answer position
        ans_vals = a_vals + b_vals
        h_ans_fit_aug = _augmented(h_ans[fit_idx])[1]
        target_ans = B_answer[ans_vals[fit_idx]]
        C_ans_i = _lstsq(h_ans_fit_aug, target_ans)

        # Phase-law quality: on held-out, how well does phase law hold?
        h_a_test_aug = _augmented(operand_source[test_idx, pos_a, :])[1]
        h_b_test_aug = _augmented(operand_source[test_idx, pos_b, :])[1]
        h_ans_test_aug = _augmented(h_ans[test_idx])[1]

        pred_basis_a = h_a_test_aug @ R_a
        pred_basis_b = h_b_test_aug @ R_b
        pred_basis_ans = h_ans_test_aug @ C_ans_i

        score = _phase_law_quality(pred_basis_a, pred_basis_b, pred_basis_ans, periods_to_verify)
        layer_scores[i] = score
        layer_C_ans[i] = C_ans_i

    if not layer_scores:
        R_a = torch.zeros(d_model + 1, basis_dim)
        R_b = torch.zeros(d_model + 1, basis_dim)
        W_ans = torch.zeros(basis_dim, d_model)
        return _make_degenerate_extraction(R_a, R_b, W_ans, periods_to_verify, n_layers)

    # l_construct = layer with lowest phase-law residual (best clock encoding)
    layer_construct = min(layer_scores, key=lambda k: layer_scores[k])
    # l_readout = one layer after l_construct (where we write-and-resume)
    layer_readout = min(layer_construct + 1, n_layers - 1)

    C_ans = layer_C_ans[layer_construct]

    # ── 5. Phase-law verification (on held-out, from decoded helix coords) ────
    h_a_test_aug = _augmented(operand_source[test_idx, pos_a, :])[1]
    h_b_test_aug = _augmented(operand_source[test_idx, pos_b, :])[1]

    h_ans_layer = layer_hiddens.get(layer_construct)
    if h_ans_layer is None:
        phase_residual = float("inf")
        affine_residual = float("inf")
    else:
        h_ans_test_aug = _augmented(h_ans_layer[test_idx, pos_ans_resolved, :])[1]

        pred_basis_a = h_a_test_aug @ R_a  # (n_test, basis_dim)
        pred_basis_b = h_b_test_aug @ R_b
        pred_basis_ans = h_ans_test_aug @ C_ans

        phase_residual = _phase_law_residual(
            pred_basis_a, pred_basis_b, pred_basis_ans, periods_to_verify
        )
        affine_residual = _affine_law_residual(pred_basis_a, pred_basis_b, pred_basis_ans)

    # ── 6. Answer encoder W_ans ───────────────────────────────────────────────
    # Minimum-norm write: delta_h = (B_answer[s] - h @ C_linear) @ C_pinv
    # where C_linear = C_ans[:d_model, :] (linear part, without bias column)
    C_ans_linear = C_ans[:d_model, :]  # (d_model, basis_dim)
    W_ans = torch.linalg.pinv(C_ans_linear, rtol=_PINV_RTOL)  # (basis_dim, d_model)

    # ── 7. Decode integer accuracy on held-out ────────────────────────────────
    pred_a_int = _decode_integers(h_a_test_aug @ R_a, B_operand)
    pred_b_int = _decode_integers(h_b_test_aug @ R_b, B_operand)

    a_test = a_vals[test_idx]
    b_test = b_vals[test_idx]
    acc_a = float((pred_a_int == a_test).float().mean().item())
    acc_b = float((pred_b_int == b_test).float().mean().item())

    # ── 8. held_out_kl via real write-and-resume forward pass ─────────────────
    test_tokens = operand_tokens[test_idx]
    true_ans_test = a_vals[test_idx] + b_vals[test_idx]
    held_out_kl, patched_acc = _compute_kl_write_and_resume(
        model=model,
        enc_module=enc_module,
        test_tokens=test_tokens,
        true_answer=true_ans_test,
        layer_readout=layer_readout,
        n_layers=n_layers,
        B_answer=B_answer,
        C_ans_linear=C_ans_linear,
        W_ans=W_ans,
        pos_ans=pos_ans_resolved,
        layer_hiddens=layer_hiddens,
        resume_fn=resume_fn,
    )

    # ── 9. Kill criterion ─────────────────────────────────────────────────────
    # Operational gate (fits_kill_criterion): held_out_kl + decoder accuracy.
    #   These are the BEHAVIORAL gates — does the write-and-resume preserve
    #   the model's continuation, and are the operand decoders accurate enough
    #   to drive a correct write?  Both are observed at runtime from real
    #   forward passes, so they are the right gates for "should the JIT fire".
    # Mechanistic advisory (mechanistic_advisory): phase + affine law residuals.
    #   These describe HOW the model implements the op.  Synthetic-calibrated
    #   thresholds reject real LMs where the helix is present but weaker than
    #   the planted synthetic, even when the write-and-resume operationally
    #   preserves behavior — so they are reported but NOT gates.
    phase_law_holds = phase_residual < phase_law_tolerance
    affine_law_holds = affine_residual < affine_law_tolerance
    decoders_good = acc_a >= MIN_DECODER_ACCURACY and acc_b >= MIN_DECODER_ACCURACY
    kl_acceptable = held_out_kl < KL_THRESHOLD
    fits_kill_criterion = decoders_good and kl_acceptable
    mechanistic_advisory = phase_law_holds and affine_law_holds

    # ── 10. Staged NSJIR program ──────────────────────────────────────────────
    staged_family = _build_staged_family(
        periods=periods_to_verify,
        d_model=d_model,
        layer_construct=layer_construct,
        layer_readout=layer_readout,
        n_layers=n_layers,
    )

    return ClockExtraction(
        staged_family=staged_family,
        R_a=R_a,
        R_b=R_b,
        W_ans=W_ans,
        C_ans_linear=C_ans_linear,
        layer_construct=layer_construct,
        layer_readout=layer_readout,
        phase_law_residual=float(phase_residual),
        affine_law_residual=float(affine_residual),
        held_out_kl=float(held_out_kl),
        fits_kill_criterion=fits_kill_criterion,
        mechanistic_advisory=mechanistic_advisory,
    )


# ─── Private helpers ───────────────────────────────────────────────────────────


def _run_batched(model: nn.Module, tokens: Tensor, batch_size: int = 256) -> None:
    """Run forward passes through *model* in batches to populate hooks."""
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            model(tokens[start : start + batch_size])


def _count_encoder_layers(enc_module: nn.Module) -> int:
    """Count TransformerEncoder-style sub-layers, or return 0 if structure is absent."""
    if hasattr(enc_module, "layers"):
        return len(enc_module.layers)
    return 0


def _resolve_embedding_module(model: nn.Module, embedding_attr: str | None) -> nn.Module | None:
    """Return the embedding nn.Module to hook for output capture, or None.

    Resolution:
      - If ``embedding_attr`` is provided, traverse dotted attribute path
        (e.g. ``"gpt_neox.embed_in"``) and return the module if found.
      - Otherwise, fall back to ``model.token_embedding`` if it exists.

    Returns None when nothing matches; the embedding capture is then skipped
    and operand fitting falls back to the earliest available layer output.
    """
    if embedding_attr is not None:
        target: object = model
        for part in embedding_attr.split("."):
            target = getattr(target, part, None)
            if target is None:
                return None
        return target if isinstance(target, nn.Module) else None
    if hasattr(model, "token_embedding"):
        candidate = model.token_embedding
        return candidate if isinstance(candidate, nn.Module) else None
    return None


def _augmented(h: Tensor) -> tuple[Tensor, Tensor]:
    """Return (h, [h; 1_column]) for the bias-augmented linear system."""
    bias = torch.ones(h.shape[0], 1, dtype=h.dtype)
    return h, torch.cat([h, bias], dim=1)


def _lstsq(A: Tensor, B: Tensor) -> Tensor:
    """Least-squares: argmin_X ||A X - B||_F.  Returns X of shape (A.shape[1], B.shape[1])."""
    result = torch.linalg.lstsq(A, B)
    return result.solution


def _helix_basis_matrix(n_max: int, periods: tuple[int, ...], *, affine: bool = True) -> Tensor:
    """Return (n_max, basis_dim) helix basis matrix for integers 0 .. n_max - 1.

    With affine=True: columns are [n, cos(2πn/T_0), sin(2πn/T_0), cos(2πn/T_1), ...].
    The integers are stored as RAW values (not normalized), consistent with HelixBasis.encode().
    """
    vals = torch.arange(n_max, dtype=torch.float32)
    cols = []
    if affine:
        cols.append(vals.unsqueeze(1))
    for T in periods:
        angle = 2.0 * math.pi * vals / T
        cols.append(torch.cos(angle).unsqueeze(1))
        cols.append(torch.sin(angle).unsqueeze(1))
    return torch.cat(cols, dim=1)


def _decode_integers(pred_basis: Tensor, B: Tensor) -> Tensor:
    """Nearest-neighbour decode: argmin_n ||pred_basis - B[n]||_2.

    pred_basis: (N, basis_dim)
    B: (n_max, basis_dim)
    Returns: (N,) LongTensor of decoded integers.
    """
    dists = torch.cdist(pred_basis.float(), B.float())  # (N, n_max)
    return dists.argmin(dim=1).long()


def _phase_law_quality(
    pred_a: Tensor,
    pred_b: Tensor,
    pred_ans: Tensor,
    periods: tuple[int, ...],
) -> float:
    """Phase-law quality score: average ||ẑ_T^ans - ẑ_T^a · ẑ_T^b||^2 over periods.

    Lower is better (0 = perfect).  Uses DECODED HELIX COORDS from hidden states,
    never ground-truth integers.
    """
    return _phase_law_residual(pred_a, pred_b, pred_ans, periods)


def _phase_law_residual(
    pred_a: Tensor,
    pred_b: Tensor,
    pred_ans: Tensor,
    periods: tuple[int, ...],
) -> float:
    """Compute ||ẑ_T^ans - ẑ_T^a · ẑ_T^b||^2 averaged over periods and samples.

    All inputs are (N, basis_dim) predicted helix coords from fitted decoders.
    Period T occupies columns [1+2*T_idx, 2+2*T_idx] (affine col is index 0).
    """
    total = 0.0
    for T_idx in range(len(periods)):
        cos_col = 1 + 2 * T_idx
        sin_col = 2 + 2 * T_idx
        if sin_col >= pred_a.shape[1]:
            continue
        z_a = pred_a[:, cos_col : sin_col + 1].float()  # (N, 2)
        z_b = pred_b[:, cos_col : sin_col + 1].float()
        z_ans = pred_ans[:, cos_col : sin_col + 1].float()

        z_prod = _complex_product(z_a, z_b)
        # Normalise to unit circle before measuring residual (scale-invariant law)
        z_prod_n = F.normalize(z_prod, dim=1, eps=1e-8)
        z_ans_n = F.normalize(z_ans, dim=1, eps=1e-8)
        residual = ((z_prod_n - z_ans_n) ** 2).sum(dim=1).mean().item()
        total += residual
    n_periods = len(periods)
    return total / n_periods if n_periods > 0 else float("inf")


def _affine_law_residual(pred_a: Tensor, pred_b: Tensor, pred_ans: Tensor) -> float:
    """||û^ans - (û^a + û^b)||^2 where û is the affine coordinate (column 0)."""
    u_a = pred_a[:, 0].float()
    u_b = pred_b[:, 0].float()
    u_ans = pred_ans[:, 0].float()
    return float(((u_ans - (u_a + u_b)) ** 2).mean().item())


def _complex_product(z1: Tensor, z2: Tensor) -> Tensor:
    """Complex multiplication: (cos θ1 + i sin θ1)(cos θ2 + i sin θ2).

    Inputs: (N, 2) as [cos, sin].  Returns (N, 2) as [cos, sin].
    """
    cos12 = z1[:, 0] * z2[:, 0] - z1[:, 1] * z2[:, 1]
    sin12 = z1[:, 0] * z2[:, 1] + z1[:, 1] * z2[:, 0]
    return torch.stack([cos12, sin12], dim=1)


def _pick_operand_source(
    emb_hidden: Tensor | None,
    layer_hiddens: dict[int, Tensor],
) -> Tensor | None:
    """Select the best operand source tensor.

    Preference order:
    1. Token-embedding output (purest helix signal, before attention mixing).
    2. Earliest available encoder layer output.
    3. None if no valid (ndim=3) source exists.
    """
    if emb_hidden is not None and emb_hidden.ndim == 3:
        return emb_hidden
    for i in sorted(layer_hiddens.keys()):
        h = layer_hiddens[i]
        if h.ndim == 3:
            return h
    return None


def _compute_kl_write_and_resume(
    *,
    model: nn.Module,
    enc_module: nn.Module,
    test_tokens: Tensor,
    true_answer: Tensor,
    layer_readout: int,
    n_layers: int,
    B_answer: Tensor,
    C_ans_linear: Tensor,
    W_ans: Tensor,
    pos_ans: int,
    layer_hiddens: dict[int, Tensor],
    resume_fn: Callable[[Tensor, int], Tensor] | None = None,
) -> tuple[float, float]:
    """Run real write-and-resume forward pass and return (held_out_kl, patched_acc).

    Writes helix(s) into the residual at layer_readout and resumes the model from
    layer_readout+1 to get logits.  Compares to original model logits.
    KL is KL(original_probs || patched_probs).
    """
    # Build the patched hidden state from cached captures.
    # We captured the output of each encoder layer during the main activation run.
    # layer_hiddens[layer_readout - 1] is the input to layer_readout (= output of previous layer).
    # If layer_readout == 0, we need the embedding input, which we do not have from these hooks.
    # In that case, use layer_readout=1 as fallback.
    source_layer = layer_readout - 1 if layer_readout > 0 else 0
    resume_from = layer_readout if layer_readout > 0 else 1

    if source_layer not in layer_hiddens:
        # No valid source — return neutral values
        return float("inf"), 0.0

    # Remap: layer_hiddens was captured over ALL operand_tokens; test_tokens
    # is a subset.  We stored samples in order, so test_idx positions map correctly.
    # However, we passed test_tokens directly to this function.
    # We need to re-capture activations for just the test subset.
    # Re-capture by running a fresh forward pass over test_tokens only.

    test_layer_buffers: dict[int, list[Tensor]] = {i: [] for i in range(n_layers)}
    handles: list[Any] = []
    for i in range(n_layers):
        layer_mod = enc_module.layers[i]

        def _make_hook(idx: int):
            def _hook(module: nn.Module, inp: Any, output: Any) -> None:  # noqa: ARG001
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, Tensor):
                    # Cast to float32 for downstream tensor algebra; matches
                    # the activation-capture hook in extract_clock_arithmetic.
                    test_layer_buffers[idx].append(tensor.detach().float().cpu())

            return _hook

        handles.append(layer_mod.register_forward_hook(_make_hook(i)))

    with torch.inference_mode():
        orig_logits_list = []
        for start in range(0, len(test_tokens), 256):
            batch = test_tokens[start : start + 256]
            orig_logits_list.append(model(batch).detach().float())
    for h in handles:
        h.remove()

    orig_logits = torch.cat(orig_logits_list, dim=0)

    test_layer_hiddens: dict[int, Tensor] = {}
    for i in range(n_layers):
        if test_layer_buffers[i]:
            cat_h = torch.cat(test_layer_buffers[i], dim=0)
            if cat_h.ndim == 3:
                test_layer_hiddens[i] = cat_h

    if source_layer not in test_layer_hiddens:
        return float("inf"), 0.0

    h_pre_readout = test_layer_hiddens[source_layer].clone()  # (n_test, seq_len, d_model)

    # Compute helix write delta for the true answer
    target_helix = B_answer[true_answer.long()]  # (n_test, basis_dim)
    h_ans_pos = h_pre_readout[:, pos_ans, :]  # (n_test, d_model)
    current_helix = h_ans_pos @ C_ans_linear  # (n_test, basis_dim)
    # delta_h = (target - current) @ W_ans  where W_ans = pinv(C_ans_linear): (basis_dim, d_model)
    delta_h = (target_helix - current_helix) @ W_ans  # (n_test, d_model)

    h_patched = h_pre_readout.clone()
    h_patched[:, pos_ans, :] = h_ans_pos + delta_h

    # Resume model from resume_from layer.  resume_fn (when provided) handles
    # architectures like Pythia/GPTNeoX whose layer forward signature requires
    # extra arguments (RoPE position_embeddings).  The default chain assumes
    # a vanilla nn.TransformerEncoder + model.final_norm + model.unembed
    # (HelixAddTransformer).
    if resume_fn is not None:
        patched_logits = resume_fn(h_patched, resume_from)
    else:
        patched_logits = _run_from_layer(model, enc_module, h_patched, resume_from, n_layers)

    # KL(original || patched)
    orig_probs = F.softmax(orig_logits, dim=-1)
    patched_probs = F.softmax(patched_logits.float(), dim=-1)
    kl = float(
        F.kl_div(
            patched_probs.log().clamp_min(-1e9), orig_probs, reduction="batchmean"
        ).item()
    )

    # Accuracy of patched model on true answer
    vocab_size = patched_logits.shape[-1]
    clamped = true_answer.long().clamp(0, vocab_size - 1)
    patched_acc = float((patched_logits.argmax(1) == clamped).float().mean().item())

    return max(0.0, kl), patched_acc


def _run_from_layer(
    model: nn.Module,
    enc_module: nn.Module,
    h: Tensor,
    from_layer: int,
    n_layers: int,
) -> Tensor:
    """Run encoder layers from_layer..n_layers-1 then final_norm + unembed.

    This function manually chains the encoder sub-layers to implement write-and-resume
    without requiring a pre-hook.  It assumes the model has:
      - enc_module.layers[i] for i in [from_layer, n_layers)
      - model.final_norm(h[:, -1]) or equivalent
      - model.unembed(...)
    """
    current = h
    with torch.inference_mode():
        for i in range(from_layer, n_layers):
            current = enc_module.layers[i](current)
        # Apply model's final projection
        if hasattr(model, "final_norm") and hasattr(model, "unembed"):
            last_token = current[:, -1, :]
            logits = model.unembed(model.final_norm(last_token))
        else:
            # Fallback: run just the encoder part through the model's full head
            # This should not happen for HelixAddTransformer
            logits = current[:, -1, :]
    return logits


def _build_staged_family(
    *,
    periods: tuple[int, ...],
    d_model: int,
    layer_construct: int,
    layer_readout: int,
    n_layers: int,
) -> StagedMechanismFamily:
    """Construct the StagedMechanismFamily NSJIR object for Clock arithmetic."""
    operand_type = IntRange(0, 99)
    answer_type = IntRange(0, 198)

    transport_stage = MechanismStageContract(
        stage=MechanismStage.TRANSPORT,
        layer_range=("layers.0", f"layers.{layer_construct}"),
        input_type=operand_type,
        output_type=operand_type,
        semantics=call("gather_helix", var("a"), var("b")),
    )
    construct_stage = MechanismStageContract(
        stage=MechanismStage.CONSTRUCT,
        layer_range=(f"layers.{layer_construct}", f"layers.{layer_readout}"),
        input_type=operand_type,
        output_type=answer_type,
        semantics=call(
            "helix_encode",
            var("B_T"),
            call("add_int", var("a"), var("b")),
        ),
    )
    readout_stage = MechanismStageContract(
        stage=MechanismStage.READOUT,
        layer_range=(f"layers.{layer_readout}", f"layers.{n_layers - 1}"),
        input_type=answer_type,
        output_type=answer_type,
        semantics=call("resume_model_from", var("l_readout")),
    )

    basis = HelixBasis(
        periods=periods,
        affine=True,
        input_range=(0, 99),
    )
    semantics = ClockAdd(
        operand_range=(0, 99),
        result_range=(0, 198),
        basis=basis,
    )

    return StagedMechanismFamily(
        id=f"clock_add_{uuid.uuid4().hex[:8]}",
        semantics=semantics,
        stages=(transport_stage, construct_stage, readout_stage),
        stage_interfaces=(operand_type, answer_type),
        realizations=(),
    )


def _make_degenerate_extraction(
    R_a: Tensor,
    R_b: Tensor,
    W_ans: Tensor,
    periods: tuple[int, ...],
    n_layers: int,
) -> ClockExtraction:
    """Return a ClockExtraction with fits_kill_criterion=False for models that lack structure."""
    basis = HelixBasis(periods=periods, affine=True, input_range=(0, 99))
    semantics = ClockAdd(operand_range=(0, 99), result_range=(0, 198), basis=basis)
    transport_stage = MechanismStageContract(
        stage=MechanismStage.TRANSPORT,
        layer_range=("layers.0", "layers.0"),
        input_type=IntRange(0, 99),
        output_type=IntRange(0, 99),
        semantics=call("gather_helix", var("a"), var("b")),
    )
    construct_stage = MechanismStageContract(
        stage=MechanismStage.CONSTRUCT,
        layer_range=("layers.0", "layers.0"),
        input_type=IntRange(0, 99),
        output_type=IntRange(0, 198),
        semantics=call("add_int", var("a"), var("b")),
    )
    readout_stage = MechanismStageContract(
        stage=MechanismStage.READOUT,
        layer_range=("layers.0", "layers.0"),
        input_type=IntRange(0, 198),
        output_type=IntRange(0, 198),
        semantics=call("resume_model_from", var("l_readout")),
    )
    family = StagedMechanismFamily(
        id=f"clock_add_degenerate_{uuid.uuid4().hex[:8]}",
        semantics=semantics,
        stages=(transport_stage, construct_stage, readout_stage),
        stage_interfaces=(IntRange(0, 99), IntRange(0, 198)),
        realizations=(),
    )
    basis_dim = W_ans.shape[0]
    d_model_fallback = W_ans.shape[1]
    C_ans_linear_zero = torch.zeros(d_model_fallback, basis_dim)
    return ClockExtraction(
        staged_family=family,
        R_a=R_a,
        R_b=R_b,
        W_ans=W_ans,
        C_ans_linear=C_ans_linear_zero,
        layer_construct=0,
        layer_readout=min(1, n_layers - 1),
        phase_law_residual=float("inf"),
        affine_law_residual=float("inf"),
        held_out_kl=float("inf"),
        fits_kill_criterion=False,
        mechanistic_advisory=False,
    )
