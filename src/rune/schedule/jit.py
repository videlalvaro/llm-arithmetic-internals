"""Runtime JIT fire-or-fallback router for extracted Clock arithmetic mechanisms.

Composes Lane 2.E's ``ClockExtraction`` with Lane 3.D's ``HelixClockCert`` to run
a per-prompt write-and-resume:

  1. Compute the native answer ``s = a + b`` (integer addition outside the model).
  2. Encode ``s`` into the answer-helix basis.
  3. Capture the residual hidden state at ``layer_readout - 1`` via an output hook.
  4. Compute ``delta_h = (target_helix - current_helix @ C_ans_linear) @ W_ans``.
  5. Write ``delta_h`` into the answer position of the residual and run the
     remaining encoder layers + final_norm + unembed.

The fallback path runs the model normally.

The router is black-box: only ``register_forward_hook`` on
``model.<output_attr>`` and its sub-layers (output capture).  No model.config
introspection, no ``register_forward_pre_hook``, no ``_inputs[0]`` reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rune.extract.clock import ClockExtraction
from rune.schedule.fallback import (
    FallbackPolicy,
    compute_fallback_mask,
)
from rune.schedule.monitor import JitMonitor, MonitorDecision
from rune.verify.phase_alias import HelixClockCert

# Per-prompt batch size for the write-and-resume capture pass.
_DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class JitFireResult:
    """Per-prompt outcome of the JIT fire-or-fallback router."""

    logits: Tensor
    """(N, vocab) — per-prompt logits.  JIT path or fallback path per-row."""

    fired_mask: Tensor
    """(N,) bool — True iff the prompt was routed to the JIT path."""

    raw_logits: Tensor
    """(N, vocab) — raw-model logits on the same prompts, for comparison.
    These are the prompts' logits under the model alone (no write-and-resume)."""

    abstention_rate: float
    """Fraction of prompts routed to the fallback path."""

    jit_accuracy: float
    """Fraction of prompts where the JIT path argmax equals the true ``a + b``."""

    raw_accuracy: float
    """Fraction of prompts where the raw model's argmax equals the true ``a + b``."""

    parity_with_raw: float
    """Fraction of prompts where JIT argmax equals raw model argmax (combined path)."""

    monitor_decisions: tuple[MonitorDecision, ...] | None = None
    """Per-prompt MonitorDecision from JitMonitor, or None when no monitor was used."""

    fallback_reasons: tuple[str, ...] = ()
    """Per-prompt fallback reason string (one per prompt).

    Each entry is one of: 'ok', 'monitor_abstain', 'cert_abstain',
    'post_injection_mismatch', 'decoder_disagreement', 'phase_consistency_fail',
    'kill_criterion_failed'.  Empty tuple when no FallbackPolicy was used (legacy path).
    """


def fire_helix_clock(
    model: nn.Module,
    extraction: ClockExtraction,
    operand_tokens: Tensor,
    *,
    output_attr: str = "encoder",
    cert: HelixClockCert | None = None,
    abstain_on_cert_fail: bool = True,
    answer_position: int = 1,
    answer_vocab: int = 199,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    monitor: JitMonitor | None = None,
    policy: FallbackPolicy | None = None,
) -> JitFireResult:
    """Run the JIT fire-or-fallback router on operand_tokens.

    Args:
        model: The model to JIT.  Must have ``model.<output_attr>.layers`` and
            ``model.final_norm`` + ``model.unembed`` (matches HelixAddTransformer).
        extraction: From ``extract_clock_arithmetic``.  Must satisfy
            ``extraction.fits_kill_criterion``; otherwise the router refuses to
            fire and falls back unconditionally.
        operand_tokens: (N, 2) integer tensor of (a, b) operand pairs.
        cert: Optional ``HelixClockCert``.  When supplied AND
            ``abstain_on_cert_fail`` is True, prompts whose ``per_prompt_certs``
            entry has ``passes=False`` take the fallback path.
        answer_position: Sequence position where the answer helix is written.
            Defaults to 1 (the second token's residual) — matches the
            HelixAddTransformer convention where the readout reads pos=-1.
        answer_vocab: Size of the answer vocabulary.  199 for HelixAdd over
            [0,99]² → [0,198].
        monitor: Optional ``JitMonitor`` (Lane 4.D).  When supplied, called
            once per batch to produce per-prompt ``MonitorDecision``s that are
            fed into the FallbackPolicy.
        policy: Optional ``FallbackPolicy`` (Lane 4.F).  When supplied, the
            monitor + cert signals are combined via ``route_prompt``.  When None
            (legacy), the original cert-only abstention logic is used, preserving
            backward compatibility.

    Returns:
        ``JitFireResult`` with per-prompt logits, fired_mask, and the
        raw/jit accuracy comparison.  Also includes ``monitor_decisions`` and
        ``fallback_reasons`` when a policy is supplied.
    """
    n = operand_tokens.shape[0]
    model.eval()
    enc_module = getattr(model, output_attr)
    n_layers = len(enc_module.layers)

    # ── Monitor path: Lane 4.D + Lane 4.F ─────────────────────────────────────
    # When both monitor and policy are supplied, use the full fallback policy.
    # This populates fallback_reasons and monitor_decisions.
    result_monitor_decisions: tuple[MonitorDecision, ...] | None = None
    result_fallback_reasons: tuple[str, ...] = ()

    if monitor is not None and policy is not None:
        # Run the monitor to get per-prompt MonitorDecision.
        trace = monitor.classify(model, operand_tokens)
        result_monitor_decisions = trace.decisions

        # Build the per-prompt cert lookup aligned by prompt index.
        per_prompt_certs = (
            cert.per_prompt_certs if (cert is not None and abstain_on_cert_fail) else None
        )

        fallback_decisions, fallback_mask = compute_fallback_mask(
            policy=policy,
            monitor_decisions=result_monitor_decisions,
            per_prompt_certs=per_prompt_certs,
            kill_criterion=extraction.fits_kill_criterion,
            n=n,
        )
        result_fallback_reasons = tuple(d.reason for d in fallback_decisions)

    else:
        # ── Legacy cert-only path (backward-compatible) ────────────────────────
        # Per-prompt fallback mask: prompts that take the fallback path.
        fallback_mask = torch.zeros(n, dtype=torch.bool)

        # Refuse to fire if the extraction failed the kill criterion.
        if not extraction.fits_kill_criterion:
            fallback_mask[:] = True

        # Cert-based abstention.
        if cert is not None and abstain_on_cert_fail and not fallback_mask.all():
            cert_pass = torch.zeros(n, dtype=torch.bool)
            for prompt in cert.per_prompt_certs:
                pid = int(prompt.prompt_id)
                if 0 <= pid < n:
                    cert_pass[pid] = bool(prompt.passes)
            # If cert covers fewer than n prompts (e.g. cert was fit on a subset),
            # un-covered prompts default to fallback.
            fallback_mask = fallback_mask | (~cert_pass)

    fired_mask = ~fallback_mask

    # Compute raw-model logits for all prompts (used both as fallback output
    # and as comparison baseline).
    raw_logits = _run_raw_model(model, operand_tokens, batch_size)

    # Default: output = raw_logits everywhere.  We overwrite JIT-fired rows below.
    out_logits = raw_logits.clone()

    if fired_mask.any():
        fire_idx = fired_mask.nonzero(as_tuple=False).flatten()
        fire_tokens = operand_tokens[fire_idx]
        jit_logits = _jit_write_and_resume(
            model=model,
            enc_module=enc_module,
            extraction=extraction,
            fire_tokens=fire_tokens,
            answer_position=answer_position,
            answer_vocab=answer_vocab,
            n_layers=n_layers,
            batch_size=batch_size,
        )
        out_logits[fire_idx] = jit_logits

    # Metrics.
    true_answer = (operand_tokens[:, 0] + operand_tokens[:, 1]).clamp(0, answer_vocab - 1)
    out_argmax = out_logits.argmax(dim=-1)
    raw_argmax = raw_logits.argmax(dim=-1)

    return JitFireResult(
        logits=out_logits,
        fired_mask=fired_mask,
        raw_logits=raw_logits,
        abstention_rate=float(fallback_mask.float().mean().item()),
        jit_accuracy=float((out_argmax == true_answer).float().mean().item()),
        raw_accuracy=float((raw_argmax == true_answer).float().mean().item()),
        parity_with_raw=float((out_argmax == raw_argmax).float().mean().item()),
        monitor_decisions=result_monitor_decisions,
        fallback_reasons=result_fallback_reasons,
    )


def _run_raw_model(model: nn.Module, tokens: Tensor, batch_size: int) -> Tensor:
    """Run the raw model on tokens; return (N, vocab) logits."""
    out: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            out.append(model(tokens[start : start + batch_size]).detach().float())
    return torch.cat(out, dim=0)


def _jit_write_and_resume(
    *,
    model: nn.Module,
    enc_module: nn.Module,
    extraction: ClockExtraction,
    fire_tokens: Tensor,
    answer_position: int,
    answer_vocab: int,
    n_layers: int,
    batch_size: int,
) -> Tensor:
    """Capture residual at layer_readout-1, write helix(a+b), resume from layer_readout."""
    layer_readout = extraction.layer_readout
    source_layer = layer_readout - 1 if layer_readout > 0 else 0
    resume_from = layer_readout if layer_readout > 0 else 1

    # Hook layer outputs.
    layer_buffer: list[Tensor] = []

    def _hook(_m: nn.Module, _inp: Any, output: Tensor) -> None:
        layer_buffer.append(output.detach().cpu())

    handle = enc_module.layers[source_layer].register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            for start in range(0, len(fire_tokens), batch_size):
                model(fire_tokens[start : start + batch_size])
    finally:
        handle.remove()

    h_pre_readout = torch.cat(layer_buffer, dim=0)  # (n_fire, seq, d_model)

    # Native a+b → answer-helix basis row.
    a_vals = fire_tokens[:, 0].long()
    b_vals = fire_tokens[:, 1].long()
    s = (a_vals + b_vals).clamp(0, answer_vocab - 1)
    periods = _periods_from_basis_dim(extraction.W_ans.shape[0])
    B_answer = _helix_basis_matrix(answer_vocab, periods, affine=True)
    target_helix = B_answer[s]  # (n_fire, basis_dim)

    # delta_h = (target - current) @ W_ans
    h_ans = h_pre_readout[:, answer_position, :]  # (n_fire, d_model)
    current_helix = h_ans @ extraction.C_ans_linear  # (n_fire, basis_dim)
    delta_h = (target_helix - current_helix) @ extraction.W_ans  # (n_fire, d_model)

    h_patched = h_pre_readout.clone()
    h_patched[:, answer_position, :] = h_ans + delta_h

    # Resume.
    return _run_from_layer(model, enc_module, h_patched, resume_from, n_layers)


def _run_from_layer(
    model: nn.Module,
    enc_module: nn.Module,
    h: Tensor,
    from_layer: int,
    n_layers: int,
) -> Tensor:
    """Run encoder layers from_layer..n_layers-1 then final_norm + unembed."""
    current = h
    with torch.inference_mode():
        for i in range(from_layer, n_layers):
            current = enc_module.layers[i](current)
        if hasattr(model, "final_norm") and hasattr(model, "unembed"):
            last_token = current[:, -1, :]
            logits = model.unembed(model.final_norm(last_token))
        else:
            logits = current[:, -1, :]
    return logits.detach().float()


def _periods_from_basis_dim(basis_dim: int) -> tuple[int, ...]:
    """Recover the period tuple from the basis dimension.

    Basis layout is [affine, cos(2π/T_0), sin(2π/T_0), cos(2π/T_1), ...], so
    n_periods = (basis_dim - 1) / 2.  The actual period values are not stored
    in the W_ans matrix; we rely on the matching B_answer construction in the
    extractor.  For HelixAddTransformer the canonical set is (2, 5, 10, 100).
    """
    n_periods = (basis_dim - 1) // 2
    canonical = (2, 5, 10, 100)
    if n_periods == len(canonical):
        return canonical
    # Fallback: small primes + 100.  Matches the Lane 2.E default.
    candidates = (2, 3, 5, 7, 10, 100)
    return candidates[:n_periods]


def _helix_basis_matrix(
    n_max: int, periods: tuple[int, ...], *, affine: bool = True
) -> Tensor:
    """Return (n_max, basis_dim) helix basis matrix for integers 0..n_max-1.

    Layout matches Lane 2.E's ``_helix_basis_matrix``: affine column carries
    the RAW integer value; phase columns are cos/sin pairs per period.
    """
    vals = torch.arange(n_max, dtype=torch.float32)
    cols: list[Tensor] = []
    if affine:
        cols.append(vals.unsqueeze(1))
    for period in periods:
        angle = 2.0 * math.pi * vals / float(period)
        cols.append(torch.cos(angle).unsqueeze(1))
        cols.append(torch.sin(angle).unsqueeze(1))
    return torch.cat(cols, dim=1)


__all__ = ["JitFireResult", "fire_helix_clock"]
