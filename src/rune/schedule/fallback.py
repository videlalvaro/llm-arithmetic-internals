"""Lane 4.F — Fallback logic: stateless policy combining runtime gates.

Combines signals from the Lane 4.D Monitor (JitMonitor / MonitorDecision) and
Lane 3.D offline cert (PromptAliasCert) to produce a per-prompt JIT-or-raw decision.

Design notes:
  - ``route_prompt`` is a pure function of its arguments — no model calls, no state.
  - ``post_injection_check`` is a pure function — no model calls.
  - All numeric thresholds are module-level named constants with calibration docstrings.

Forbidden patterns (enforced by the anti-cheat audit in the scheduler tests):
  - model.config, *Config(...), .modulus, .periods (model attribute)
  - token_embedding.weight
  - register_forward_pre_hook
  - Anonymous magic numbers (all thresholds are named constants)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rune.schedule.monitor import MonitorDecision
from rune.verify.phase_alias import PromptAliasCert

# ---------------------------------------------------------------------------
# Named calibration constants — every threshold has a documented calibration note
# ---------------------------------------------------------------------------

_DISAGREEMENT_KL_MAX: float = 0.5
"""Default KL-divergence threshold for post-injection disagreement fallback.

post_injection_check returns False (disagree → fall back) when
KL(raw_logits || jit_logits) > _DISAGREEMENT_KL_MAX.

Calibration: KL(p || q) between two distributions that agree on the argmax but
differ in probability mass on neighbouring tokens averages ~0.08 for the
HelixAddTransformer JIT path (where the write-and-resume is accurate).  A threshold
of 0.5 is intentionally permissive: it catches large distributional mismatches
(e.g. the JIT put a large spike on the wrong class) while allowing small
probability-mass differences that don't affect the argmax.

A value of 0.0 would reject any logit difference; a value of +inf would never reject.
For safety-critical applications, lower this to 0.1 or measure the mean KL on a
validation set and set the threshold at mean + 3 * std."""

_PHASE_CONSISTENCY_MIN: float = 0.05
"""Minimum alias margin required from PromptAliasCert for phase-consistency gate.

If ANY entry in ``PromptAliasCert.alias_margins`` is below this threshold and the
offset is not ±inf (i.e. the alias exists within the valid range), the fallback
fires with reason='phase_consistency_fail'.

This directly implements the PLAN.md bullet:
  "Phase-consistency fallback: affine says s but periods vote s±10/±100 → abstain"

Calibration: matches ``_MIN_ALIAS_MARGIN`` from phase_alias.py (0.05).  Prompts
where any alias margin is below 0.05 have two helix-energy candidates within 0.05
energy units of each other — insufficient separation for a safe JIT write."""

_QUORUM_AGREEMENT_THRESHOLD: float = 1.0
"""Minimum quorum_agreement from MonitorDecision for the disagreement fallback.

When extraction has K > 1 realizations, MonitorDecision.quorum_agreement measures
the fraction of realization pairs agreeing on the decoded answer.  A value below
this threshold triggers reason='decoder_disagreement'.

Calibration: 1.0 = strict (all realizations agree).  For the current HelixAdd
single-realization extraction, quorum_agreement is always 1.0, so this gate is
a no-op.  It activates when a multi-realization extraction is used (Lane 4.G)."""


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FallbackDecision:
    """Per-prompt fallback routing decision."""

    fire_jit: bool
    """True iff this prompt should be routed to the JIT path."""

    reason: str
    """One of:
      'ok'                         — all gates passed; JIT fires.
      'monitor_abstain'            — monitor.fire=False (manifold_fit or overlap_risk gate).
      'cert_abstain'               — offline cert passes=False.
      'post_injection_mismatch'    — KL(raw || jit) > threshold after injection.
      'decoder_disagreement'       — realization quorum_agreement < threshold.
      'phase_consistency_fail'     — alias margin below threshold in PromptAliasCert.
      'kill_criterion_failed'      — extraction did not pass kill criterion.
    """

    monitor_decision: MonitorDecision | None
    """The MonitorDecision used to make this routing decision (None if no monitor)."""

    cert_passes: bool | None
    """Whether the per-prompt cert passed (None if no cert was provided)."""


@dataclass(frozen=True)
class FallbackPolicy:
    """Stateless policy: given runtime signals, decide JIT-or-raw.

    All fields are named constants.  Construct once and reuse across prompts.
    """

    use_monitor: bool = True
    """If True and a MonitorDecision is provided, gate on monitor.fire."""

    use_offline_cert: bool = True
    """If True and a PromptAliasCert is provided, gate on cert.passes."""

    use_post_injection_check: bool = True
    """Reserved: set True to enable post-injection KL check when raw/jit logits
    are both available.  The post_injection_check function is always callable
    independently of this flag; the flag only affects route_prompt behaviour when
    logits are not available at routing time."""

    disagreement_kl_max: float = _DISAGREEMENT_KL_MAX
    """KL threshold for post_injection_check.  Named constant with calibration doc."""

    phase_consistency_min: float = _PHASE_CONSISTENCY_MIN
    """Minimum alias margin for the phase-consistency gate.  Named constant."""

    quorum_agreement_min: float = _QUORUM_AGREEMENT_THRESHOLD
    """Minimum quorum_agreement for the decoder-disagreement gate."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route_prompt(
    policy: FallbackPolicy,
    monitor_decision: MonitorDecision | None,
    per_prompt_cert: PromptAliasCert | None,
    kill_criterion: bool,
) -> FallbackDecision:
    """Pure-function policy: combine gate signals and return a routing decision.

    Gate evaluation order (cheapest / most informative first):

    1. kill_criterion_failed  — extraction-level gate; never fire if extraction is bad.
    2. decoder_disagreement   — realization quorum agreement below threshold.
    3. monitor_abstain        — monitor.fire=False.
    4. cert_abstain           — offline cert passes=False.
    5. phase_consistency_fail — any alias margin below threshold.
    6. ok                     — all gates passed.

    post_injection_check (reason='post_injection_mismatch') is a separate function
    called AFTER the JIT fires; it is not evaluated here because logits are not
    available at routing time.

    Parameters
    ----------
    policy:
        FallbackPolicy controlling which gates are active and their thresholds.
    monitor_decision:
        Per-prompt MonitorDecision from JitMonitor.classify().  May be None when
        no monitor is attached.
    per_prompt_cert:
        Per-prompt PromptAliasCert from certify_helix_clock().  May be None when
        no offline cert is available.
    kill_criterion:
        True iff the ClockExtraction passed its kill criterion
        (extraction.fits_kill_criterion).

    Returns
    -------
    FallbackDecision
        fire_jit=True iff all active gates pass.
    """
    cert_passes: bool | None = None
    if per_prompt_cert is not None:
        cert_passes = bool(per_prompt_cert.passes)

    # Gate 1: extraction kill criterion
    if not kill_criterion:
        return FallbackDecision(
            fire_jit=False,
            reason="kill_criterion_failed",
            monitor_decision=monitor_decision,
            cert_passes=cert_passes,
        )

    # Gate 2: decoder disagreement (multi-realization quorum)
    if monitor_decision is not None and policy.use_monitor:
        if monitor_decision.quorum_agreement < policy.quorum_agreement_min:
            return FallbackDecision(
                fire_jit=False,
                reason="decoder_disagreement",
                monitor_decision=monitor_decision,
                cert_passes=cert_passes,
            )

    # Gate 3: monitor abstain (manifold_fit or overlap_risk)
    if monitor_decision is not None and policy.use_monitor:
        if not monitor_decision.fire:
            return FallbackDecision(
                fire_jit=False,
                reason="monitor_abstain",
                monitor_decision=monitor_decision,
                cert_passes=cert_passes,
            )

    # Gate 4: offline cert
    if per_prompt_cert is not None and policy.use_offline_cert:
        if not per_prompt_cert.passes:
            return FallbackDecision(
                fire_jit=False,
                reason="cert_abstain",
                monitor_decision=monitor_decision,
                cert_passes=cert_passes,
            )

    # Gate 5: phase consistency (alias margin check)
    if per_prompt_cert is not None and policy.use_offline_cert:
        for _delta, margin in per_prompt_cert.alias_margins.items():
            if not math.isinf(margin) and margin < policy.phase_consistency_min:
                return FallbackDecision(
                    fire_jit=False,
                    reason="phase_consistency_fail",
                    monitor_decision=monitor_decision,
                    cert_passes=cert_passes,
                )

    # All gates passed.
    return FallbackDecision(
        fire_jit=True,
        reason="ok",
        monitor_decision=monitor_decision,
        cert_passes=cert_passes,
    )


def post_injection_check(
    raw_logits: Tensor,
    jit_logits: Tensor,
    *,
    disagreement_kl_max: float = _DISAGREEMENT_KL_MAX,
) -> bool:
    """Check whether JIT and raw logits agree sufficiently.

    Computes KL(raw_probs || jit_probs) per prompt and returns True (passes, JIT is
    consistent) if ALL prompts have KL below the threshold.

    Returns True iff KL(raw || jit) ≤ disagreement_kl_max for every prompt.

    This is a per-batch check: if ANY prompt's KL exceeds the threshold, the whole
    batch is flagged.  For per-prompt routing, call this once per prompt.

    Parameters
    ----------
    raw_logits:
        (N, vocab) raw-model logits.
    jit_logits:
        (N, vocab) JIT-path logits.
    disagreement_kl_max:
        KL threshold.  Uses module-level _DISAGREEMENT_KL_MAX by default.

    Returns
    -------
    bool
        True iff the check passes (KL below threshold for all prompts).
        False iff any prompt's KL exceeds the threshold — caller should fall back.

    Notes
    -----
    KL(raw || jit) is asymmetric: it measures how much information is needed to
    encode samples from raw using jit's distribution.  We use this direction because:
    - raw_logits are from the model's own forward pass (reference distribution).
    - jit_logits are the modified output (approximate distribution).
    - High KL means the JIT path diverged significantly from the model's natural output.
    """
    raw_logits_f = raw_logits.float()
    jit_logits_f = jit_logits.float()

    raw_log_probs = F.log_softmax(raw_logits_f, dim=-1)
    jit_log_probs = F.log_softmax(jit_logits_f, dim=-1)

    # KL(raw || jit) = sum_v  raw_prob[v] * (log_raw[v] - log_jit[v])
    # F.kl_div(input=log_q, target=p) computes KL(p || q) when reduction='none'
    # and returns per-element terms.  Summing over vocab gives per-prompt KL.
    # Here: p = raw, q = jit.
    kl_per_token = F.kl_div(
        input=jit_log_probs,
        target=raw_log_probs.exp(),
        reduction="none",
    )  # (N, vocab)

    kl_per_prompt = kl_per_token.sum(dim=-1)  # (N,)
    return bool((kl_per_prompt <= disagreement_kl_max).all().item())


# ---------------------------------------------------------------------------
# Vectorised batch helper (used by fire_helix_clock)
# ---------------------------------------------------------------------------


def compute_fallback_mask(
    policy: FallbackPolicy,
    monitor_decisions: tuple[MonitorDecision, ...] | None,
    per_prompt_certs: tuple[PromptAliasCert, ...] | None,
    kill_criterion: bool,
    n: int,
) -> tuple[list[FallbackDecision], torch.Tensor]:
    """Vectorised wrapper: compute per-prompt FallbackDecision for a batch of N prompts.

    Parameters
    ----------
    policy:
        FallbackPolicy.
    monitor_decisions:
        Length-N tuple of MonitorDecision (one per prompt), or None.
    per_prompt_certs:
        Length-N tuple of PromptAliasCert (one per prompt), or None.
        The cert tuple must be aligned by prompt index (cert[i] covers prompt i).
    kill_criterion:
        extraction.fits_kill_criterion.
    n:
        Total number of prompts.

    Returns
    -------
    (decisions, fallback_mask_tensor)
        decisions: list of length-N FallbackDecision.
        fallback_mask_tensor: (N,) bool tensor; True = fall back, False = fire JIT.
    """
    import torch

    # Build per-prompt cert lookup: cert[i] = PromptAliasCert for prompt_id == i.
    cert_by_idx: dict[int, PromptAliasCert] = {}
    if per_prompt_certs is not None:
        for c in per_prompt_certs:
            pid = int(c.prompt_id)
            if 0 <= pid < n:
                cert_by_idx[pid] = c

    decisions: list[FallbackDecision] = []
    fallback_list: list[bool] = []

    for i in range(n):
        md = monitor_decisions[i] if monitor_decisions is not None else None
        cert = cert_by_idx.get(i)
        dec = route_prompt(
            policy=policy,
            monitor_decision=md,
            per_prompt_cert=cert,
            kill_criterion=kill_criterion,
        )
        decisions.append(dec)
        fallback_list.append(not dec.fire_jit)

    fallback_mask = torch.tensor(fallback_list, dtype=torch.bool)
    return decisions, fallback_mask


__all__ = [
    "FallbackDecision",
    "FallbackPolicy",
    "route_prompt",
    "post_injection_check",
    "compute_fallback_mask",
]
