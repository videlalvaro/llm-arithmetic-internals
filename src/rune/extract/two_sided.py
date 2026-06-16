"""Two-sided helix extraction — TODO 29.

Composes the operand-side ClockExtraction (existing) with a readout-side
TokenHelixDecoder (new) to produce a write basis that lives in the readout's
phase frame, not the operand's.

Background
----------
``extract_clock_arithmetic`` fits C_ans_linear from hidden states at the answer
position: ``h_ans @ C_ans_linear ≈ B_helix(a+b)``.  The write target is in the
*operand-side* phase convention — the same basis used to decode hidden states.

But ``lm_head`` / ``embed_out`` may decode in a different phase convention,
possibly at different operational periods.  The ``docs/pythia_2.8b_nonlinear_readout.md``
probe found composite R²=0.65 for Nanda Eq. (1) but 0% argmax recovery —
the readout aggregates a character bank but at phases offset from B[n].

Two candidate write strategies:

  (a) "readout-pinv" — write ``h_target[n] = pinv(W_token) @ B_token[n]``.
      This finds a d_model vector that, when passed through lm_head, produces
      a logit vector approximating the Nanda-form output for integer n.
      ``pinv(W_token)`` here is shape (d_model, 2K+1): maps character coords
      → residual space.

  (b) "ans-write-rotated" — align the operand-side write with the readout-side
      phase convention by finding, per period, a 2D rotation R_T that maps
      B_operand's (cos, sin) pair to B_token's (cos, sin) convention for the
      same integer value, then write via the rotated encoder.

Both strategies are evaluated; the one with higher argmax_recovery on a
validation fold is selected.

Cheat audit (relaxed)
---------------------
- ✅ Only register_forward_hook for operand-side capture.
- ❌ lm_head.weight IS READ for readout-side extraction (intentional; see
     TokenHelixDecoder docstring).
- ✅ No model.config reads, no embed_in.weight reads.
- ✅ All patched continuations route through real model layers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rune.extract.clock import (
    ClockExtraction,
    _helix_basis_matrix,
    extract_clock_arithmetic,
)
from rune.extract.token_helix import TokenHelixDecoder, extract_token_helix

# ─── Public result type ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TwoSidedClockExtraction:
    """ClockExtraction + TokenHelixDecoder + readout-aligned write basis.

    Attributes
    ----------
    operand_side
        The existing ClockExtraction from ``extract_clock_arithmetic``.
    readout_side
        TokenHelixDecoder fitted from ``lm_head.weight``.
    write_target_fn
        Callable ``(n: int, h_current: Tensor) -> Tensor`` that returns a
        d_model residual vector ``h_target`` such that writing it at the
        answer position and resuming the model is expected to produce peak
        logit at ``token_id(n)``.  ``h_current`` is the pre-write residual
        at the answer position (shape ``(d_model,)``).  Non-projection
        strategies ignore ``h_current``.
    method
        ``"readout-pinv"`` | ``"ans-write-rotated"`` |
        ``"readout-pinv-projection"`` | ``"failed_validation"``
        — identifies which composition strategy was selected (or that none
        exceeded the 0.30 validation threshold).
    argmax_recovery_validation
        Fraction of validation pairs where argmax of model output after the
        two-sided write equals the correct answer token.
    argmax_recovery_by_strategy
        Dict mapping strategy name → validation argmax_recovery (all
        strategies are always evaluated and stored).
    write_fns_by_strategy
        Dict mapping strategy name → write_target_fn callable.  Lets callers
        evaluate any of the three strategies on the full fold (not just the
        winner) without rebuilding the extraction.
    """

    operand_side: ClockExtraction
    readout_side: TokenHelixDecoder
    write_target_fn: Callable[[int, Tensor], Tensor]
    method: str
    argmax_recovery_validation: float
    argmax_recovery_by_strategy: dict[str, float]
    write_fns_by_strategy: dict[str, Callable[[int, Tensor], Tensor]]


# ─── Main public function ──────────────────────────────────────────────────────


def extract_clock_two_sided(
    model: nn.Module,
    operand_tokens: Tensor,
    *,
    output_attr: str,
    operand_positions: tuple[int, int],
    answer_position: int,
    operand_value_range: tuple[int, int],
    answer_value_range: tuple[int, int],
    periods_to_verify: tuple[int, ...],
    resume_fn: Callable[[Tensor, int], Tensor],
    embedding_attr: str | None = None,
    lm_head_attr: str = "lm_head",
    integer_token_ids: Sequence[int],
    seed: int = 0,
) -> TwoSidedClockExtraction:
    """Extract a two-sided helix basis aligned to the readout phase convention.

    Parameters
    ----------
    model
        The language model.  For GPTNeoX (Pythia), the wrapper exposing
        ``gpt_neox`` and ``embed_out``.
    operand_tokens
        ``(N, 2)`` integer operand pairs used for Lane 2.E extraction.
    output_attr
        Backbone attribute name (e.g. ``"gpt_neox"``).
    operand_positions
        ``(pos_a, pos_b)`` in the sequence.
    answer_position
        Sequence position where the answer prediction lives.
    operand_value_range
        ``(lo, hi)`` for operands a, b.
    answer_value_range
        ``(lo, hi)`` for the sum a+b.
    periods_to_verify
        Helix periods for the operand-side extraction (and shared with
        the readout-side extraction).
    resume_fn
        Callable ``(h_patched, from_layer) -> logits`` for write-and-resume.
    embedding_attr
        Override for the embedding hook target.
    lm_head_attr
        Dotted attribute path to the lm_head module on ``model``
        (e.g. ``"embed_out"`` for Pythia, ``"lm_head"`` for Llama).
    integer_token_ids
        BPE token IDs for integers ``[lo, hi]`` in order.
    seed
        Random seed (passed to both operand-side and readout-side extractors,
        and to the validation fold sampler).

    Returns
    -------
    TwoSidedClockExtraction
    """
    # ── 1. Operand-side ClockExtraction ───────────────────────────────────────
    operand_side = extract_clock_arithmetic(
        model,
        operand_tokens,
        output_attr=output_attr,
        operand_positions=operand_positions,
        answer_position=answer_position,
        operand_value_range=operand_value_range,
        answer_value_range=answer_value_range,
        periods_to_verify=periods_to_verify,
        resume_fn=resume_fn,
        embedding_attr=embedding_attr,
        seed=seed,
    )

    # ── 2. Readout-side TokenHelixDecoder ─────────────────────────────────────
    lm_head_module = _resolve_module(model, lm_head_attr)
    readout_side = extract_token_helix(
        lm_head_module,
        integer_token_ids,
        answer_value_range,
        periods_to_verify,
        holdout_frac=0.2,
        seed=seed,
    )

    # ── 3. Build both write strategies ────────────────────────────────────────
    ans_lo, ans_hi = answer_value_range
    n_candidates = ans_hi - ans_lo + 1

    # Strategy (a): readout-pinv — full residual replacement
    # h_target[n] = pinv(W_token) @ B_token[n]
    # W_token: (2K+1, d_model), pinv: (d_model, 2K+1)
    W_token = readout_side.W_token.float()  # (2K+1, d_model)
    # pinv of W_token maps from character space to residual space
    W_token_pinv = torch.linalg.pinv(W_token, rtol=1e-3)  # (d_model, 2K+1)
    B_token = readout_side.B_token.float()  # (V_int, 2K+1)

    def write_readout_pinv(n: int, _h_current: Tensor) -> Tensor:  # noqa: ARG001
        # n is an absolute integer value in [ans_lo, ans_hi]
        c = n - ans_lo
        c_clamped = max(0, min(n_candidates - 1, c))
        b_n = B_token[c_clamped]  # (2K+1,)
        h_target = W_token_pinv @ b_n  # (d_model,)
        return h_target.float()

    # Strategy (b): ans-write-rotated
    # Find per-period rotation R_T that maps operand-side helix basis to
    # readout-side phase convention, then write via rotated B_n via W_ans.
    B_answer = _helix_basis_matrix(  # (n_candidates, 2K+1)
        n_candidates, periods_to_verify, affine=True
    )

    # For each period T, find R_T (2x2 rotation) such that
    # B_token[:, cos_col:sin_col+1] ≈ B_answer[:, cos_col:sin_col+1] @ R_T
    # using least squares across the answer integer range.
    rotations: dict[int, Tensor] = {}
    for T_idx, T in enumerate(periods_to_verify):
        cos_col = 1 + 2 * T_idx
        sin_col = 2 + 2 * T_idx
        if sin_col >= B_answer.shape[1]:
            rotations[T] = torch.eye(2)
            continue
        # Source: operand-side coordinates for integers in [ans_lo, ans_hi]
        src = B_answer[:, cos_col : sin_col + 1]  # (n_candidates, 2)
        # Target: token-side coordinates
        tgt = B_token[:, cos_col : sin_col + 1]  # (n_candidates, 2)
        # Least-squares rotation: R_T = argmin ||src @ R - tgt||_F
        # Solution via SVD of tgt.T @ src
        try:
            U, _, Vh = torch.linalg.svd(tgt.T @ src)
            R_T = U @ Vh  # (2, 2) — closest rotation (Procrustes)
        except Exception:
            R_T = torch.eye(2)
        rotations[T] = R_T.float()

    def write_rotated(n: int, _h_current: Tensor) -> Tensor:  # noqa: ARG001
        # Build rotated B_n: apply per-period rotation to B_answer[n - ans_lo]
        c = max(0, min(n_candidates - 1, n - ans_lo))
        b_n = B_answer[c].clone()  # (2K+1,)
        for T_idx, T in enumerate(periods_to_verify):
            cos_col = 1 + 2 * T_idx
            sin_col = 2 + 2 * T_idx
            if sin_col >= len(b_n):
                continue
            xy = b_n[cos_col : sin_col + 1]  # (2,)
            R_T = rotations[T]
            b_n[cos_col : sin_col + 1] = R_T @ xy
        # Write: h_target = pinv(C_ans_linear) @ b_n_rotated
        W_ans = operand_side.W_ans.float()  # (2K+1, d_model)
        h_target = b_n @ W_ans  # (d_model,)
        return h_target.float()

    # Strategy (c): readout-pinv-projection — projection-style write
    # Preserves the orthogonal complement of the readout-character subspace
    # (linguistic context, digit features, sibling mechanisms) while replacing
    # only the character coords with the correct ones for integer n.
    #
    # Computation:
    #   P_readout = pinv(W_token) @ W_token          (d_model, d_model), rank 2K+1
    #   h_orth    = h_current - P_readout @ h_current  (orthogonal complement)
    #   h_chars   = pinv(W_token) @ B_token[n, :]      (desired character contribution)
    #   h_new     = h_orth + h_chars
    #
    # Sanity check: W_token @ h_new = W_token @ h_chars = B_token[n, :] (exactly),
    # so lm_head sees the correct character coords AND the orthogonal content survives.
    P_readout = W_token_pinv @ W_token  # (d_model, d_model), rank 2K+1
    I_d = torch.eye(P_readout.shape[0], dtype=P_readout.dtype)
    P_orth = I_d - P_readout  # projector onto the orthogonal complement

    def write_readout_pinv_projection(n: int, h_current: Tensor) -> Tensor:
        # h_current: (d_model,) — pre-write residual at pos_ans
        c = n - ans_lo
        c_clamped = max(0, min(n_candidates - 1, c))
        b_n = B_token[c_clamped]  # (2K+1,)
        h_chars = W_token_pinv @ b_n  # (d_model,) — desired character contribution
        h_orth = P_orth @ h_current.float()  # (d_model,) — orthogonal complement
        return (h_orth + h_chars).float()

    # ── 4. Validate all three strategies on a held-out 20-pair fold ──────────
    rng = torch.Generator().manual_seed(seed + 1337)
    n_total = operand_tokens.shape[0]
    n_val = min(20, n_total)
    val_idx = torch.randperm(n_total, generator=rng)[:n_val]
    val_tokens = operand_tokens[val_idx]

    results_by_strategy = _validate_strategies(
        model=model,
        val_tokens=val_tokens,
        operand_side=operand_side,
        resume_fn=resume_fn,
        answer_position=answer_position,
        answer_value_range=answer_value_range,
        integer_token_ids=integer_token_ids,
        strategies={
            "readout-pinv": write_readout_pinv,
            "ans-write-rotated": write_rotated,
            "readout-pinv-projection": write_readout_pinv_projection,
        },
    )

    # Pick the better strategy (projection eligible)
    best_method = max(results_by_strategy, key=lambda k: results_by_strategy[k])
    best_recovery = results_by_strategy[best_method]

    if best_recovery < 0.30:
        best_method = "failed_validation"

    final_fn: Callable[[int, Tensor], Tensor]
    if best_method == "readout-pinv-projection":
        final_fn = write_readout_pinv_projection
    elif best_method == "ans-write-rotated":
        final_fn = write_rotated
    else:
        # readout-pinv or failed_validation — default to readout-pinv
        final_fn = write_readout_pinv

    write_fns_by_strategy = {
        "readout-pinv": write_readout_pinv,
        "ans-write-rotated": write_rotated,
        "readout-pinv-projection": write_readout_pinv_projection,
    }

    return TwoSidedClockExtraction(
        operand_side=operand_side,
        readout_side=readout_side,
        write_target_fn=final_fn,
        method=best_method,
        argmax_recovery_validation=best_recovery,
        argmax_recovery_by_strategy=results_by_strategy,
        write_fns_by_strategy=write_fns_by_strategy,
    )


# ─── Private helpers ───────────────────────────────────────────────────────────


def _resolve_module(model: nn.Module, attr: str) -> nn.Module:
    """Traverse dotted attr path and return the nn.Module."""
    target: Any = model
    for part in attr.split("."):
        target = getattr(target, part, None)
        if target is None:
            raise AttributeError(
                f"_resolve_module: attribute '{part}' not found on "
                f"{type(target).__name__} when resolving '{attr}'."
            )
    if not isinstance(target, nn.Module):
        raise TypeError(
            f"_resolve_module: resolved '{attr}' is {type(target).__name__}, "
            f"not nn.Module."
        )
    return target


def _validate_strategies(
    *,
    model: nn.Module,
    val_tokens: Tensor,
    operand_side: ClockExtraction,
    resume_fn: Callable[[Tensor, int], Tensor],
    answer_position: int,
    answer_value_range: tuple[int, int],
    integer_token_ids: Sequence[int],
    strategies: dict[str, Callable[[int, Tensor], Tensor]],
) -> dict[str, float]:
    """Measure argmax_recovery on validation pairs for each write strategy.

    For each pair (a, b), compute n = clamp(a+b, lo, hi), produce h_target,
    REPLACE the residual at pos_ans with h_target (preserving everything else),
    run resume_fn from layer_readout, check if argmax(logits) == token_id(n).

    Each ``write_fn`` has signature ``(n: int, h_current: Tensor) -> Tensor``
    where ``h_current`` is the pre-write residual at the answer position
    (shape ``(d_model,)``).  Non-projection strategies may ignore it.

    This requires re-capturing the pre-readout residual for each validation
    prompt via a fresh forward pass + hook.

    Returns
    -------
    Dict mapping strategy name → argmax_recovery fraction.
    """
    layer_readout = operand_side.layer_readout
    source_layer = layer_readout - 1 if layer_readout > 0 else 0
    resume_from = layer_readout if layer_readout > 0 else 1

    ans_lo, ans_hi = answer_value_range
    tok_ids = torch.tensor(list(integer_token_ids), dtype=torch.long)

    # Capture pre-readout residuals for validation tokens
    enc_module = _resolve_attr_or_model(model, operand_side)
    h_pre, _ = _capture_layer_output(model, enc_module, val_tokens, source_layer)

    a_vals = val_tokens[:, 0].long()
    b_vals = val_tokens[:, 1].long()
    true_n = (a_vals + b_vals).clamp(ans_lo, ans_hi)
    true_token_ids = tok_ids[true_n - ans_lo]

    results: dict[str, float] = {}
    for strategy_name, write_fn in strategies.items():
        hits = 0
        for i in range(len(val_tokens)):
            n_i = int(true_n[i].item())
            h_current_i = h_pre[i, answer_position, :]  # (d_model,)
            h_target = write_fn(n_i, h_current_i)  # (d_model,)

            h_patched = h_pre[i : i + 1].clone()  # (1, seq_len, d_model)
            h_patched[0, answer_position, :] = h_target.float()

            logits = resume_fn(h_patched, resume_from)  # (1, vocab)
            argmax_tok = int(logits[0].argmax().item())
            if argmax_tok == int(true_token_ids[i].item()):
                hits += 1

        results[strategy_name] = hits / len(val_tokens) if len(val_tokens) > 0 else 0.0

    return results


def _capture_layer_output(
    model: nn.Module,
    enc_module: nn.Module,
    tokens: Tensor,
    layer_idx: int,
) -> tuple[Tensor, Tensor]:
    """Capture residual at layer_idx output and raw last-token logits."""
    buffer: list[Tensor] = []

    def _hook(_mod: nn.Module, _inp: Any, output: Any) -> None:  # noqa: ARG001
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, Tensor):
            buffer.append(tensor.detach().float().cpu())

    handle = enc_module.layers[layer_idx].register_forward_hook(_hook)
    logits_list: list[Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(tokens), 32):
                batch = tokens[start : start + 32]
                out = model(batch)
                if hasattr(out, "logits"):
                    logits_list.append(out.logits[:, -1, :].detach().float().cpu())
                elif isinstance(out, Tensor):
                    logits_list.append(out.detach().float().cpu())
    finally:
        handle.remove()

    h_all = torch.cat(buffer, dim=0)
    if logits_list:
        logits_all = torch.cat(logits_list, dim=0)
    else:
        logits_all = torch.empty(0)
    return h_all, logits_all


def _resolve_attr_or_model(model: nn.Module, operand_side: ClockExtraction) -> nn.Module:
    """Resolve the encoder backbone from the model.

    We can't read output_attr from ClockExtraction (it doesn't store it), so
    we probe the model structure.  The only enc_module we need is the one that
    has ``layers[i]``, which is ``model.gpt_neox`` for Pythia or
    ``model.model`` for Llama.  We use a simple structural probe.
    """
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox
    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
    ):
        return model.model
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder
    # Last resort: try the model itself
    if hasattr(model, "layers"):
        return model
    raise AttributeError(
        "_resolve_attr_or_model: cannot find encoder backbone with 'layers' "
        f"attribute on {type(model).__name__}.  Supported: gpt_neox, model, encoder."
    )


__all__ = [
    "TwoSidedClockExtraction",
    "extract_clock_two_sided",
]
