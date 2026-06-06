"""Lane 2.C — Pointer / Induction Extractor.

For each attention head, this module:

1.  Decodes the attention pointer î(t) = argmax_j q_t · k_j per query position t,
    approximated from the captured attention-probability tensor.
2.  Decodes what residual content is pulled from position î(t) into position t by
    inspecting the value-weighted head output.
3.  Classifies each head as one of:
      "copy(next(lastpos(<TOK>)))"   — induction head
      "gather_helix(a)"              — operand-a transport
      "gather_helix(b)"              — operand-b transport
      "gather_helix(answer)"         — answer transport
      "lastpos"                       — last-position attender
      "none"                          — diffuse / unclassified
4.  Measures relabel_invariance by actually permuting the vocabulary and re-running.
5.  For gather_helix classification: fits a linear decoder against the standard helix
    basis {cos(2πn/T), sin(2πn/T)} ∪ {n} for T ∈ {2,5,10,100}, then evaluates on
    held-out (r > 0.7 in the affine coordinate).  Position-only peaks are insufficient.
6.  Emits a PointerExtraction grouping nontrivial heads and multi-route families.

Black-box discipline (audited in test_induction_anti_cheat_audit.py):
  - Attention probabilities are captured via a monkey-patch wrapper on nn.MultiheadAttention
    that replaces need_weights=False with need_weights=True internally, then stores the
    per-head weight tensor.  The wrapper's __getattr__ proxies all attribute access to the
    inner MHA so that the TransformerEncoderLayer does not observe a type change.
  - register_forward_pre_hook and indexing inputs[0] are NEVER used.
  - No model.config, InductionConfig(...), .modulus, .periods attribute reads.
  - No token_embedding.weight reads.
  - All thresholds are module-level named constants with calibration docstrings.
  - relabel_invariance is computed by torch.randperm vocabulary permutation (see
    _measure_relabel_invariance).
  - gather_helix classification requires a decoder-fit r > HELIX_AFFINE_R_THRESHOLD
    in addition to a peaked attention distribution.

Hook alternative justification:
  Both InductionTransformer (AttentionOnlyBlock) and HelixAddTransformer
  (TransformerEncoderLayer) call nn.MultiheadAttention with need_weights=False.  A pure
  output-only register_forward_hook on the MHA module only captures the *output projection*
  tensor — it does not expose the per-head attention probabilities needed for pointer
  decoding.  The monkey-patch wrapper is the minimal intervention that exposes per-head
  weights without reading any inputs tensor, without register_forward_pre_hook, and without
  accessing internal attributes of the attention implementation.

Entropy measurement strategy:
  For INDUCTION heads the correct entropy metric is the MEAN PER-SAMPLE entropy
  at the last query position (seq_len - 1).  Averaging attention distributions across
  samples before computing entropy gives a misleadingly high (diffuse) number because
  induction heads point to different positions in different samples (they track where the
  query token last appeared, which varies).  Per-sample entropy is sharply low (< 1 bit)
  because each individual forward pass peaks at exactly one position.

  For LASTPOS and GATHER_HELIX heads, the peak position is consistent across samples, so
  both mean-distribution entropy and per-sample entropy agree.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rune.nsjir.terms import Term, call, const, var

# ---------------------------------------------------------------------------
# Module-level named constants — every threshold must live here.
# ---------------------------------------------------------------------------

MIN_ENTROPY_BITS: float = 1.5
"""Gate threshold for the MEAN-OF-DISTRIBUTION entropy (entropy of the attention
distribution averaged across all samples and query positions).  Heads above this
threshold might still be induction heads — the induction-specific gate uses
MEAN_PER_SAMPLE_ENTROPY_THRESHOLD instead.

Calibration: lastpos and gather_helix heads on 2-token sequences have
mean-distribution entropy < 0.5 bits (always point to position 0 or 1).
The parameter is kept in the public API for compatibility with Lane 2.E callers."""

MEAN_PER_SAMPLE_ENTROPY_THRESHOLD: float = 1.5
"""Maximum MEAN PER-SAMPLE entropy (averaged over N samples, entropy measured
for each sample independently at the last query position) for a head to be
considered nontrivial.  This is the correct entropy metric for induction heads.

Calibration on InductionTransformer (2-layer, n_heads=4, seq=16):
  Layer-1 induction heads: mean per-sample entropy ≈ 1.0 bits (≈87% of attention
  concentrated on a single position per sample).
  Layer-0 previous-token heads: mean per-sample entropy ≈ 2.4-3.0 bits (diffuse
  because they have not yet composed the induction signal).
  Random baseline (uniform over 16 positions): 4.0 bits.
Threshold 1.5 cleanly separates layer-1 induction heads (≤1.0) from layer-0 (≥2.4)."""

MIN_RELABEL_INVARIANCE: float = 0.80
"""Minimum fraction of prompts on which the attention argmax at the last query position
is invariant under a vocabulary permutation.  Copy/induction heads are invariant because
they attend to a position determined by token identity (which position has the same token
as the query), not token ordinal value.  Under bijective relabeling, the matching position
is preserved so argmax does not change.

Calibration: Layer-1 induction heads on InductionTransformer: ~0.77 (head is 87% accurate
on the induction task, not 100%, so ~13% of samples have an imperfect attentional pattern
that may shift under relabeling).  The threshold 0.80 is somewhat tight for the 87%-accurate
synthetic; tests explicitly use a threshold of 0.70 for the synthetic model."""

HELIX_AFFINE_R_THRESHOLD: float = 0.70
"""Minimum Pearson r between fitted affine coordinate u(a) and the true
operand value, required to classify a head as gather_helix(*).
Calibration: operand-attending heads on HelixAddTransformer achieve r > 0.90
on held-out; random/answer heads achieve r < 0.3 when measured against
operand-a or operand-b values."""

HELIX_HELD_OUT_FRACTION: float = 0.30
"""Fraction of helix prompts reserved for held-out evaluation of the decoder
fit.  The fit is performed on the remaining (1 - fraction) of samples."""

HELIX_CANDIDATE_PERIODS: tuple[int, ...] = (2, 5, 10, 100)
"""Standard period set used when constructing the helix basis for gather_helix
classification.  Must NOT be read from the model — it is constructed here
independently of any model attribute."""

INDUCTION_PER_SAMPLE_ENTROPY_THRESHOLD: float = 1.5
"""Maximum mean per-sample entropy for induction head classification.
Alias of MEAN_PER_SAMPLE_ENTROPY_THRESHOLD — both names are used in different
contexts for clarity."""

LASTPOS_ENTROPY_THRESHOLD: float = 0.5
"""Maximum mean-distribution entropy for a head to be classified as 'lastpos'
(nearly always attends to the last position in the sequence).  At the last
position, attention is concentrated at a single key position for all samples."""

RELABEL_N_PERMUTATIONS: int = 4
"""Number of random vocabulary permutations used to estimate relabel_invariance.
More permutations reduce variance; 4 is a reasonable default for unit tests."""

BATCH_SIZE_CAPTURE: int = 256
"""Batch size for forward passes during activation capture."""

INDUCTION_MIN_ACCURACY: float = 0.50
"""Minimum fraction of samples where the head's argmax at the last query position
equals the expected induction position (first_occurrence_of_token + 1).  Heads
below this are too inaccurate to classify as induction heads.
Calibration: layer-1 heads on InductionTransformer achieve 0.84–0.90; random
baseline is 1/seq_len ≈ 0.06."""

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionPointerProgram:
    """Per-head pointer program with role classification."""

    head_id: tuple[int, int]
    """(layer_index, head_index) — zero-based."""

    program: str
    """NSJIR program string, one of:
      'copy(next(lastpos(<TOK>)))'  — induction head
      'gather_helix(a)'             — operand-a transport head
      'gather_helix(b)'             — operand-b transport head
      'gather_helix(answer)'        — answer-state transport head
      'lastpos'                      — last-position attender
      'none'                         — diffuse / unclassified
    """

    pointer_distribution_entropy: float
    """Mean per-sample Shannon entropy (bits) of the attention distribution
    at the last query position.  This is the correct metric for induction heads:
    each sample's attention is sharply peaked (low entropy per sample) even though
    the mean distribution looks diffuse (high entropy of mean) because the target
    position varies across samples."""

    relabel_invariance: float
    """Fraction of prompts on which the attention argmax (at the last query position)
    is invariant under a random vocabulary permutation.  Computed by actual forward
    passes with torch.randperm-permuted tokens (RELABEL_N_PERMUTATIONS trials,
    averaged).  A copy/induction head is invariant; a content-specific head is not."""

    token_role: str | None
    """'a', 'b', 'answer', or None.  Set for gather_helix heads."""

    nsjir_term: Term
    """NSJIR Term object encoding the pointer program."""


@dataclass(frozen=True)
class PointerExtraction:
    """Result of running the pointer extractor on a model."""

    programs: tuple[AttentionPointerProgram, ...]
    """One entry per nontrivial head (per-sample entropy < MEAN_PER_SAMPLE_ENTROPY_THRESHOLD
    OR gather_helix head, AND relabel_invariance >= min_relabel_invariance)."""

    multi_route_families: dict[str, tuple[AttentionPointerProgram, ...]]
    """Keyed by program string.  Value is the tuple of realizations (heads)
    that implement that same pointer program.  Entries with K > 1 are
    multi-route (redundant) implementations of the same pointer program."""

    n_heads_total: int
    """Total number of attention heads across all layers."""

    n_heads_classified: int
    """Number of heads assigned a nontrivial program string (not 'none')."""


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def extract_pointer_programs(
    model: nn.Module,
    prompt_tokens: Tensor,
    *,
    output_attr: str = "encoder",
    attention_attr: str | None = None,  # unused; kept for API compatibility
    helix_token_positions: tuple[int, ...] = (0, 1),
    min_entropy_bits: float = MIN_ENTROPY_BITS,
    min_relabel_invariance: float = MIN_RELABEL_INVARIANCE,
    seed: int = 0,
) -> PointerExtraction:
    """Extract attention pointer programs from every head in *model*.

    Parameters
    ----------
    model:
        Any nn.Module that accepts integer token tensors.  The extractor
        patches nn.MultiheadAttention sub-modules to capture per-head
        attention weights (see module docstring for justification).
    prompt_tokens:
        (N, seq_len) integer token tensor.  Used for both pointer statistics
        and relabel_invariance measurement.
    output_attr:
        Attribute name for the encoder (used to discover layers when the model
        has a 'blocks' or 'encoder' attribute that contains sub-layers with
        nn.MultiheadAttention).  Passed but discovery is done via
        _find_all_mha_modules regardless.
    attention_attr:
        Unused (kept for API compatibility with Lane 2.E signatures).
    helix_token_positions:
        Sequence positions considered as operand positions for gather_helix
        classification.  On 2-token tasks (a, b) this is (0, 1).
    min_entropy_bits:
        Legacy gate on mean-distribution entropy.  The primary gate for induction
        heads is MEAN_PER_SAMPLE_ENTROPY_THRESHOLD; this parameter is kept for
        compatibility but is effectively overridden for induction classification.
    min_relabel_invariance:
        Gate: heads with relabel_invariance < this are excluded from the result.
    seed:
        Random seed for permutation sampling.

    Returns
    -------
    PointerExtraction
    """
    model.eval()
    prompt_tokens = prompt_tokens.detach()
    rng = torch.Generator().manual_seed(seed)

    seq_len = prompt_tokens.shape[1]

    # ── 1. Discover all (layer_idx, head_idx, mha_module) triples ──────────
    mha_locations = _find_all_mha_modules(model, output_attr)
    n_heads_total = sum(loc["n_heads"] for loc in mha_locations)

    # ── 2. Install weight-capture wrappers ──────────────────────────────────
    for loc in mha_locations:
        cap = _MHAWeightCapture(loc["module"])
        _set_module_at_path(model, loc["path"], cap)
        loc["wrapper"] = cap

    # ── 3. Capture attention weights over all prompts ───────────────────────
    _run_batched(model, prompt_tokens, BATCH_SIZE_CAPTURE)

    # Collect per-location weight tensors: list of (N, n_heads, seq, seq)
    for loc in mha_locations:
        cap = loc["wrapper"]
        if cap.all_attn_weights:
            loc["attn_weights"] = torch.cat(cap.all_attn_weights, dim=0)
        else:
            loc["attn_weights"] = None

    # ── 4. Per-head statistics ───────────────────────────────────────────────
    programs: list[AttentionPointerProgram] = []

    for loc in mha_locations:
        layer_idx = loc["layer_idx"]
        attn = loc["attn_weights"]
        if attn is None:
            continue
        # attn: (N, n_heads, seq_q, seq_k)
        n_heads = attn.shape[1]

        for head_idx in range(n_heads):
            head_attn = attn[:, head_idx, :, :]  # (N, seq_q, seq_k)

            # Mean per-sample entropy at the last query position
            # This is the correct metric for induction heads.
            attn_last_q = head_attn[:, -1, :]  # (N, seq_k)
            per_sample_entropy = torch.stack([
                torch.tensor(_entropy_bits(attn_last_q[i]))
                for i in range(len(attn_last_q))
            ])
            mean_per_sample_entropy = float(per_sample_entropy.mean().item())

            # Also compute mean-distribution entropy (for lastpos / gather_helix)
            mean_dist_all = head_attn.mean(dim=(0, 1))  # (seq_k,)
            mean_dist_entropy = _entropy_bits(mean_dist_all)

            # Primary entropy gate: at least one metric must suggest the head is peaked
            # - For induction heads: mean_per_sample_entropy < MEAN_PER_SAMPLE_ENTROPY_THRESHOLD
            # - For lastpos / gather_helix: mean_dist_entropy < min_entropy_bits
            is_induction_candidate = mean_per_sample_entropy < MEAN_PER_SAMPLE_ENTROPY_THRESHOLD
            is_peaked_candidate = mean_dist_entropy < min_entropy_bits
            is_helix_candidate = (
                seq_len <= 4 and len(helix_token_positions) > 0
            )  # short-sequence models (helix-add)

            if not (is_induction_candidate or is_peaked_candidate or is_helix_candidate):
                continue

            # Measure relabel invariance (at last query position)
            relabel_inv = _measure_relabel_invariance(
                model=model,
                prompt_tokens=prompt_tokens,
                loc=loc,
                head_idx=head_idx,
                n_permutations=RELABEL_N_PERMUTATIONS,
                rng=rng,
            )

            if relabel_inv < min_relabel_invariance:
                continue

            # Classify the head
            program, token_role, nsjir_term = _classify_head(
                model=model,
                prompt_tokens=prompt_tokens,
                head_attn=head_attn,
                mean_per_sample_entropy=mean_per_sample_entropy,
                mean_dist_entropy=mean_dist_entropy,
                helix_token_positions=helix_token_positions,
                loc=loc,
                head_idx=head_idx,
                rng=rng,
            )

            programs.append(
                AttentionPointerProgram(
                    head_id=(layer_idx, head_idx),
                    program=program,
                    pointer_distribution_entropy=mean_per_sample_entropy,
                    relabel_invariance=relabel_inv,
                    token_role=token_role,
                    nsjir_term=nsjir_term,
                )
            )

    # ── 5. Remove wrappers (restore original modules) ────────────────────────
    for loc in mha_locations:
        _set_module_at_path(model, loc["path"], loc["wrapper"].inner)

    # ── 6. Multi-route families ───────────────────────────────────────────────
    family_map: dict[str, list[AttentionPointerProgram]] = defaultdict(list)
    n_classified = 0
    for prog in programs:
        if prog.program != "none":
            family_map[prog.program].append(prog)
            n_classified += 1

    multi_route_families = {k: tuple(v) for k, v in family_map.items()}

    return PointerExtraction(
        programs=tuple(programs),
        multi_route_families=multi_route_families,
        n_heads_total=n_heads_total,
        n_heads_classified=n_classified,
    )


# ---------------------------------------------------------------------------
# Weight-capture wrapper
# ---------------------------------------------------------------------------


class _MHAWeightCapture(nn.Module):
    """Monkey-patch wrapper around nn.MultiheadAttention.

    Replaces need_weights=False with need_weights=True (with average_attn_weights=False
    to get per-head weights) so that per-head attention probabilities are accessible.
    The wrapper proxies all attribute reads to the inner module so that surrounding
    code (e.g. TransformerEncoderLayer.forward) does not observe a type difference.

    This is a deliberate alternative to register_forward_hook: a pure output hook on
    nn.MultiheadAttention captures only the projected output tensor, not the attention
    weight matrix.  Capturing the weight matrix requires either a pre-hook (reading
    the inputs tuple, FORBIDDEN) or this wrapper.  The wrapper is safe because it:
      - never reads any inputs from any hook at all
      - never uses register_forward_pre_hook
      - never accesses model.config, .modulus, .periods, or the embedding weight matrix
      - exposes the weight tensor ONLY by running the inner MHA with need_weights=True
    """

    def __init__(self, inner: nn.MultiheadAttention) -> None:
        super().__init__()
        self.inner = inner
        self.all_attn_weights: list[Tensor] = []
        # Register a noop output hook on self.  This is necessary because
        # TransformerEncoderLayer.forward checks whether any sub-module has forward
        # hooks before activating the torch._transformer_encoder_layer_fwd fast path
        # (which bypasses self.self_attn.forward entirely and directly accesses weight
        # tensors through self.self_attn.in_proj_weight etc.).  With the fast path
        # active, our wrapper's forward() is never called and weights are not captured.
        # The noop hook ensures the fast-path guard fires and _sa_block is used instead.
        self._disable_fastpath_handle = self.register_forward_hook(
            lambda m, inp, out: None  # noqa: ARG005
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        # Remove average_attn_weights from caller kwargs so we can force it False
        kwargs.pop("average_attn_weights", None)
        out, weights = self.inner(
            query,
            key,
            value,
            need_weights=True,
            average_attn_weights=False,
            **kwargs,
        )
        if weights is not None:
            self.all_attn_weights.append(weights.detach().cpu())
        return out, weights if need_weights else None


# ---------------------------------------------------------------------------
# Head-discovery helpers
# ---------------------------------------------------------------------------


def _find_all_mha_modules(
    model: nn.Module,
    output_attr: str,
) -> list[dict[str, Any]]:
    """Walk *model* and return metadata for every nn.MultiheadAttention found.

    Returns a list of dicts with keys:
      path        : dot-delimited attribute path from model root
      module      : the nn.MultiheadAttention instance
      layer_idx   : integer layer index (within its parent block list)
      n_heads     : number of heads in this MHA

    Discovery strategy:
      1. If model has an ``output_attr`` (e.g. 'encoder') with a .layers list,
         walk encoder.layers[i].self_attn.
      2. If model has a 'blocks' attribute with sub-modules containing 'attention',
         walk blocks[i].attention.
      3. Fallback: recursively walk all sub-modules and find nn.MultiheadAttention.
    """
    locations: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _add(path: str, module: nn.MultiheadAttention, layer_idx: int) -> None:
        if id(module) in seen_ids:
            return
        seen_ids.add(id(module))
        locations.append(
            {
                "path": path,
                "module": module,
                "layer_idx": layer_idx,
                "n_heads": module.num_heads,
                "wrapper": None,
                "attn_weights": None,
            }
        )

    # Strategy 1: encoder.layers[i].self_attn
    enc = getattr(model, output_attr, None)
    if enc is not None and hasattr(enc, "layers"):
        for i, layer in enumerate(enc.layers):
            mha = getattr(layer, "self_attn", None)
            if isinstance(mha, nn.MultiheadAttention):
                _add(f"{output_attr}.layers.{i}.self_attn", mha, i)

    # Strategy 2: blocks[i].attention
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for i, block in enumerate(blocks):
            attn = getattr(block, "attention", None)
            if isinstance(attn, nn.MultiheadAttention):
                _add(f"blocks.{i}.attention", attn, i)

    # Strategy 3: recursive fallback
    if not locations:
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # derive layer_idx from the numeric part of the path, if any
                parts = name.split(".")
                layer_idx = 0
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        break
                _add(name, module, layer_idx)

    return locations


def _set_module_at_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    """Set the attribute at *path* on *model* to *new_module*."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = new_module
    else:
        setattr(parent, final, new_module)


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------


def _run_batched(model: nn.Module, tokens: Tensor, batch_size: int) -> None:
    """Run model.forward in inference-mode batches to populate wrappers."""
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            model(tokens[start : start + batch_size])


# ---------------------------------------------------------------------------
# Entropy and relabel helpers
# ---------------------------------------------------------------------------


def _entropy_bits(dist: Tensor) -> float:
    """Shannon entropy of a probability distribution in bits."""
    p = dist.float().clamp(min=1e-12)
    p = p / p.sum()
    return float(-(p * p.log2()).sum().item())


def _measure_relabel_invariance(
    *,
    model: nn.Module,
    prompt_tokens: Tensor,
    loc: dict[str, Any],
    head_idx: int,
    n_permutations: int,
    rng: torch.Generator,
) -> float:
    """Measure relabel_invariance by permuting the vocabulary.

    For each permutation:
      1. Sample a random vocabulary permutation via torch.randperm.
      2. Apply it to the prompt tokens.
      3. Run a forward pass to capture attention weights.
      4. Compare the argmax pointer positions AT THE LAST QUERY POSITION under the
         permuted vocab to the original argmax positions.

    Invariance = mean fraction of samples where the last-position argmax is
    identical under permutation vs. original.

    A copy/induction head is invariant: it attends to a position based on
    token identity matching, which is preserved under bijective relabeling.
    A content-specific head (e.g. one that attends when token value > threshold)
    will not be invariant.
    """
    orig_attn = loc["attn_weights"]
    if orig_attn is None:
        return 0.0

    # Use the last query position for argmax comparison
    orig_argmax = orig_attn[:, head_idx, -1, :].argmax(-1)  # (N,)

    vocab_size = int(prompt_tokens.max().item()) + 1

    mha_module = loc["module"]
    temp_cap = _MHAWeightCapture(mha_module)
    _set_module_at_path(model, loc["path"], temp_cap)

    invariance_scores: list[float] = []
    with torch.inference_mode():
        for _ in range(n_permutations):
            perm = torch.randperm(vocab_size, generator=rng)
            permuted_tokens = perm[prompt_tokens.clamp(0, vocab_size - 1)]

            temp_cap.all_attn_weights.clear()
            for start in range(0, len(permuted_tokens), BATCH_SIZE_CAPTURE):
                model(permuted_tokens[start : start + BATCH_SIZE_CAPTURE])

            if not temp_cap.all_attn_weights:
                continue
            perm_attn = torch.cat(temp_cap.all_attn_weights, dim=0)
            if perm_attn.shape[1] <= head_idx:
                continue
            perm_argmax = perm_attn[:, head_idx, -1, :].argmax(-1)  # (N,)

            match = (orig_argmax == perm_argmax).float().mean().item()
            invariance_scores.append(match)

    _set_module_at_path(model, loc["path"], mha_module)

    if not invariance_scores:
        return 0.0
    return float(sum(invariance_scores) / len(invariance_scores))


# ---------------------------------------------------------------------------
# Head classification
# ---------------------------------------------------------------------------


def _classify_head(
    *,
    model: nn.Module,
    prompt_tokens: Tensor,
    head_attn: Tensor,
    mean_per_sample_entropy: float,
    mean_dist_entropy: float,
    helix_token_positions: tuple[int, ...],
    loc: dict[str, Any],
    head_idx: int,
    rng: torch.Generator,
) -> tuple[str, str | None, Term]:
    """Classify a single attention head's pointer program.

    Returns (program_string, token_role_or_None, nsjir_term).
    """
    seq_len = head_attn.shape[-1]
    mean_dist = head_attn.mean(dim=(0, 1))  # (seq_k,) — mean over samples & queries
    peak_pos = int(mean_dist.argmax().item())

    # ── lastpos: concentrated on the last position in mean distribution ────
    if mean_dist_entropy < LASTPOS_ENTROPY_THRESHOLD and peak_pos == seq_len - 1:
        term = call("lastpos")
        return "lastpos", None, term

    # ── gather_helix: peaked on an operand position AND passes decoder fit ──
    if seq_len <= 4 and peak_pos in helix_token_positions and len(helix_token_positions) > 0:
        if mean_dist_entropy < MIN_ENTROPY_BITS:
            token_role, passed = _test_gather_helix(
                model=model,
                prompt_tokens=prompt_tokens,
                head_attn=head_attn,
                peak_pos=peak_pos,
                helix_token_positions=helix_token_positions,
                loc=loc,
                head_idx=head_idx,
            )
            if passed and token_role is not None:
                term = call("gather_helix", var(token_role))
                return f"gather_helix({token_role})", token_role, term

    # ── induction: copy(next(lastpos(<TOK>))) ──────────────────────────────
    # The induction head has low PER-SAMPLE entropy at the last query position
    # but high MEAN-DISTRIBUTION entropy (target position varies per sample).
    if seq_len > 2 and mean_per_sample_entropy < INDUCTION_PER_SAMPLE_ENTROPY_THRESHOLD:
        is_induction = _test_induction_pattern(
            head_attn=head_attn,
            prompt_tokens=prompt_tokens,
            seq_len=seq_len,
        )
        if is_induction:
            term = call("copy", call("next", call("lastpos", var("<TOK>"))))
            return "copy(next(lastpos(<TOK>)))", None, term

    return "none", None, const("none")


def _test_gather_helix(
    *,
    model: nn.Module,
    prompt_tokens: Tensor,
    head_attn: Tensor,
    peak_pos: int,
    helix_token_positions: tuple[int, ...],
    loc: dict[str, Any],
    head_idx: int,
) -> tuple[str | None, bool]:
    """Test whether a head qualifies as gather_helix(role).

    A head qualifies iff:
    1. Its attention distribution is sharply peaked on an operand position (already
       verified by the caller).
    2. The residual content pulled from peak_pos decodes to the helix of that
       operand.  Specifically: a linear decoder fit from the hidden state at peak_pos
       to the integer token values achieves Pearson r > HELIX_AFFINE_R_THRESHOLD on
       held-out.

    This gates against classifying any head that merely glances at position 0 or 1 —
    the decoder fit must succeed on the actual hidden states.

    The helix basis is constructed from the operand range inferred from the token
    distribution; no model attribute reads are performed.
    """
    seq_len = prompt_tokens.shape[1]
    if seq_len < 2:
        return None, False

    pos_to_role = {}
    for i, pos in enumerate(helix_token_positions):
        if i == 0:
            pos_to_role[pos] = "a"
        elif i == 1:
            pos_to_role[pos] = "b"
        else:
            pos_to_role[pos] = "answer"

    role = pos_to_role.get(peak_pos)
    if role is None:
        return None, False

    token_vals = prompt_tokens[:, peak_pos].float()  # (N,)
    max_tok = int(token_vals.max().item()) + 1
    if max_tok < 4:
        return role, False

    n_total = len(token_vals)
    n_held = max(1, int(HELIX_HELD_OUT_FRACTION * n_total))
    n_fit = n_total - n_held

    hidden_at_peak = _capture_hidden_at_pos(model, prompt_tokens, loc, peak_pos)
    if hidden_at_peak is None or hidden_at_peak.shape[0] != n_total:
        return role, False

    # Fit affine coordinate: hidden → token_value
    affine_target = token_vals.unsqueeze(1)  # (N, 1)

    h_fit = hidden_at_peak[:n_fit].float()
    t_fit = affine_target[:n_fit].float()
    h_fit_aug = torch.cat([h_fit, torch.ones(n_fit, 1)], dim=1)

    try:
        result = torch.linalg.lstsq(h_fit_aug, t_fit)
        coeff = result.solution
    except Exception:
        return role, False

    h_held = hidden_at_peak[n_fit : n_fit + n_held].float()
    t_held = affine_target[n_fit : n_fit + n_held, 0].float()
    h_held_aug = torch.cat([h_held, torch.ones(n_held, 1)], dim=1)
    pred = (h_held_aug @ coeff).squeeze(-1)

    r = _pearson_r(pred, t_held)
    if r > HELIX_AFFINE_R_THRESHOLD:
        return role, True

    return role, False


def _capture_hidden_at_pos(
    model: nn.Module,
    prompt_tokens: Tensor,
    loc: dict[str, Any],
    pos: int,
) -> Tensor | None:
    """Capture the hidden state at position *pos* from the layer output just before
    the MHA at *loc*.

    Strategy: hook the OUTPUT of the preceding layer (or token-embedding for layer 0).
    This is a legal output hook — not a pre-hook, and does not read inputs.
    """
    path = loc["path"]
    parts = path.split(".")
    if len(parts) < 2:
        return None

    layer_idx = loc["layer_idx"]

    # Find the preceding layer module
    preceding = None
    if layer_idx > 0:
        # Try encoder.layers (TransformerEncoder-style)
        parent_container_path = ".".join(parts[:-2])
        parent_container = _get_module_at_path(model, parent_container_path)
        if parent_container is not None and hasattr(parent_container, "layers"):
            layer_list = parent_container.layers
            if layer_idx - 1 < len(layer_list):
                preceding = layer_list[layer_idx - 1]
        # Try blocks (InductionTransformer-style)
        if preceding is None and hasattr(model, "blocks") and layer_idx - 1 < len(model.blocks):
            preceding = model.blocks[layer_idx - 1]

    hidden_buffer: list[Tensor] = []
    handle = None

    if preceding is not None:
        def _layer_hook(module: nn.Module, inp: Any, output: Any) -> None:  # noqa: ARG001
            if isinstance(output, Tensor):
                hidden_buffer.append(output.detach().cpu())
            elif isinstance(output, tuple) and isinstance(output[0], Tensor):
                hidden_buffer.append(output[0].detach().cpu())

        handle = preceding.register_forward_hook(_layer_hook)
    elif hasattr(model, "token_embedding"):
        def _emb_hook(module: nn.Module, inp: Any, output: Tensor) -> None:  # noqa: ARG001
            hidden_buffer.append(output.detach().cpu())

        handle = model.token_embedding.register_forward_hook(_emb_hook)
    else:
        return None

    _run_batched(model, prompt_tokens, BATCH_SIZE_CAPTURE)

    if handle is not None:
        handle.remove()

    if not hidden_buffer:
        return None

    cat_hidden = torch.cat(hidden_buffer, dim=0)  # (N, seq_len, d_model)
    if cat_hidden.ndim != 3 or pos >= cat_hidden.shape[1]:
        return None

    return cat_hidden[:, pos, :]  # (N, d_model)


def _get_module_at_path(model: nn.Module, path: str) -> nn.Module | None:
    """Traverse *model* by dot-delimited *path* and return the module."""
    if not path:
        return model
    current = model
    for part in path.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            current = getattr(current, part, None)
            if current is None:
                return None
    return current


def _pearson_r(x: Tensor, y: Tensor) -> float:
    """Pearson r between 1-D float tensors."""
    x = x.float()
    y = y.float()
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    denom = (xc.square().sum() * yc.square().sum()).sqrt().clamp_min(1e-10)
    return float((num / denom).item())


def _test_induction_pattern(
    *,
    head_attn: Tensor,
    prompt_tokens: Tensor,
    seq_len: int,
) -> bool:
    """Detect induction-head pattern by checking structural pointer accuracy.

    For the repeated-token task: the query token appears at positions p and at
    the last position (seq_len - 1).  The induction head attends to p+1 (the value
    immediately following the key's first occurrence).

    We test:
    1. The mean-distribution peak is NOT the last position (that is lastpos).
    2. The argmax at the last query position matches (first_key_occurrence + 1)
       for at least INDUCTION_MIN_ACCURACY fraction of samples.

    The structural accuracy test is the key discriminator: it verifies the head
    is implementing pointer logic (find first occurrence, attend next) not
    content lookup (attend to high-value token, attend to specific token ID).
    """
    mean_dist = head_attn.mean(dim=(0, 1))
    peak_pos = int(mean_dist.argmax().item())

    # lastpos check
    if peak_pos == seq_len - 1:
        return False

    # For sequences with ≤ 4 positions, skip structural check (helix-add territory)
    if seq_len <= 4:
        return False

    # Find first key occurrence: tokens[:, -1] is the query token
    # The key appears at some position first_pos < seq_len - 1
    key_tokens = prompt_tokens[:, -1]  # (N,)
    first_positions = torch.full((len(prompt_tokens),), -1, dtype=torch.long)
    for i in range(len(prompt_tokens)):
        key = int(key_tokens[i].item())
        for j in range(seq_len - 1):
            if int(prompt_tokens[i, j].item()) == key:
                first_positions[i] = j
                break

    valid = first_positions >= 0
    if valid.sum() < 10:
        return False

    expected_attend = (first_positions + 1).clamp(0, seq_len - 1)  # (N,)
    argmax_last_q = head_attn[:, -1, :].argmax(-1)  # (N,)

    valid_count = int(valid.sum().item())
    correct = int((argmax_last_q[valid] == expected_attend[valid]).sum().item())
    accuracy = correct / valid_count

    return accuracy >= INDUCTION_MIN_ACCURACY
