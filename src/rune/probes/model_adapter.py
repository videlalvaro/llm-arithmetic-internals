"""Per-architecture model-structure paths and resume-function factories.

Probes need to run "resume from layer L" continuations that differ by
architecture.  This module centralises the architecture-specific logic so
probe scripts don't each carry their own ``build_*_resume_fn``.

Supported architectures
-----------------------
- ``helix_add``  — Rune's HelixAddTransformer (``encoder`` / ``final_norm``
                   / ``unembed`` structure).  Resume logic lifted from
                   ``src/rune/schedule/jit.py::_run_from_layer``.
- ``pythia``     — EleutherAI GPTNeoX family (``gpt_neox.layers``,
                   ``gpt_neox.final_layer_norm``, ``embed_out``).  Resume
                   logic lifted from the ``build_pythia_resume_fn`` closures
                   in ``scripts/jit_demo_pythia_2.8b.py`` and sibling probe
                   scripts.
- ``llama``      — LlamaForCausalLM family (``model.layers``,
                   ``model.norm``, ``lm_head``).  Resume logic mirrors
                   ``scripts/jit_demo_llama_3.1_8b.py::build_llama_resume_fn``:
                   RoPE ``(cos, sin)`` captured via a forward hook on
                   ``model.rotary_emb`` during a one-shot priming pass; an
                   explicit lower-triangular ``-inf`` causal mask is
                   precomputed; each ``LlamaDecoderLayer`` is called with
                   ``attention_mask``, ``position_ids``, ``position_embeddings``,
                   ``past_key_values=None``, ``use_cache=False``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

# ─── Adapter dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelAdapter:
    """Per-architecture model-structure paths and helpers.

    Used by probes to abstract over HelixAddTransformer / GPTNeoX (Pythia)
    / LlamaForCausalLM differences without scattering architecture-specific
    logic across scripts.

    Attributes
    ----------
    name
        Short identifier: ``"helix_add"`` | ``"pythia"`` | ``"llama"``.
    output_attr
        Attribute path of the encoder/backbone module (e.g.
        ``"gpt_neox"``).  Used by probes that need the backbone directly.
    embedding_attr
        Attribute path of the embedding module (e.g.
        ``"gpt_neox.embed_in"``).  Used by ``extract_clock_arithmetic``
        ``embedding_attr`` argument.
    final_norm_attr
        Attribute path of the final layer norm (e.g.
        ``"gpt_neox.final_layer_norm"``).
    unembed_attr
        Attribute path of the unembed / lm_head module (e.g.
        ``"embed_out"``).
    layer_returns_tuple
        True if ``encoder.layers[i](h, ...)`` returns a tuple whose first
        element is the updated hidden state (GPTNeoX, Llama).  False if it
        returns a plain Tensor (HelixAddTransformer).
    needs_position_ids
        True if the architecture requires explicit ``position_ids`` tensors
        for RoPE (Llama).  False for GPTNeoX (uses separate rotary_emb) and
        HelixAdd (no RoPE).
    resume_fn_factory
        Callable ``(model, wrapper_or_none, seq_len) -> resume_fn`` that
        builds the architecture-specific resume closure.  The resulting
        ``resume_fn(h_patched, from_layer) -> Tensor`` runs the remaining
        layers and returns ``(batch, vocab)`` logits on CPU float32.
    """

    name: str
    output_attr: str
    embedding_attr: str
    final_norm_attr: str
    unembed_attr: str
    layer_returns_tuple: bool
    needs_position_ids: bool
    resume_fn_factory: Callable[..., Any]


# ─── HelixAdd resume ──────────────────────────────────────────────────────────


def _helix_add_resume_factory(model: nn.Module, _wrapper: Any, _seq_len: int) -> Any:
    """Build a resume_fn for HelixAddTransformer.

    The HelixAdd encoder layers return plain Tensors (no tuple unpacking
    needed).  Lifted from ``src/rune/schedule/jit.py::_run_from_layer``.
    """
    encoder = model.encoder
    n_layers = len(encoder.layers)

    def resume_fn(h_patched: Tensor, from_layer: int) -> Tensor:
        h = h_patched
        with torch.inference_mode():
            for i in range(from_layer, n_layers):
                h = encoder.layers[i](h)
            if hasattr(model, "final_norm") and hasattr(model, "unembed"):
                last = h[:, -1, :]
                logits = model.unembed(model.final_norm(last))
            else:
                logits = h[:, -1, :]
        return logits.detach().float().cpu()

    return resume_fn


# ─── Pythia resume ─────────────────────────────────────────────────────────────


def _pythia_resume_factory(model: nn.Module, wrapper: Any, seq_len: int) -> Any:
    """Build a resume_fn for Pythia / GPTNeoX.

    Captures ``position_embeddings`` (cos, sin) via a single priming
    forward through ``gpt_neox.rotary_emb``, builds the additive causal
    mask, then runs ``gpt_neox.layers[from_layer:]`` + final_layer_norm +
    embed_out.

    ``wrapper`` must expose a callable ``__call__(operand_tokens) ->
    logits`` and a ``gpt_neox`` property (like
    ``PythiaIntegerOperandWrapper`` in the probe scripts).

    Black-box: no model.config reads, no pre-hooks, no embedding-weight
    reads.  Logic lifted from ``build_pythia_resume_fn`` in
    ``scripts/pythia_restricted_excluded_probe.py`` and sibling scripts.
    """
    gpt_neox = model.gpt_neox
    embed_out = model.embed_out
    n_layers = len(gpt_neox.layers)

    # ── Capture position_embeddings via priming forward ───────────────────
    pos_emb_buffer: list[tuple[Tensor, Tensor]] = []

    def _rotary_hook(
        _mod: nn.Module, _inputs: Any, output: Any
    ) -> None:  # noqa: ARG001
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], Tensor)
            and isinstance(output[1], Tensor)
        ):
            pos_emb_buffer.append(
                (output[0].detach().clone(), output[1].detach().clone())
            )

    handle = gpt_neox.rotary_emb.register_forward_hook(_rotary_hook)
    try:
        with torch.inference_mode():
            # Use a dummy operand pair (0, 0); wrapper tokenises + runs forward
            dummy = torch.tensor([[0, 0]], dtype=torch.long)
            wrapper(dummy)
    finally:
        handle.remove()

    if not pos_emb_buffer:
        raise RuntimeError(
            "Pythia resume factory: failed to capture position_embeddings "
            "from gpt_neox.rotary_emb."
        )
    cos_cached, sin_cached = pos_emb_buffer[-1]

    # ── Build causal mask ─────────────────────────────────────────────────
    device = next(gpt_neox.parameters()).device
    mask_dtype = next(gpt_neox.parameters()).dtype
    neg_inf = torch.finfo(mask_dtype).min
    causal_2d = torch.full(
        (seq_len, seq_len), neg_inf, dtype=mask_dtype, device=device
    )
    causal_2d = torch.triu(causal_2d, diagonal=1)
    causal_mask = causal_2d.unsqueeze(0).unsqueeze(0)

    def resume_fn(h_patched: Tensor, from_layer: int) -> Tensor:
        if from_layer >= n_layers:
            with torch.inference_mode():
                h = h_patched.to(device=device, dtype=mask_dtype)
                h_norm = gpt_neox.final_layer_norm(h)
                logits = embed_out(h_norm[:, -1, :])
            return logits.detach().float().cpu()

        batch = h_patched.shape[0]
        h = h_patched.to(device=device, dtype=mask_dtype)
        cos = cos_cached.to(device=device)
        sin = sin_cached.to(device=device)
        position_embeddings = (cos, sin)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        if batch > 1:
            position_ids = position_ids.expand(batch, -1)

        with torch.inference_mode():
            for i in range(from_layer, n_layers):
                out = gpt_neox.layers[i](
                    h,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    layer_past=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                h = out[0] if isinstance(out, tuple) else out
            h = gpt_neox.final_layer_norm(h)
            logits = embed_out(h[:, -1, :])
        return logits.detach().float().cpu()

    return resume_fn


# ─── Llama resume ─────────────────────────────────────────────────────────────


def _llama_resume_factory(model: nn.Module, _wrapper: Any, seq_len: int) -> Any:
    """Build a resume_fn for LlamaForCausalLM.

    Mirrors ``scripts/jit_demo_llama_3.1_8b.py::build_llama_resume_fn``
    (verified against the Llama-3.1-8B strict-compilation run that
    produced ``docs/llama_3.1_8b_jit_operational.md``).

    Module layout
    -------------
    - encoder backbone: ``model.model.layers[i]``
    - final norm:       ``model.model.norm``
    - unembed:          ``model.lm_head``
    - rotary module:    ``model.model.rotary_emb`` — emits ``(cos, sin)``
                        position embeddings as a 2-tuple of Tensors of
                        shape ``(1, seq_len, head_dim)``.

    RoPE priming
    ------------
    Llama-3.1's ``LlamaDecoderLayer`` requires precomputed
    ``position_embeddings=(cos, sin)``.  We capture them via a forward
    hook on ``model.rotary_emb`` during a one-shot priming pass on a
    dummy ``(1, seq_len)`` input.  The hook is removed immediately.

    Black-box discipline
    --------------------
    No ``model.config`` reads; no ``embed_tokens.weight`` shape reads;
    no ``lm_head.weight`` reads.  The dummy priming input is
    ``torch.zeros(1, seq_len)`` so we do not need to know the model's
    vocab size.
    """
    llama_model = model.model
    lm_head = model.lm_head
    n_layers = len(llama_model.layers)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    pos_emb_buffer: list[tuple[Tensor, Tensor]] = []

    def _rotary_hook(_mod: nn.Module, _inputs: Any, output: Any) -> None:  # noqa: ARG001
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], Tensor)
            and isinstance(output[1], Tensor)
        ):
            pos_emb_buffer.append(
                (output[0].detach().clone(), output[1].detach().clone())
            )

    handle = llama_model.rotary_emb.register_forward_hook(_rotary_hook)
    try:
        with torch.inference_mode():
            dummy = torch.zeros((1, seq_len), dtype=torch.long, device=device)
            model(input_ids=dummy)
    finally:
        handle.remove()

    if not pos_emb_buffer:
        raise RuntimeError(
            "Failed to capture position_embeddings from model.rotary_emb. "
            "Llama-3.1 rotary_emb should emit (cos, sin) tuple — investigate."
        )
    cos_cached, sin_cached = pos_emb_buffer[-1]

    neg_inf = torch.finfo(dtype).min
    causal_2d = torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device)
    causal_2d = torch.triu(causal_2d, diagonal=1)
    causal_mask = causal_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def resume_fn(h_patched: Tensor, from_layer: int) -> Tensor:
        if from_layer >= n_layers:
            with torch.inference_mode():
                h = h_patched.to(device=device, dtype=dtype)
                h_norm = llama_model.norm(h)
                logits = lm_head(h_norm[:, -1, :])
            return logits.detach().float().cpu()

        batch = h_patched.shape[0]
        h = h_patched.to(device=device, dtype=dtype)
        cos = cos_cached.to(device=device)
        sin = sin_cached.to(device=device)
        position_embeddings = (cos, sin)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        if batch > 1:
            position_ids = position_ids.expand(batch, -1)

        with torch.inference_mode():
            for i in range(from_layer, n_layers):
                out = llama_model.layers[i](
                    h,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                h = out[0] if isinstance(out, tuple) else out
            h = llama_model.norm(h)
            logits = lm_head(h[:, -1, :])
        return logits.detach().float().cpu()

    return resume_fn


# ─── Canonical adapter singletons ─────────────────────────────────────────────

ADAPTER_HELIX_ADD: ModelAdapter = ModelAdapter(
    name="helix_add",
    output_attr="encoder",
    embedding_attr="token_embedding",
    final_norm_attr="final_norm",
    unembed_attr="unembed",
    layer_returns_tuple=False,
    needs_position_ids=False,
    resume_fn_factory=_helix_add_resume_factory,
)

ADAPTER_PYTHIA: ModelAdapter = ModelAdapter(
    name="pythia",
    output_attr="gpt_neox",
    embedding_attr="gpt_neox.embed_in",
    final_norm_attr="gpt_neox.final_layer_norm",
    unembed_attr="embed_out",
    layer_returns_tuple=True,
    needs_position_ids=False,
    resume_fn_factory=_pythia_resume_factory,
)

ADAPTER_LLAMA: ModelAdapter = ModelAdapter(
    name="llama",
    output_attr="model",
    embedding_attr="model.embed_tokens",
    final_norm_attr="model.norm",
    unembed_attr="lm_head",
    layer_returns_tuple=True,
    needs_position_ids=True,
    resume_fn_factory=_llama_resume_factory,
)


# ─── Auto-detect ──────────────────────────────────────────────────────────────


def detect_adapter(model: nn.Module) -> ModelAdapter:
    """Inspect the model's module structure and return the matching adapter.

    Detection order matters: check GPTNeoX before Llama because both have
    an ``embed_out`` / ``lm_head`` pattern.
    """
    if hasattr(model, "gpt_neox"):
        return ADAPTER_PYTHIA
    if (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and hasattr(model, "lm_head")
    ):
        return ADAPTER_LLAMA
    if (
        hasattr(model, "encoder")
        and hasattr(model, "final_norm")
        and hasattr(model, "unembed")
    ):
        return ADAPTER_HELIX_ADD
    raise ValueError(
        f"detect_adapter: unrecognised model architecture {type(model).__name__!r}. "
        "Expected one of: HelixAddTransformer (encoder/final_norm/unembed), "
        "GPTNeoXForCausalLM (gpt_neox), LlamaForCausalLM (model.layers + lm_head)."
    )


__all__ = [
    "ModelAdapter",
    "ADAPTER_HELIX_ADD",
    "ADAPTER_PYTHIA",
    "ADAPTER_LLAMA",
    "detect_adapter",
]
