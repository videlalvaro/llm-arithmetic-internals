"""Canonical activation cache storage.

Phase 0 uses this as the shared substrate for gold parity, detection, extraction,
and verification work. A cache is a directory containing `activations.h5` and
`manifest.json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

MANIFEST_VERSION = 1
ACTIVATIONS_FILE = "activations.h5"
MANIFEST_FILE = "manifest.json"
ArrayIndex = int | slice
GraphManifest = Mapping[str, Any]


@dataclass(frozen=True)
class ActivationSpec:
    """Manifest entry for one cached activation tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, order=True)
class GraphNode:
    """Stable node identifier for a model component."""

    id: str
    kind: str


@dataclass(frozen=True, order=True)
class GraphEdge:
    """Stable edge identifier between model components."""

    src_component: str
    dst_component: str
    ablation: str = "zero"

    @property
    def id(self) -> str:
        return f"{self.src_component}->{self.dst_component}:{self.ablation}"


@dataclass(frozen=True, order=True)
class ActivationHookPoint:
    """Canonical activation hook point mapped onto a PyTorch module path."""

    name: str
    module_name: str
    kind: str


@dataclass(frozen=True)
class ActivationCache:
    """Read-only view over an activation cache directory."""

    root: Path
    manifest: Mapping[str, Any]

    @classmethod
    def open(cls, root: str | Path) -> ActivationCache:
        cache_root = Path(root)
        with (cache_root / MANIFEST_FILE).open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported activation cache version: {manifest.get('version')}")
        return cls(root=cache_root, manifest=manifest)

    @property
    def hdf5_path(self) -> Path:
        return self.root / ACTIVATIONS_FILE

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(spec["name"] for spec in self.manifest["activations"])

    @property
    def specs(self) -> tuple[ActivationSpec, ...]:
        return tuple(
            ActivationSpec(
                name=spec["name"],
                shape=tuple(spec["shape"]),
                dtype=spec["dtype"],
            )
            for spec in self.manifest["activations"]
        )

    @property
    def node_index(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest.get("node_index", ()))

    @property
    def edge_index(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest.get("edge_index", ()))

    def get(self, name: str) -> NDArray[Any]:
        if name not in self.keys:
            raise KeyError(f"Activation {name!r} is not present in cache")
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            return np.asarray(hdf5_file[name])

    def get_slice(
        self,
        name: str,
        *,
        batch: ArrayIndex | None = None,
        token: ArrayIndex | None = None,
        component: ArrayIndex | None = None,
    ) -> NDArray[Any]:
        array = self.get(name)
        index: list[ArrayIndex] = [slice(None)] * array.ndim
        requested = ((0, batch, "batch"), (1, token, "token"), (2, component, "component"))
        for axis, axis_index, label in requested:
            if axis_index is None:
                continue
            if array.ndim <= axis:
                raise IndexError(f"Cannot slice {label} axis from {array.ndim}D activation")
            index[axis] = axis_index
        return array[tuple(index)]


def tensor_to_numpy(value: Tensor) -> NDArray[Any]:
    """Detach a tensor and move it to CPU NumPy storage for cache writing."""

    return value.detach().cpu().numpy()


def extract_tensor_output(output: Any) -> Tensor:
    """Extract the tensor payload from a module output accepted by the recorder."""

    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Cannot cache module output of type {type(output).__name__}")


def infer_activation_hook_points(model: nn.Module) -> tuple[ActivationHookPoint, ...]:
    """Infer canonical hook points for supported Rune transformer modules."""

    modules = dict(model.named_modules())
    hook_points: list[ActivationHookPoint] = []
    if "token_embedding" in modules:
        hook_points.append(
            ActivationHookPoint(
                name="post_embedding",
                module_name="token_embedding",
                kind="post_embedding",
            )
        )
    elif "model.embed_tokens" in modules:
        hook_points.append(
            ActivationHookPoint(
                name="post_embedding",
                module_name="model.embed_tokens",
                kind="post_embedding",
            )
        )

    layer_indices = sorted(
        {
            int(name.split(".")[2])
            for name in modules
            if name.startswith("encoder.layers.") and len(name.split(".")) >= 3
        }
    )
    for layer_index in layer_indices:
        layer_prefix = f"encoder.layers.{layer_index}"
        attention = f"{layer_prefix}.self_attn"
        mlp_intermediate = f"{layer_prefix}.linear1"
        mlp_out = f"{layer_prefix}.linear2"
        if attention in modules:
            hook_points.extend(
                [
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.attention_scores",
                        module_name=attention,
                        kind="attention_scores",
                    ),
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.post_attention_out",
                        module_name=attention,
                        kind="post_attention_out",
                    ),
                ]
            )
        if mlp_intermediate in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.mlp_intermediate",
                    module_name=mlp_intermediate,
                    kind="mlp_intermediate",
                )
            )
        if mlp_out in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.post_mlp",
                    module_name=mlp_out,
                    kind="post_mlp",
                )
            )
        if layer_prefix in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.post_layer_residual",
                    module_name=layer_prefix,
                    kind="post_layer_residual",
                )
            )

    block_indices = sorted(
        {
            int(name.split(".")[1])
            for name in modules
            if name.startswith("blocks.") and len(name.split(".")) >= 2
        }
    )
    for block_index in block_indices:
        block_prefix = f"blocks.{block_index}"
        attention = f"{block_prefix}.attention"
        if attention in modules:
            hook_points.extend(
                [
                    ActivationHookPoint(
                        name=f"layers.{block_index}.attention_scores",
                        module_name=attention,
                        kind="attention_scores",
                    ),
                    ActivationHookPoint(
                        name=f"layers.{block_index}.post_attention_out",
                        module_name=attention,
                        kind="post_attention_out",
                    ),
                ]
            )
        if block_prefix in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{block_index}.post_layer_residual",
                    module_name=block_prefix,
                    kind="post_layer_residual",
                )
            )

    llama_layer_indices = sorted(
        {
            int(name.split(".")[2])
            for name in modules
            if name.startswith("model.layers.") and len(name.split(".")) >= 3
        }
    )
    for layer_index in llama_layer_indices:
        layer_prefix = f"model.layers.{layer_index}"
        attention = f"{layer_prefix}.self_attn"
        mlp_intermediate = f"{layer_prefix}.mlp.up_proj"
        mlp_out = f"{layer_prefix}.mlp.down_proj"
        if attention in modules:
            hook_points.extend(
                [
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.attention_scores",
                        module_name=attention,
                        kind="attention_scores",
                    ),
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.post_attention_out",
                        module_name=attention,
                        kind="post_attention_out",
                    ),
                ]
            )
        if mlp_intermediate in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.mlp_intermediate",
                    module_name=mlp_intermediate,
                    kind="mlp_intermediate",
                )
            )
        if mlp_out in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.post_mlp",
                    module_name=mlp_out,
                    kind="post_mlp",
                )
            )
        if layer_prefix in modules:
            hook_points.append(
                ActivationHookPoint(
                    name=f"layers.{layer_index}.post_layer_residual",
                    module_name=layer_prefix,
                    kind="post_layer_residual",
                )
            )

    # GPTNeoX layout (Pythia family)
    # Verified by inspecting GPTNeoXForCausalLM(GPTNeoXConfig(hidden_size=32, ...)).named_modules():
    #   gpt_neox.embed_in
    #   gpt_neox.layers.N.attention
    #   gpt_neox.layers.N.mlp.dense_h_to_4h
    #   gpt_neox.layers.N.mlp.dense_4h_to_h
    #   gpt_neox.layers.N   (post_layer_residual)
    #   embed_out
    gpt_neox_layer_indices = sorted(
        {
            int(name.split(".")[2])
            for name in modules
            if name.startswith("gpt_neox.layers.") and len(name.split(".")) >= 3
        }
    )
    if gpt_neox_layer_indices and "gpt_neox.embed_in" in modules:
        hook_points.append(
            ActivationHookPoint(
                name="post_embedding",
                module_name="gpt_neox.embed_in",
                kind="post_embedding",
            )
        )
        for layer_index in gpt_neox_layer_indices:
            layer_prefix = f"gpt_neox.layers.{layer_index}"
            attention = f"{layer_prefix}.attention"
            mlp_intermediate = f"{layer_prefix}.mlp.dense_h_to_4h"
            mlp_out = f"{layer_prefix}.mlp.dense_4h_to_h"
            if attention in modules:
                hook_points.extend(
                    [
                        ActivationHookPoint(
                            name=f"layers.{layer_index}.attention_scores",
                            module_name=attention,
                            kind="attention_scores",
                        ),
                        ActivationHookPoint(
                            name=f"layers.{layer_index}.post_attention_out",
                            module_name=attention,
                            kind="post_attention_out",
                        ),
                    ]
                )
            if mlp_intermediate in modules:
                hook_points.append(
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.mlp_intermediate",
                        module_name=mlp_intermediate,
                        kind="mlp_intermediate",
                    )
                )
            if mlp_out in modules:
                hook_points.append(
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.post_mlp",
                        module_name=mlp_out,
                        kind="post_mlp",
                    )
                )
            if layer_prefix in modules:
                hook_points.append(
                    ActivationHookPoint(
                        name=f"layers.{layer_index}.post_layer_residual",
                        module_name=layer_prefix,
                        kind="post_layer_residual",
                    )
                )

    return tuple(sorted(hook_points))


class ActivationRecorder:
    """Capture PyTorch module outputs with forward hooks."""

    def __init__(self, model: nn.Module, module_names: list[str] | tuple[str, ...]) -> None:
        self.model = model
        self.module_names = tuple(module_names)
        self.activations: dict[str, NDArray[Any]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> ActivationRecorder:
        modules = dict(self.model.named_modules())
        missing = sorted(set(self.module_names) - set(modules))
        if missing:
            raise KeyError(f"Unknown module names: {', '.join(missing)}")

        for name in self.module_names:
            self._handles.append(modules[name].register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.activations[name] = tensor_to_numpy(extract_tensor_output(output))

        return hook


def canonical_node_id(module_name: str) -> str:
    """Return the manifest node ID for a PyTorch module path."""

    return "model" if module_name == "" else module_name


def build_module_graph_manifest(
    model: nn.Module,
    *,
    ablation: str = "zero",
) -> dict[str, Any]:
    """Build a deterministic computation-graph manifest from module hierarchy."""

    if ablation not in {"zero", "interchange", "mean"}:
        raise ValueError(f"Unsupported ablation mode: {ablation}")

    modules = list(model.named_modules())
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    module_kinds: dict[str, str] = {}
    for name, module in modules:
        node_id = canonical_node_id(name)
        module_kinds[node_id] = module.__class__.__name__
        if name:
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            children_by_parent[canonical_node_id(parent_name)].append(node_id)

    nodes = [GraphNode(id=node_id, kind=module_kinds[node_id]) for node_id in sorted(module_kinds)]
    edges = [
        GraphEdge(src_component=parent_id, dst_component=child_id, ablation=ablation)
        for parent_id in sorted(children_by_parent)
        for child_id in sorted(children_by_parent[parent_id])
    ]
    node_index = [{"id": node.id, "kind": node.kind} for node in nodes]
    edge_index = [
        {
            "id": edge.id,
            "src_component": edge.src_component,
            "dst_component": edge.dst_component,
            "ablation": edge.ablation,
        }
        for edge in edges
    ]
    graph_payload = {
        "version": MANIFEST_VERSION,
        "node_index": node_index,
        "edge_index": edge_index,
    }
    graph_payload["model_graph_id"] = sha256(
        json.dumps(graph_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return graph_payload


def write_activation_cache(
    root: str | Path,
    activations: Mapping[str, ArrayLike],
    *,
    metadata: Mapping[str, Any] | None = None,
    graph: GraphManifest | None = None,
) -> ActivationCache:
    """Write activations to a canonical cache directory and return a read view."""

    if not activations:
        raise ValueError("Activation cache must contain at least one tensor")

    cache_root = Path(root)
    cache_root.mkdir(parents=True, exist_ok=True)

    specs: list[dict[str, Any]] = []
    with h5py.File(cache_root / ACTIVATIONS_FILE, "w") as hdf5_file:
        for name in sorted(activations):
            array = np.asarray(activations[name])
            hdf5_file.create_dataset(name, data=array)
            specs.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )

    manifest = {
        "version": MANIFEST_VERSION,
        "format": "rune.activation_cache",
        "activations_file": ACTIVATIONS_FILE,
        "metadata": dict(metadata or {}),
        "activations": specs,
    }
    if graph is not None:
        manifest.update(
            {
                "model_graph_id": graph["model_graph_id"],
                "node_index": list(graph["node_index"]),
                "edge_index": list(graph["edge_index"]),
            }
        )
    with (cache_root / MANIFEST_FILE).open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")

    return ActivationCache.open(cache_root)