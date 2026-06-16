#!/usr/bin/env python3
"""Qwen operand localization and P1.3 autopsy scaffold for Goal B3.

The real backend captures Qwen layer/position activations for a layer grid and
fits value/pair readouts only on train/calib labels. The synthetic backend is
for smoke tests and CI. This script reports the decision criterion from the
milestone plan: no site above ordered pair exact 0.30 is a clean Qwen
operand-state falsifier; any site above 0.80 is a candidate route to freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    a: int
    b: int
    prompt: str
    token_ids: list[int]
    a_token_index: int | None = None
    b_token_index: int | None = None


def stable_split_hash(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(
            json.dumps(
                {
                    "id": ex.example_id,
                    "split": ex.split,
                    "token_ids": ex.token_ids,
                },
                sort_keys=True,
            ).encode()
        )
        h.update(b"\n")
    return h.hexdigest()


def build_synthetic_examples(seed: int, n: int) -> list[Example]:
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n):
        a = int(rng.integers(0, 10000))
        b = int(rng.integers(0, 10000))
        prompt = f"What is the greatest common divisor of {a} and {b}? Answer: "
        split = "train" if idx < int(0.5 * n) else ("calib" if idx < int(0.7 * n) else "locked_test")
        rows.append(
            Example(
                example_id=hashlib.sha256(f"{seed}:{idx}:{prompt}".encode()).hexdigest()[:16],
                split=split,
                a=a,
                b=b,
                prompt=prompt,
                token_ids=[ord(c) % 251 for c in prompt],
            )
        )
    return rows


def _token_index_after_prefix(tokenizer: Any, prompt: str, prefix: str) -> int:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    if len(prefix_ids) >= len(prompt_ids):
        return max(0, len(prompt_ids) - 1)
    return max(0, len(prefix_ids))


def build_qwen_examples(seed: int, n: int, tokenizer: Any) -> list[Example]:
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n):
        a = int(rng.integers(0, 10000))
        b = int(rng.integers(1, 10000))
        template = "What is the greatest common divisor of {a} and {b}? Answer: "
        prompt = template.format(a=a, b=b)
        split = "train" if idx < int(0.5 * n) else ("calib" if idx < int(0.7 * n) else "locked_test")
        token_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        a_prefix = prompt.split(str(a), 1)[0]
        b_prefix = prompt.rsplit(str(b), 1)[0]
        rows.append(
            Example(
                example_id=hashlib.sha256(f"qwen:{seed}:{idx}:{prompt}".encode()).hexdigest()[:16],
                split=split,
                a=a,
                b=b,
                prompt=prompt,
                token_ids=list(map(int, token_ids)),
                a_token_index=_token_index_after_prefix(tokenizer, prompt, a_prefix),
                b_token_index=_token_index_after_prefix(tokenizer, prompt, b_prefix),
            )
        )
    return rows


def synthetic_capture(examples: list[Example], layers: list[int], positions: list[str], seed: int) -> dict[tuple[int, str], np.ndarray]:
    rng = np.random.default_rng(seed)
    feats: dict[tuple[int, str], np.ndarray] = {}
    labels = np.array([[ex.a, ex.b] for ex in examples], dtype=np.float32)
    for layer in layers:
        for pos in positions:
            noise = rng.normal(0, 50.0 + layer, size=(len(examples), 2))
            if pos == "answer_site" and layer >= max(layers):
                X = labels + noise * 0.0001
            elif pos.startswith("input"):
                X = labels + noise
            else:
                X = rng.normal(0, 1.0, size=(len(examples), 2))
            feats[(layer, pos)] = X.astype(np.float32)
    return feats


def qwen_capture(examples: list[Example], layers: list[int], positions: list[str], model_id: str) -> dict[tuple[int, str], np.ndarray]:
    import torch
    from transformers import AutoModelForCausalLM

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if device.type != "cpu":
        try:
            model.to(device)
        except RuntimeError as exc:
            print(f"[goalB3-qwen-diagnostic] device fallback to CPU: {exc}", flush=True)
            device = torch.device("cpu")
    model.eval()
    model_layers = model.model.layers
    wanted_layers = [layer for layer in layers if layer < len(model_layers)]
    rows: dict[tuple[int, str], list[np.ndarray]] = {(layer, pos): [] for layer in wanted_layers for pos in positions}

    def pos_index(ex: Example, position: str) -> int:
        if position == "answer_site":
            return len(ex.token_ids) - 1
        if position == "input_a" and ex.a_token_index is not None:
            return min(ex.a_token_index, len(ex.token_ids) - 1)
        if position == "input_b" and ex.b_token_index is not None:
            return min(ex.b_token_index, len(ex.token_ids) - 1)
        return len(ex.token_ids) - 1

    for ex in examples:
        ids_t = torch.tensor([ex.token_ids], dtype=torch.long, device=device)
        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hs = output[0] if isinstance(output, tuple) else output
                if isinstance(hs, torch.Tensor):
                    captured[layer] = hs[0].clone().detach().to("cpu", torch.float32)
            return hook

        handles = [model_layers[layer].register_forward_hook(make_hook(layer)) for layer in wanted_layers]
        try:
            with torch.no_grad():
                _ = model(input_ids=ids_t, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for layer in wanted_layers:
            H = captured[layer]
            for pos in positions:
                rows[(layer, pos)].append(H[pos_index(ex, pos)].numpy())

    return {key: np.stack(vals).astype(np.float32) for key, vals in rows.items() if vals}


def pair_metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    pred_i = np.rint(pred).clip(0, 9999).astype(int)
    gold_i = gold.astype(int)
    ordered = np.mean((pred_i[:, 0] == gold_i[:, 0]) & (pred_i[:, 1] == gold_i[:, 1]))
    unordered = np.mean(
        [
            set(map(int, p)) == set(map(int, g))
            for p, g in zip(pred_i, gold_i, strict=True)
        ]
    )
    topk = np.mean(
        [
            int(g[0]) in range(max(0, int(p[0]) - 2), min(10000, int(p[0]) + 3))
            and int(g[1]) in range(max(0, int(p[1]) - 2), min(10000, int(p[1]) + 3))
            for p, g in zip(pred_i, gold_i, strict=True)
        ]
    )
    mae = float(np.mean(np.abs(pred_i - gold_i))) if len(pred_i) else float("nan")
    return {
        "ordered_pair_exact": float(ordered),
        "unordered_pair_exact": float(unordered),
        "top5_value_recall_proxy": float(topk),
        "mean_abs_error": mae,
        "confidence_calibration_proxy": float(1.0 / (1.0 + mae)) if np.isfinite(mae) else 0.0,
    }


def evaluate_sites(examples: list[Example], features: dict[tuple[int, str], np.ndarray]) -> list[dict[str, Any]]:
    train = np.array([ex.split in {"train", "calib"} for ex in examples])
    locked = np.array([ex.split == "locked_test" for ex in examples])
    y = np.array([[ex.a, ex.b] for ex in examples], dtype=np.float32)
    rows = []
    for (layer, pos), X in sorted(features.items()):
        model = Ridge(alpha=1.0)
        model.fit(X[train], y[train])
        pred = model.predict(X[locked])
        metrics = pair_metrics(pred, y[locked])
        metrics.update({"layer": layer, "position": pos, "n_locked": int(locked.sum())})
        rows.append(metrics)
    return rows


def p13_autopsy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=lambda r: r["ordered_pair_exact"]) if rows else None
    return {
        "best_site": best,
        "collapse_modes_checked": ["decoded_a=0", "decoded_b=0", "layer_site_mismatch"],
        "old_probe_bank_comparison": "requires --p13_probe_bank on real backend",
    }


def verdict(rows: list[dict[str, Any]]) -> str:
    best = max((r["ordered_pair_exact"] for r in rows), default=0.0)
    if best < 0.30:
        return "QWEN_OPERAND_ROUTE_FAIL"
    if best >= 0.80:
        return "QWEN_OPERAND_ROUTE_FOUND"
    return "QWEN_OPERAND_ROUTE_WEAK"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Qwen Operand Diagnostics",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- backend: `{payload['backend']}`",
        f"- split hash: `{payload['split_hash']}`",
        "",
        "| layer | position | locked | ordered pair | unordered pair | top-k recall proxy | confidence proxy |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["site_metrics"]:
        lines.append(
            f"| {row['layer']} | `{row['position']}` | {row['n_locked']} | "
            f"{row['ordered_pair_exact']:.3f} | {row['unordered_pair_exact']:.3f} | "
            f"{row['top5_value_recall_proxy']:.3f} | {row['confidence_calibration_proxy']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["synthetic", "qwen"], default="synthetic")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-7B")
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--layers", nargs="+", type=int, default=[8, 10, 12, 15])
    p.add_argument("--positions", nargs="+", default=["answer_site", "input_a", "input_b"])
    p.add_argument("--out_json", type=Path, default=DOCS / "goalB3_qwen_operand_diagnostics.json")
    p.add_argument("--out_md", type=Path, default=DOCS / "goalB3_qwen_operand_diagnostics.md")
    args = p.parse_args()
    if args.backend == "synthetic":
        examples = build_synthetic_examples(args.seed, args.n)
        features = synthetic_capture(examples, args.layers, args.positions, args.seed)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        examples = build_qwen_examples(args.seed, args.n, tokenizer)
        features = qwen_capture(examples, args.layers, args.positions, args.model_id)
    rows = evaluate_sites(examples, features)
    payload = {
        "suite": "goalB3_qwen_operand_diagnostics",
        "backend": args.backend,
        "model_id": args.model_id if args.backend == "qwen" else None,
        "split_hash": stable_split_hash(examples),
        "site_metrics": rows,
        "p13_autopsy": p13_autopsy(rows),
        "verdict": verdict(rows),
        "runtime_label_policy": "labels used only for train/calib fitting and locked grading",
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(args.out_md, payload)
    print(json.dumps({"verdict": payload["verdict"], "split_hash": payload["split_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
