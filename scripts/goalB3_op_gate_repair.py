#!/usr/bin/env python3
"""Goal B3 distractor-robust op-gate repair.

This follows the B3 safe-gate repair. It retrains the per-op binary op readout
on activation features from clean/semantic/story/distractor positives and hard
negatives, keeps the repaired activation-only safe gate, and re-runs the
unchanged opaque operand-routing path.

Runtime receives only token IDs and captured activations. Prompt text and
generated `(a, b)` metadata are used only for fitting labels and grading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

import goalB2_lcm_benchmark_pipeline as b2  # noqa: E402
from goalB2_neurips_adversarial_audit import (  # noqa: E402
    OP_CONFIG,
    build_examples,
    load_runtime_components,
)
from goalB3_gate_robustness_repair import (  # noqa: E402
    fit_safe_gate,
    summarize,
    verdict,
    write_md as write_gate_md,
)


def split_examples(examples: list[b2.Example], seed: int, calib_frac: float) -> tuple[list[b2.Example], list[b2.Example]]:
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[b2.Example]] = {}
    for ex in examples:
        by_family.setdefault(ex.family, []).append(ex)
    train: list[b2.Example] = []
    calib: list[b2.Example] = []
    for rows in by_family.values():
        rows = list(rows)
        rng.shuffle(rows)
        n_calib = max(1, int(round(len(rows) * calib_frac)))
        calib.extend(rows[:n_calib])
        train.extend(rows[n_calib:])
    return train, calib


def fit_op_readout(
    base_readouts: b2.Readouts,
    examples: list[b2.Example],
    *,
    op: str,
    seed: int,
    backend: str,
    model: Any,
    device: Any,
    threshold_min: float,
    neg_margin: float,
    calib_frac: float,
) -> tuple[b2.Readouts, dict[str, Any]]:
    train, calib = split_examples(examples, seed, calib_frac)
    Xtr = b2._X(train, seed, backend, model, device)
    ytr = np.array([e.is_target for e in train], dtype=np.float32)
    W, b_arr = b2.ridge_fit(Xtr, ytr[:, None], lam=1e-2)
    op_w = W[0]
    op_b = float(b_arr[0])
    Xc = b2._X(calib, seed, backend, model, device)
    yc = [e.is_target for e in calib]
    scores = [float(b2.sigmoid(float(x @ op_w + op_b))) for x in Xc]
    neg_scores = [s for s, y in zip(scores, yc, strict=True) if y == 0]
    pos_scores = [s for s, y in zip(scores, yc, strict=True) if y == 1]
    threshold = max(float(threshold_min), max(neg_scores, default=0.5) + float(neg_margin))
    readouts = b2.Readouts(
        op_w=op_w,
        op_b=op_b,
        op_threshold=threshold,
        operand_W=base_readouts.operand_W,
        operand_b=base_readouts.operand_b,
        operand_rmse=base_readouts.operand_rmse,
        pair_conf_threshold=base_readouts.pair_conf_threshold,
    )
    summary = {
        "op": op,
        "train_n": len(train),
        "calib_n": len(calib),
        "train_pos": int(sum(e.is_target for e in train)),
        "train_neg": int(len(train) - sum(e.is_target for e in train)),
        "calib_pos": int(sum(yc)),
        "calib_neg": int(len(yc) - sum(yc)),
        "threshold": threshold,
        "threshold_min": threshold_min,
        "neg_margin": neg_margin,
        "calib_auroc": b2.auroc(scores, yc),
        "calib_min_pos": min(pos_scores) if pos_scores else None,
        "calib_max_neg": max(neg_scores) if neg_scores else None,
        "calib_pos_fire": (
            float(np.mean([s >= threshold for s in pos_scores])) if pos_scores else float("nan")
        ),
        "calib_neg_fire": (
            float(np.mean([s >= threshold for s in neg_scores])) if neg_scores else float("nan")
        ),
    }
    return readouts, summary


def write_md(path: Path, payload: dict[str, Any]) -> None:
    write_gate_md(path, payload)
    text = path.read_text()
    text = text.replace("# Goal B3 Gate Robustness Repair", "# Goal B3 Op-Gate Repair")
    text = text.replace("## Gate Fit", "## Op/Safe Gate Fit")
    text += (
        "\n## Op Readout Fit\n\n"
        "| op | train | calib | AUROC | threshold | pos fire | neg fire |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for op, row in payload["op_fit"].items():
        text += (
            f"| `{op}` | {row['train_n']} | {row['calib_n']} | "
            f"{row['calib_auroc']:.3f} | {row['threshold']:.6f} | "
            f"{row['calib_pos_fire']:.3f} | {row['calib_neg_fire']:.3f} |\n"
        )
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--fit_seed", type=int, default=691)
    p.add_argument("--eval_seed", type=int, default=692)
    p.add_argument("--fit_n_per_family", type=int, default=30)
    p.add_argument("--eval_n_per_family", type=int, default=20)
    p.add_argument("--calib_frac", type=float, default=0.25)
    p.add_argument("--op_threshold_min", type=float, default=0.65)
    p.add_argument("--op_neg_margin", type=float, default=0.05)
    p.add_argument("--safe_threshold_min", type=float, default=0.65)
    p.add_argument("--safe_neg_margin", type=float, default=0.05)
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--min_distractor_fire", type=float, default=0.80)
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--out_json", default=str(DOCS / "goalB3_op_gate_repair.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_op_gate_repair.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB3_op_gate_repair_records.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    op_fit: dict[str, Any] = {}
    safe_fit: dict[str, Any] = {}
    records = []
    for op in args.ops:
        base_readouts, _old_safe_gate, selector = load_runtime_components(base, op, args.artifact_seed)
        fit_examples = build_examples(op, args.fit_seed, args.fit_n_per_family, tok)
        readouts, op_summary = fit_op_readout(
            base_readouts,
            fit_examples,
            op=op,
            seed=args.fit_seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=args.op_threshold_min,
            neg_margin=args.op_neg_margin,
            calib_frac=args.calib_frac,
        )
        safe_gate, safe_summary = fit_safe_gate(
            fit_examples,
            op=op,
            seed=args.fit_seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=args.safe_threshold_min,
            neg_margin=args.safe_neg_margin,
            calib_frac=args.calib_frac,
        )
        op_fit[op] = op_summary
        safe_fit[op] = safe_summary
        guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=op)
        for ex in build_examples(op, args.eval_seed, args.eval_n_per_family, tok):
            runtime = b2.runtime_from_example(ex, args.eval_seed, "llama", model, device)
            pipe = b2.run_opaque_pipeline(
                runtime,
                readouts,
                guard,
                backend="llama",
                target_op=op,
                operand_lo=0,
                operand_hi=9999,
                operand_decode_mode="attention_j16_l22_chunk",
                chunk_probe=chunk_probe,
                safe_gate=safe_gate,
                chunk_pair_selector=selector,
            )
            rec = b2.score_eval_record(ex, pipe, None)
            rec["target_op"] = op
            rec["a"] = ex.a
            rec["b"] = ex.b
            records.append(rec)
    with Path(args.out_records).open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    payload = {
        "artifact_seed": args.artifact_seed,
        "fit_seed": args.fit_seed,
        "eval_seed": args.eval_seed,
        "fit_n_per_family": args.fit_n_per_family,
        "eval_n_per_family": args.eval_n_per_family,
        "op_fit": op_fit,
        "gate_fit": safe_fit,
        "ops": summarize(records),
        "records_path": args.out_records,
    }
    payload["verdict"] = verdict(payload, args.max_false_fire, args.min_distractor_fire)
    if payload["verdict"] == "GATE_REPAIR_DISTRACTOR_PASS":
        payload["verdict"] = "OP_GATE_REPAIR_DISTRACTOR_PASS"
    elif payload["verdict"] == "GATE_REPAIR_UNSAFE_FALSE_FIRE":
        payload["verdict"] = "OP_GATE_REPAIR_UNSAFE_FALSE_FIRE"
    else:
        payload["verdict"] = "OP_GATE_REPAIR_PARTIAL_OR_FAIL"
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({"verdict": payload["verdict"], "ops": payload["ops"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
