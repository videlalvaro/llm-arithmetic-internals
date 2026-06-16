#!/usr/bin/env python3
"""Goal B3 operand-side repair.

This keeps the repaired op/safe gates from the B3 gate work, then repairs the
operand side:

- retrain the L22 chunk pair selector on adversarial/distractor target
  positives;
- set frozen per-op pair-confidence thresholds.

Runtime remains opaque: prompt text and generated `(a, b)` metadata are used
only for fitting labels and grading; runtime receives token IDs and captured
activations only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

import goalB2_lcm_benchmark_pipeline as b2  # noqa: E402
from goalB2_neurips_adversarial_audit import (  # noqa: E402
    OP_CONFIG,
    build_examples,
    load_runtime_components,
)
from goalB3_gate_robustness_repair import summarize, verdict, write_md as write_gate_md  # noqa: E402
from goalB3_op_gate_repair import fit_op_readout  # noqa: E402
from goalB3_gate_robustness_repair import fit_safe_gate  # noqa: E402


def parse_pair_thresholds(raw: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in raw:
        op, value = item.split("=", 1)
        out[op] = float(value)
    return out


def fit_selector(
    examples: list[b2.Example],
    *,
    op: str,
    seed: int,
    model: Any,
    device: Any,
    chunk_probe: b2.J16ChunkProbe,
    include_embeddings: bool,
    chunk_top_k: int,
    chunk_window: int,
    chunk_pos_threshold: float,
    chunk_value_margin_threshold: float,
) -> tuple[b2.ChunkPairSelector | None, dict[str, Any]]:
    fit_args = SimpleNamespace(
        target_op=op,
        seed=seed,
        backend="llama",
        chunk_top_k=chunk_top_k,
        chunk_window=chunk_window,
        chunk_pos_threshold=chunk_pos_threshold,
        chunk_value_margin_threshold=chunk_value_margin_threshold,
    )
    return b2.fit_chunk_pair_selector(
        [ex for ex in examples if ex.is_target],
        fit_args,
        model,
        device,
        chunk_probe,
        include_embeddings=include_embeddings,
    )


def write_md(path: Path, payload: dict[str, Any]) -> None:
    write_gate_md(path, payload)
    text = path.read_text()
    text = text.replace("# Goal B3 Gate Robustness Repair", "# Goal B3 Operand Repair")
    text = text.replace("## Gate Fit", "## Op/Safe Gate Fit")
    text += (
        "\n## Operand Repair Fit\n\n"
        "| op | selector trained | positives | negatives | AUROC | pair threshold |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for op, row in payload["selector_fit"].items():
        text += (
            f"| `{op}` | {row.get('trained')} | {row.get('n_positive')} | "
            f"{row.get('n_negative')} | {row.get('train_auroc', float('nan')):.3f} | "
            f"{payload['pair_conf_thresholds'][op]:.3f} |\n"
        )
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--fit_seed", type=int, default=701)
    p.add_argument("--eval_seed", type=int, default=702)
    p.add_argument("--fit_n_per_family", type=int, default=30)
    p.add_argument("--eval_n_per_family", type=int, default=20)
    p.add_argument("--calib_frac", type=float, default=0.25)
    p.add_argument("--op_threshold_min", type=float, default=0.65)
    p.add_argument("--op_neg_margin", type=float, default=0.05)
    p.add_argument("--safe_threshold_min", type=float, default=0.65)
    p.add_argument("--safe_neg_margin", type=float, default=0.05)
    p.add_argument("--pair_threshold", action="append", default=["mul=0.05", "div_remainder=0.20", "lcm=0.20"])
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--min_distractor_fire", type=float, default=0.80)
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--chunk_top_k", type=int, default=12)
    p.add_argument("--chunk_window", type=int, default=1)
    p.add_argument("--chunk_pos_threshold", type=float, default=0.5)
    p.add_argument("--chunk_value_margin_threshold", type=float, default=0.0)
    p.add_argument("--selector_include_embeddings", action="store_true", default=True)
    p.add_argument("--out_json", default=str(DOCS / "goalB3_operand_repair.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_operand_repair.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB3_operand_repair_records.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    pair_thresholds = parse_pair_thresholds(args.pair_threshold)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    op_fit: dict[str, Any] = {}
    safe_fit: dict[str, Any] = {}
    selector_fit: dict[str, Any] = {}
    records = []
    for op in args.ops:
        base_readouts, _old_safe_gate, base_selector = load_runtime_components(base, op, args.artifact_seed)
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
        readouts.pair_conf_threshold = pair_thresholds.get(op, base_readouts.pair_conf_threshold)
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
        selector, selector_summary = fit_selector(
            fit_examples,
            op=op,
            seed=args.fit_seed,
            model=model,
            device=device,
            chunk_probe=chunk_probe,
            include_embeddings=args.selector_include_embeddings,
            chunk_top_k=args.chunk_top_k,
            chunk_window=args.chunk_window,
            chunk_pos_threshold=args.chunk_pos_threshold,
            chunk_value_margin_threshold=args.chunk_value_margin_threshold,
        )
        op_fit[op] = op_summary
        safe_fit[op] = safe_summary
        selector_fit[op] = selector_summary
        selector = selector if selector is not None else base_selector
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
                chunk_top_k=args.chunk_top_k,
                chunk_window=args.chunk_window,
                chunk_pos_threshold=args.chunk_pos_threshold,
                chunk_value_margin_threshold=args.chunk_value_margin_threshold,
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
        "selector_fit": selector_fit,
        "pair_conf_thresholds": {op: pair_thresholds.get(op, float("nan")) for op in args.ops},
        "ops": summarize(records),
        "records_path": args.out_records,
    }
    payload["verdict"] = verdict(payload, args.max_false_fire, args.min_distractor_fire)
    if payload["verdict"] == "GATE_REPAIR_DISTRACTOR_PASS":
        payload["verdict"] = "OPERAND_REPAIR_DISTRACTOR_PASS"
    elif payload["verdict"] == "GATE_REPAIR_UNSAFE_FALSE_FIRE":
        payload["verdict"] = "OPERAND_REPAIR_UNSAFE_FALSE_FIRE"
    else:
        payload["verdict"] = "OPERAND_REPAIR_PARTIAL_OR_FAIL"
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({"verdict": payload["verdict"], "ops": payload["ops"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
