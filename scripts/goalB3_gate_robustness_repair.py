#!/usr/bin/env python3
"""Goal B3 distractor-robust safe-gate repair.

The B3 operand-locator audit showed L22 operand chunks are usually available
under distractors. The earlier routed adversarial audit failed because the
safe gate rejected all distractor targets. This script retrains only the
activation-only safe gate with distractor target positives and hard negatives,
then re-runs the same opaque runtime path.

Runtime still receives only token IDs and captured activations. Prompt text and
generated `(a, b)` metadata are used only for gate fitting labels and grading.
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


def fit_safe_gate(
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
) -> tuple[b2.SafeGateReadout, dict[str, Any]]:
    train, calib = split_examples(examples, seed, calib_frac)
    Xtr = b2._X_safe(train, seed, backend, model, device)
    ytr = np.array([e.is_target for e in train], dtype=np.float32)
    W, b_arr = b2.ridge_fit(Xtr, ytr[:, None], lam=1e-2)
    w = W[0]
    b = float(b_arr[0])
    Xc = b2._X_safe(calib, seed, backend, model, device)
    yc = [e.is_target for e in calib]
    scores = [float(b2.sigmoid(float(x @ w + b))) for x in Xc]
    neg_scores = [s for s, y in zip(scores, yc, strict=True) if y == 0]
    pos_scores = [s for s, y in zip(scores, yc, strict=True) if y == 1]
    threshold = max(float(threshold_min), max(neg_scores, default=0.5) + float(neg_margin))
    gate = b2.SafeGateReadout(w=w, b=b, threshold=threshold, mode="l5_mean_b3_distractor")
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
    return gate, summary


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for op in sorted({r["target_op"] for r in records}):
        op_rows = [r for r in records if r["target_op"] == op]
        target = [r for r in op_rows if r["is_target"]]
        neg = [r for r in op_rows if not r["is_target"]]
        by_family = {}
        for family in OP_CONFIG[op]["families"]:
            rows = [r for r in op_rows if r["family"] == family]
            fired = [r for r in rows if r["fired"]]
            by_family[family] = {
                "n": len(rows),
                "is_target": bool(rows and rows[0]["is_target"]),
                "fire_rate": float(np.mean([r["fired"] for r in rows])) if rows else float("nan"),
                "routed_exact": (
                    float(np.mean([r["readout_routing_correct"] for r in rows]))
                    if rows
                    else float("nan")
                ),
                "pair_exact_fired": (
                    float(np.mean([r["decoded_pair_exact"] for r in fired]))
                    if fired
                    else float("nan")
                ),
                "abstain_reasons": {
                    reason: int(sum(1 for r in rows if (r.get("readout_diagnostics") or {}).get("abstain_reason") == reason))
                    for reason in sorted(
                        {
                            (r.get("readout_diagnostics") or {}).get("abstain_reason")
                            for r in rows
                            if not r["fired"]
                        }
                        - {None}
                    )
                },
            }
        out[op] = {
            "n": len(op_rows),
            "target_fire_rate": (
                float(np.mean([r["fired"] for r in target])) if target else float("nan")
            ),
            "target_exact": (
                float(np.mean([r["readout_routing_correct"] for r in target]))
                if target
                else float("nan")
            ),
            "negative_fire_rate": (
                float(np.mean([r["fired"] for r in neg])) if neg else float("nan")
            ),
            "pair_exact_fired": (
                float(np.mean([r["decoded_pair_exact"] for r in target if r["fired"]]))
                if any(r["fired"] for r in target)
                else float("nan")
            ),
            "families": by_family,
        }
    return out


def verdict(payload: dict[str, Any], max_false_fire: float, min_distractor_fire: float) -> str:
    rows = payload["ops"]
    max_neg = max(float(row["negative_fire_rate"]) for row in rows.values())
    distractor_rates = []
    for row in rows.values():
        for family, frow in row["families"].items():
            if frow["is_target"] and "distractor" in family:
                distractor_rates.append(float(frow["fire_rate"]))
    min_dist = min(distractor_rates) if distractor_rates else float("nan")
    if max_neg > max_false_fire:
        return "GATE_REPAIR_UNSAFE_FALSE_FIRE"
    if distractor_rates and min_dist >= min_distractor_fire:
        return "GATE_REPAIR_DISTRACTOR_PASS"
    return "GATE_REPAIR_PARTIAL_OR_FAIL"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Gate Robustness Repair",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- artifact seed: `{payload['artifact_seed']}`",
        f"- fit seed: `{payload['fit_seed']}`",
        f"- eval seed: `{payload['eval_seed']}`",
        "",
        "## Aggregate",
        "",
        "| op | target fire | target exact | negative fire | pair exact fired |",
        "|---|---:|---:|---:|---:|",
    ]
    for op, row in payload["ops"].items():
        lines.append(
            f"| `{op}` | {row['target_fire_rate']:.3f} | {row['target_exact']:.3f} | "
            f"{row['negative_fire_rate']:.3f} | {row['pair_exact_fired']:.3f} |"
        )
    lines.extend(["", "## Gate Fit", ""])
    for op, fit in payload["gate_fit"].items():
        lines.extend(
            [
                f"### {op}",
                "",
                f"- train/calib: {fit['train_n']} / {fit['calib_n']}",
                f"- calibration AUROC: {fit['calib_auroc']:.3f}",
                f"- threshold: {fit['threshold']:.6f}",
                f"- calib positive fire: {fit['calib_pos_fire']:.3f}",
                f"- calib negative fire: {fit['calib_neg_fire']:.3f}",
                "",
            ]
        )
    for op, row in payload["ops"].items():
        lines.extend(
            [
                f"## {op}",
                "",
                "| family | target | n | fire | routed exact | pair exact fired | abstain reasons |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for family, frow in row["families"].items():
            lines.append(
                f"| `{family}` | {frow['is_target']} | {frow['n']} | "
                f"{frow['fire_rate']:.3f} | {frow['routed_exact']:.3f} | "
                f"{frow['pair_exact_fired']:.3f} | `{json.dumps(frow['abstain_reasons'], sort_keys=True)}` |"
            )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "The repaired gate is trained on labeled construction examples, but at runtime "
            "it scores only captured `safe_gate_L5_mean` activations. Op and operands "
            "remain activation-derived; Python is called only after decoded `(op, a, b)`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--fit_seed", type=int, default=681)
    p.add_argument("--eval_seed", type=int, default=682)
    p.add_argument("--fit_n_per_family", type=int, default=30)
    p.add_argument("--eval_n_per_family", type=int, default=20)
    p.add_argument("--calib_frac", type=float, default=0.25)
    p.add_argument("--threshold_min", type=float, default=0.65)
    p.add_argument("--neg_margin", type=float, default=0.05)
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--min_distractor_fire", type=float, default=0.80)
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--out_json", default=str(DOCS / "goalB3_gate_robustness_repair.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_gate_robustness_repair.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB3_gate_robustness_repair_records.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    gate_fit: dict[str, Any] = {}
    records = []
    for op in args.ops:
        readouts, _old_safe_gate, selector = load_runtime_components(base, op, args.artifact_seed)
        fit_examples = build_examples(op, args.fit_seed, args.fit_n_per_family, tok)
        safe_gate, fit_summary = fit_safe_gate(
            fit_examples,
            op=op,
            seed=args.fit_seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=args.threshold_min,
            neg_margin=args.neg_margin,
            calib_frac=args.calib_frac,
        )
        gate_fit[op] = fit_summary
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
        "gate_fit": gate_fit,
        "ops": summarize(records),
        "records_path": args.out_records,
    }
    payload["verdict"] = verdict(payload, args.max_false_fire, args.min_distractor_fire)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({"verdict": payload["verdict"], "ops": payload["ops"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
