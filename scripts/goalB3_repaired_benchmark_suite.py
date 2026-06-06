#!/usr/bin/env python3
"""Goal B3 repaired-route benchmark suite.

This is the benchmark-facing Goal B3 runner for the repaired three-op Llama
route. It uses broad frozen arithmetic/adversarial splits from the existing
Goal B2 benchmark generator, but fits/evaluates the B3 repaired runtime:

- activation-only op gate;
- activation-only safe gate;
- activation-only L22 chunk operand selector/decoder;
- Python only after decoded activation-derived `(op, a, b)`.

Prompt text and generated operands are used only for split construction,
fitting labels, native generation prompts, and grading. Runtime inference
receives opaque prompt IDs plus captured activations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

import goalB2_lcm_benchmark_pipeline as b2  # noqa: E402
from goalB2_neurips_adversarial_audit import build_examples as build_b3_aug_examples  # noqa: E402
from goalB3_gate_robustness_repair import fit_safe_gate  # noqa: E402
from goalB3_op_gate_repair import fit_op_readout  # noqa: E402
from goalB3_operand_repair import fit_selector, parse_pair_thresholds  # noqa: E402


OP_DATASET = {
    "gcd": "gcd_chunk_frozen",
    "mul": "mul_chunk_frozen",
    "div_remainder": "div_remainder_frozen",
    "lcm": "lcm_chunk_frozen",
}


def log(msg: str) -> None:
    print(f"[goalB3 benchmark] {msg}", flush=True)


def build_split(op: str, args: argparse.Namespace, tok: Any | None) -> list[b2.Example]:
    if args.split_source == "deepmind_interpolate":
        ns = SimpleNamespace(
            seed=args.seed,
            target_op=op,
            n_per_family=args.n_per_family,
            n_natural=args.n_natural,
            train_frac=args.train_frac,
            calib_frac=args.calib_frac,
            dm_scan_limit=args.dm_scan_limit,
            dm_dir=args.dm_dir,
            include_common_denominator=args.include_common_denominator,
            require_multitoken_answers=args.require_multitoken_answers,
            operand_lo=args.operand_lo,
            operand_hi=args.operand_hi,
        )
        return b2.build_deepmind_examples(ns, "llama")
    ns = SimpleNamespace(
        seed=args.seed,
        backend="llama",
        target_op=op,
        n_per_family=args.n_per_family,
        n_adversarial_per_family=args.n_adversarial_per_family,
        train_frac=args.train_frac,
        calib_frac=args.calib_frac,
        operand_lo=args.operand_lo,
        operand_hi=args.operand_hi,
        require_multitoken_answers=args.require_multitoken_answers,
    )
    if op == "mul":
        return b2.build_frozen_mul_examples(ns, "llama")
    if op == "div_remainder":
        return b2.build_frozen_div_remainder_examples(ns, "llama")
    if op == "lcm":
        return b2.build_frozen_lcm_examples(ns, "llama")
    if op == "gcd":
        return b2.build_frozen_gcd_examples(ns, "llama")
    raise ValueError(op)


def dummy_base_readouts(pair_threshold: float) -> b2.Readouts:
    # Operand ridge fields are not used by the L22 chunk runtime, but
    # fit_op_readout preserves them in the Readouts dataclass.
    return b2.Readouts(
        op_w=np.zeros(b2.ACT_DIM, dtype=np.float32),
        op_b=0.0,
        op_threshold=0.5,
        operand_W=np.zeros((2, b2.ACT_DIM), dtype=np.float32),
        operand_b=np.zeros(2, dtype=np.float32),
        operand_rmse=1.0,
        pair_conf_threshold=pair_threshold,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def stable_json_hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def activation_manifest(runtime: b2.RuntimeInputs) -> dict[str, Any]:
    out = {}
    for name, value in sorted(runtime.activations.items()):
        arr = np.asarray(value)
        arr_c = np.ascontiguousarray(arr)
        out[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr_c.view(np.uint8)).hexdigest(),
        }
    return out


def replay_view(pipe: dict[str, Any]) -> dict[str, Any]:
    decoded = pipe.get("decoded")
    provenance = pipe.get("provenance") or {}
    return {
        "fired": bool(pipe.get("fired")),
        "decoded_tuple": None
        if decoded is None
        else {
            "op": decoded.get("op"),
            "a": decoded.get("a"),
            "b": decoded.get("b"),
            "op_source": decoded.get("op_source"),
            "operand_source": decoded.get("operand_source"),
        },
        "answer_source": provenance.get("answer_source"),
        "provenance": {
            "op_source": provenance.get("op_source"),
            "operand_source": provenance.get("operand_source"),
            "answer_source": provenance.get("answer_source"),
        }
        if provenance
        else {},
    }


def replay_bundle(
    *,
    runtime: b2.RuntimeInputs,
    expected_pipe: dict[str, Any],
    replayed_pipe: dict[str, Any],
    backend: str,
    operand_decode_mode: str,
    readouts: b2.Readouts,
    safe_gate: Any,
    selector: Any,
    pair_threshold: float,
) -> dict[str, Any]:
    """Build a replay-only provenance bundle with no grading or prompt fields."""
    safe_gate_payload = None
    if safe_gate is not None:
        safe_gate_payload = {
            "mode": getattr(safe_gate, "mode", None),
            "threshold": getattr(safe_gate, "threshold", None),
            "w_sha256": hashlib.sha256(np.ascontiguousarray(getattr(safe_gate, "w", np.array([], dtype=np.float32))).view(np.uint8)).hexdigest(),
        }
    selector_payload = None
    if selector is not None:
        selector_payload = stable_json_hash(
            {
                "type": type(selector).__name__,
                "threshold": getattr(selector, "threshold", None),
                "include_embeddings": getattr(selector, "include_embeddings", None),
            }
        )
    return {
        "example_id": runtime.example_id,
        "prompt_ids": list(runtime.prompt_ids),
        "activations": activation_manifest(runtime),
        "readouts": {
            "op_w_sha256": hashlib.sha256(np.ascontiguousarray(readouts.op_w).view(np.uint8)).hexdigest(),
            "op_b": float(readouts.op_b),
            "op_threshold": float(readouts.op_threshold),
            "pair_conf_threshold": float(pair_threshold),
        },
        "selectors": {
            "chunk_pair_selector": selector_payload,
        },
        "thresholds": {
            "op_threshold": float(readouts.op_threshold),
            "pair_conf_threshold": float(pair_threshold),
            "safe_gate_threshold": None if safe_gate is None else float(getattr(safe_gate, "threshold")),
        },
        "runtime_config": {
            "backend": backend,
            "operand_decode_mode": operand_decode_mode,
            "safe_gate": safe_gate_payload,
            "runtime_inputs_only": True,
        },
        "expected": replay_view(expected_pipe),
        "replayed": replay_view(replayed_pipe),
    }


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(r[key]) for r in rows]))


def summarize_op(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    target = [r for r in records if r["is_target"]]
    neg = [r for r in records if not r["is_target"]]
    fired_target = [r for r in target if r["fired"]]
    native_acc = mean_bool(target, "native_correct")
    routed_acc = mean_bool(target, "readout_routing_correct")
    lift = routed_acc - native_acc if not math.isnan(native_acc) and not math.isnan(routed_acc) else float("nan")
    native_wrong = [r for r in target if not r["native_correct"]]
    lift_values = [
        float(r["readout_routing_correct"]) - float(r["native_correct"]) for r in target
    ]
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in records}):
        rows = [r for r in records if r["family"] == family]
        fired = [r for r in rows if r["fired"]]
        by_family[family] = {
            "n": len(rows),
            "is_target": bool(rows and rows[0]["is_target"]),
            "native_exact": mean_bool(rows, "native_correct") if rows and rows[0]["is_target"] else float("nan"),
            "routed_exact": mean_bool(rows, "readout_routing_correct") if rows and rows[0]["is_target"] else float("nan"),
            "fire_rate": mean_bool(rows, "fired"),
            "pair_exact_fired": mean_bool(fired, "decoded_pair_exact") if fired else float("nan"),
        }
    return {
        "n_locked": len(records),
        "n_target_locked": len(target),
        "n_negative_locked": len(neg),
        "target_fire_rate": mean_bool(target, "fired"),
        "hard_negative_false_fire": mean_bool(neg, "fired"),
        "pair_exact_on_fired_target": mean_bool(fired_target, "decoded_pair_exact"),
        "decoded_target_correct_on_fired_target": mean_bool(fired_target, "decoded_target_correct"),
        "native_target_exact": native_acc,
        "readout_routing_target_exact": routed_acc,
        "exact_score_lift": lift,
        "exact_score_lift_bootstrap_ci": b2.bootstrap_ci(lift_values, seed),
        "native_wrong_routed_exact": mean_bool(native_wrong, "readout_routing_correct"),
        "families": by_family,
    }


def verdict(payload: dict[str, Any], args: argparse.Namespace) -> str:
    if payload["n_locked_total"] < args.min_locked:
        return "BENCHMARK_UNDERPOWERED"
    if any(row["n_target_locked"] < args.min_target_per_op for row in payload["ops"].values()):
        return "BENCHMARK_TARGET_COVERAGE_FAIL"
    passing = 0
    for row in payload["ops"].values():
        if (
            row["exact_score_lift"] >= args.min_lift
            and row["hard_negative_false_fire"] <= args.max_false_fire
            and row["pair_exact_on_fired_target"] >= args.min_pair_exact_fired
        ):
            passing += 1
    if passing >= args.min_ops:
        return "GOAL_B3_BENCHMARK_LIFT_PASS"
    return "GOAL_B3_BENCHMARK_LIFT_FAIL"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Repaired Benchmark Suite",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- model: `{payload['model_id']}`",
        f"- seed: `{payload['seed']}`",
        f"- split source: `{payload['split_source']}`",
        f"- locked examples total: {payload['n_locked_total']}",
        f"- target locked total: {payload['n_target_locked_total']}",
        "",
        "## Aggregate",
        "",
        "| op | locked | target | native exact | routed exact | lift | 95% CI | false-fire | pair-exact fired | verdict inputs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for op, row in payload["ops"].items():
        ci = row["exact_score_lift_bootstrap_ci"]
        lines.append(
            f"| `{op}` | {row['n_locked']} | {row['n_target_locked']} | "
            f"{row['native_target_exact']:.3f} | {row['readout_routing_target_exact']:.3f} | "
            f"{row['exact_score_lift']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{row['hard_negative_false_fire']:.3f} | {row['pair_exact_on_fired_target']:.3f} | "
            f"fire={row['target_fire_rate']:.3f}, native-wrong={row['native_wrong_routed_exact']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime Contract",
            "",
            "Runtime receives opaque prompt IDs plus captured activations. The op gate, safe gate, "
            "and operand decoder use activations only. Python is called only after a decoded "
            "`DecodedTuple(op, a, b)` with `op_source=activation` and "
            "`operand_source=activation`; fired records use "
            "`answer_source=python_from_decoded_tuple`.",
            "",
            "## Per-Family Breakdown",
            "",
        ]
    )
    for op, row in payload["ops"].items():
        lines.extend(
            [
                f"### {op}",
                "",
                "| family | target | n | native exact | routed exact | fire | pair-exact fired |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for family, frow in row["families"].items():
            lines.append(
                f"| `{family}` | {frow['is_target']} | {frow['n']} | "
                f"{frow['native_exact']:.3f} | {frow['routed_exact']:.3f} | "
                f"{frow['fire_rate']:.3f} | {frow['pair_exact_fired']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    if payload["split_source"] == "deepmind_interpolate":
        lines.append(
            "- This run uses real DeepMind interpolate files, but the verdict may still fail "
            "if a target op has insufficient locked target coverage."
        )
    else:
        lines.append(
            "- This run uses the repo's broad frozen arithmetic/adversarial benchmark generator. "
            "It is benchmark-facing and locked, but it is not a fresh real DeepMind file run."
        )
    lines.extend(
        [
            "- Native exact is measured only on target prompts, because off-target prompts are "
            "safety/false-fire controls rather than answer-accuracy tasks.",
            "- This is an activation-derived tool-use route, not a residual-write claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ops", nargs="+", choices=sorted(OP_DATASET), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--seed", type=int, default=801)
    p.add_argument("--n_per_family", type=int, default=80)
    p.add_argument("--n_adversarial_per_family", type=int, default=250)
    p.add_argument("--split_source", choices=["broad_frozen_arithmetic_adversarial", "deepmind_interpolate"], default="broad_frozen_arithmetic_adversarial")
    p.add_argument("--n_natural", type=int, default=200)
    p.add_argument("--dm_scan_limit", type=int, default=200000)
    p.add_argument("--dm_dir", default="")
    p.add_argument("--include_common_denominator", action="store_true")
    p.add_argument("--train_frac", type=float, default=0.40)
    p.add_argument("--calib_frac", type=float, default=0.20)
    p.add_argument("--fit_b3_aug_n_per_family", type=int, default=20)
    p.add_argument("--op_threshold_min", type=float, default=0.65)
    p.add_argument("--op_neg_margin", type=float, default=0.05)
    p.add_argument("--safe_threshold_min", type=float, default=0.65)
    p.add_argument("--safe_neg_margin", type=float, default=0.05)
    p.add_argument("--fit_calib_frac", type=float, default=0.25)
    p.add_argument("--pair_threshold", action="append", default=["mul=0.05", "div_remainder=0.20", "lcm=0.20", "gcd=0.20"])
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--chunk_top_k", type=int, default=12)
    p.add_argument("--chunk_window", type=int, default=1)
    p.add_argument("--chunk_pos_threshold", type=float, default=0.5)
    p.add_argument("--chunk_value_margin_threshold", type=float, default=0.0)
    p.add_argument("--operand_lo", type=int, default=0)
    p.add_argument("--operand_hi", type=int, default=9999)
    p.add_argument("--require_multitoken_answers", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--max_locked_per_op", type=int, default=0)
    p.add_argument("--min_locked", type=int, default=1000)
    p.add_argument("--min_target_per_op", type=int, default=50)
    p.add_argument("--min_ops", type=int, default=3)
    p.add_argument("--min_lift", type=float, default=0.20)
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--min_pair_exact_fired", type=float, default=0.80)
    p.add_argument("--out_json", default=str(DOCS / "goalB3_repaired_benchmark_suite.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_repaired_benchmark_suite.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB3_repaired_benchmark_suite_records.jsonl"))
    p.add_argument("--out_replay_bundles", default="")
    p.add_argument(
        "--replay_bundles_only",
        action="store_true",
        help="Run the frozen runtime twice and write replay bundles, but skip native generation and benchmark record output.",
    )
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    pair_thresholds = parse_pair_thresholds(args.pair_threshold)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    all_records: list[dict[str, Any]] = []
    replay_bundles: list[dict[str, Any]] = []
    op_payload: dict[str, Any] = {}
    fit_payload: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for op in args.ops:
        log(f"{op}: building frozen split")
        examples = build_split(op, args, tok)
        locked = [e for e in examples if e.split == "locked_test"]
        if args.max_locked_per_op > 0:
            locked = locked[: args.max_locked_per_op]
        fit_examples = [e for e in examples if e.split != "locked_test"]
        if args.fit_b3_aug_n_per_family > 0:
            fit_examples.extend(build_b3_aug_examples(op, args.seed + 17, args.fit_b3_aug_n_per_family, tok))
        pair_threshold = pair_thresholds.get(op, 0.20)
        base_readouts = dummy_base_readouts(pair_threshold)
        log(f"{op}: fitting repaired op gate on {len(fit_examples)} examples")
        readouts, op_fit = fit_op_readout(
            base_readouts,
            fit_examples,
            op=op,
            seed=args.seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=args.op_threshold_min,
            neg_margin=args.op_neg_margin,
            calib_frac=args.fit_calib_frac,
        )
        readouts.pair_conf_threshold = pair_threshold
        log(f"{op}: fitting repaired safe gate")
        safe_gate, safe_fit = fit_safe_gate(
            fit_examples,
            op=op,
            seed=args.seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=args.safe_threshold_min,
            neg_margin=args.safe_neg_margin,
            calib_frac=args.fit_calib_frac,
        )
        log(f"{op}: fitting repaired L22 selector")
        selector, selector_fit = fit_selector(
            fit_examples,
            op=op,
            seed=args.seed,
            model=model,
            device=device,
            chunk_probe=chunk_probe,
            include_embeddings=True,
            chunk_top_k=args.chunk_top_k,
            chunk_window=args.chunk_window,
            chunk_pos_threshold=args.chunk_pos_threshold,
            chunk_value_margin_threshold=args.chunk_value_margin_threshold,
        )
        guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=op)
        op_records = []
        log(f"{op}: evaluating {len(locked)} locked examples")
        for idx, ex in enumerate(locked, start=1):
            if idx == 1 or idx % 100 == 0 or idx == len(locked):
                log(f"{op}: evaluated {idx - 1}/{len(locked)} locked examples")
            runtime = b2.runtime_from_example(ex, args.seed, "llama", model, device)
            pipe = b2.run_opaque_pipeline(
                runtime,
                readouts,
                guard,
                backend="llama",
                target_op=op,
                operand_lo=args.operand_lo,
                operand_hi=args.operand_hi,
                operand_decode_mode="attention_j16_l22_chunk",
                chunk_probe=chunk_probe,
                chunk_top_k=args.chunk_top_k,
                chunk_window=args.chunk_window,
                chunk_pos_threshold=args.chunk_pos_threshold,
                chunk_value_margin_threshold=args.chunk_value_margin_threshold,
                safe_gate=safe_gate,
                chunk_pair_selector=selector,
            )
            if args.out_replay_bundles:
                replay_pipe = b2.run_opaque_pipeline(
                    runtime,
                    readouts,
                    guard,
                    backend="llama",
                    target_op=op,
                    operand_lo=args.operand_lo,
                    operand_hi=args.operand_hi,
                    operand_decode_mode="attention_j16_l22_chunk",
                    chunk_probe=chunk_probe,
                    chunk_top_k=args.chunk_top_k,
                    chunk_window=args.chunk_window,
                    chunk_pos_threshold=args.chunk_pos_threshold,
                    chunk_value_margin_threshold=args.chunk_value_margin_threshold,
                    safe_gate=safe_gate,
                    chunk_pair_selector=selector,
                )
                replay_bundles.append(
                    replay_bundle(
                        runtime=runtime,
                        expected_pipe=pipe,
                        replayed_pipe=replay_pipe,
                        backend="llama",
                        operand_decode_mode="attention_j16_l22_chunk",
                        readouts=readouts,
                        safe_gate=safe_gate,
                        selector=selector,
                        pair_threshold=pair_threshold,
                    )
                )
            native_text = None
            if ex.is_target and not args.replay_bundles_only:
                native_text = b2.generate_native_text(model, tok, device, ex.token_ids, args.max_new_tokens)
            rec = b2.score_eval_record(ex, pipe, native_text)
            rec["target_op"] = op
            rec["a"] = ex.a
            rec["b"] = ex.b
            op_records.append(rec)
            if not args.replay_bundles_only:
                all_records.append(rec)
        log(f"{op}: evaluated {len(locked)}/{len(locked)} locked examples")
        if args.replay_bundles_only:
            summary = {
                "n_locked": len(op_records),
                "n_target_locked": len([r for r in op_records if r["is_target"]]),
                "n_negative_locked": len([r for r in op_records if not r["is_target"]]),
                "target_fire_rate": mean_bool([r for r in op_records if r["is_target"]], "fired"),
                "hard_negative_false_fire": mean_bool([r for r in op_records if not r["is_target"]], "fired"),
                "pair_exact_on_fired_target": mean_bool(
                    [r for r in op_records if r["is_target"] and r["fired"]],
                    "decoded_pair_exact",
                ),
                "decoded_target_correct_on_fired_target": mean_bool(
                    [r for r in op_records if r["is_target"] and r["fired"]],
                    "decoded_target_correct",
                ),
                "native_target_exact": float("nan"),
                "readout_routing_target_exact": mean_bool([r for r in op_records if r["is_target"]], "readout_routing_correct"),
                "exact_score_lift": float("nan"),
                "exact_score_lift_bootstrap_ci": [float("nan"), float("nan")],
                "native_wrong_routed_exact": float("nan"),
                "families": {},
            }
        else:
            summary = summarize_op(op_records, args.seed)
        op_payload[op] = summary
        fit_payload[op] = {"op": op_fit, "safe_gate": safe_fit, "selector": selector_fit}
        manifests[op] = {
            "dataset_source": args.split_source if args.split_source == "deepmind_interpolate" else OP_DATASET[op],
            "locked_test_sha256": b2.locked_hash(examples),
            "n_examples": len(examples),
            "n_locked": len([e for e in examples if e.split == "locked_test"]),
            "n_target_locked": len([e for e in examples if e.split == "locked_test" and e.is_target]),
            "pair_conf_threshold": pair_threshold,
        }
        if args.replay_bundles_only:
            log(
                f"{op}: replay-export locked={summary['n_locked']} "
                f"false_fire={summary['hard_negative_false_fire']:.3f}"
            )
        else:
            log(
                f"{op}: native={summary['native_target_exact']:.3f} "
                f"routed={summary['readout_routing_target_exact']:.3f} "
                f"lift={summary['exact_score_lift']:.3f} false_fire={summary['hard_negative_false_fire']:.3f}"
            )
    records_path = Path(args.out_records)
    if not args.replay_bundles_only:
        write_jsonl(records_path, all_records)
    replay_bundles_path = Path(args.out_replay_bundles) if args.out_replay_bundles else None
    if replay_bundles_path is not None:
        write_jsonl(replay_bundles_path, replay_bundles)
    payload = {
        "suite": "goalB3_repaired_benchmark_suite",
        "mode": "replay_bundles_only" if args.replay_bundles_only else "benchmark",
        "model_id": b2.MODEL_ID,
        "seed": args.seed,
        "split_source": args.split_source,
        "ops": op_payload,
        "fit": fit_payload,
        "manifests": manifests,
        "pair_conf_thresholds": {op: pair_thresholds.get(op, float("nan")) for op in args.ops},
        "thresholds_frozen_before_eval": True,
        "n_locked_total": int(sum(row["n_locked"] for row in op_payload.values())),
        "n_target_locked_total": int(sum(row["n_target_locked"] for row in op_payload.values())),
        "records_path": None if args.replay_bundles_only else str(records_path),
        "replay_bundles_path": None if replay_bundles_path is None else str(replay_bundles_path),
        "acceptance_gates": {
            "min_locked": args.min_locked,
            "min_target_per_op": args.min_target_per_op,
            "min_ops": args.min_ops,
            "min_lift": args.min_lift,
            "max_false_fire": args.max_false_fire,
            "min_pair_exact_fired": args.min_pair_exact_fired,
        },
    }
    payload["verdict"] = "REPLAY_BUNDLE_EXPORT_PASS" if args.replay_bundles_only and replay_bundles_path is not None else verdict(payload, args)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.replay_bundles_only:
        Path(args.out_md).write_text(
            "# Goal B3 Replay Bundle Export\n\n"
            f"- verdict: **{payload['verdict']}**\n"
            f"- seed: `{payload['seed']}`\n"
            f"- split source: `{payload['split_source']}`\n"
            f"- locked examples total: {payload['n_locked_total']}\n"
            f"- replay bundles: `{payload['replay_bundles_path']}`\n"
        )
    else:
        write_md(Path(args.out_md), payload)
    print(json.dumps({"verdict": payload["verdict"], "n_locked_total": payload["n_locked_total"], "ops": {op: {k: payload["ops"][op][k] for k in ("native_target_exact", "readout_routing_target_exact", "exact_score_lift", "hard_negative_false_fire")} for op in args.ops}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
