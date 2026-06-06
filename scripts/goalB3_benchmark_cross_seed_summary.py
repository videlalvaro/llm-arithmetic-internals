#!/usr/bin/env python3
"""Aggregate Goal B3 repaired benchmark reruns across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


def aggregate(paths: list[Path], *, min_seeds: int, min_lift: float, max_false_fire: float) -> dict[str, Any]:
    runs = [json.loads(path.read_text()) for path in paths]
    ops = sorted({op for run in runs for op in run["ops"]})
    op_summary: dict[str, Any] = {}
    for op in ops:
        rows = [run["ops"][op] for run in runs if op in run["ops"]]
        op_summary[op] = {
            "n_runs": len(rows),
            "total_locked": int(sum(row["n_locked"] for row in rows)),
            "total_target_locked": int(sum(row["n_target_locked"] for row in rows)),
            "mean_native_exact": float(np.mean([row["native_target_exact"] for row in rows])),
            "mean_routed_exact": float(np.mean([row["readout_routing_target_exact"] for row in rows])),
            "mean_lift": float(np.mean([row["exact_score_lift"] for row in rows])),
            "min_lift": float(np.min([row["exact_score_lift"] for row in rows])),
            "max_false_fire": float(np.max([row["hard_negative_false_fire"] for row in rows])),
            "min_pair_exact_fired": float(np.min([row["pair_exact_on_fired_target"] for row in rows])),
        }
    pass_ops = sum(
        1
        for row in op_summary.values()
        if row["n_runs"] >= min_seeds
        and row["min_lift"] >= min_lift
        and row["max_false_fire"] <= max_false_fire
    )
    verdict = (
        "GOAL_B3_BENCHMARK_CROSS_SEED_PASS"
        if len(runs) >= min_seeds and pass_ops == len(op_summary)
        else "GOAL_B3_BENCHMARK_CROSS_SEED_INCOMPLETE_OR_FAIL"
    )
    return {
        "suite": "goalB3_benchmark_cross_seed_summary",
        "input_paths": [str(path) for path in paths],
        "n_runs": len(runs),
        "seeds": [run.get("seed") for run in runs],
        "n_locked_total": int(sum(run["n_locked_total"] for run in runs)),
        "n_target_locked_total": int(sum(run["n_target_locked_total"] for run in runs)),
        "acceptance_gates": {
            "min_seeds": min_seeds,
            "min_lift": min_lift,
            "max_false_fire": max_false_fire,
        },
        "ops": op_summary,
        "verdict": verdict,
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Benchmark Cross-Seed Summary",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- runs: {payload['n_runs']}",
        f"- seeds: {', '.join(str(s) for s in payload['seeds'])}",
        f"- locked examples total: {payload['n_locked_total']}",
        f"- target locked total: {payload['n_target_locked_total']}",
        "",
        "| op | runs | total locked | mean routed | mean lift | min lift | max false-fire | min pair-exact fired |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for op, row in payload["ops"].items():
        lines.append(
            f"| `{op}` | {row['n_runs']} | {row['total_locked']} | "
            f"{row['mean_routed_exact']:.3f} | {row['mean_lift']:.3f} | "
            f"{row['min_lift']:.3f} | {row['max_false_fire']:.3f} | "
            f"{row['min_pair_exact_fired']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is an aggregation artifact only. It does not rerun models and is only as strong as the listed input runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--min_seeds", type=int, default=3)
    p.add_argument("--min_lift", type=float, default=0.20)
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--out_json", default=str(DOCS / "goalB3_benchmark_cross_seed_summary.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_benchmark_cross_seed_summary.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = aggregate(
        [Path(path) for path in args.inputs],
        min_seeds=args.min_seeds,
        min_lift=args.min_lift,
        max_false_fire=args.max_false_fire,
    )
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({"verdict": payload["verdict"], "n_runs": payload["n_runs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
