#!/usr/bin/env python3
"""Aggregate Goal B3 causal-interchange runs across frozen seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def aggregate(
    paths: list[Path],
    *,
    min_seeds: int = 3,
    min_pairs_per_op: int = 50,
    min_follow: float = 1.0,
    max_random_routed_follow: float = 0.10,
) -> dict[str, Any]:
    per_op: dict[str, dict[str, Any]] = {}
    seed_rows = []
    for path in paths:
        data = load(path)
        seed_rows.append(
            {
                "path": str(path),
                "fit_seed": data.get("fit_seed"),
                "eval_seed": data.get("eval_seed"),
                "verdict": data.get("verdict"),
            }
        )
        for op, row in data.get("ops", {}).items():
            acc = per_op.setdefault(
                op,
                {
                    "n_runs": 0,
                    "total_pairs": 0,
                    "min_pairs": None,
                    "min_decoder_follow": 1.0,
                    "min_routed_follow": 1.0,
                    "max_random_decoder_follow": 0.0,
                    "max_random_routed_follow": 0.0,
                },
            )
            n_pairs = int(row.get("n_pairs", 0))
            acc["n_runs"] += 1
            acc["total_pairs"] += n_pairs
            acc["min_pairs"] = n_pairs if acc["min_pairs"] is None else min(acc["min_pairs"], n_pairs)
            acc["min_decoder_follow"] = min(acc["min_decoder_follow"], float(row.get("decoder_follow_donor_rate", 0.0)))
            acc["min_routed_follow"] = min(acc["min_routed_follow"], float(row.get("routed_answer_follow_donor_rate", 0.0)))
            acc["max_random_decoder_follow"] = max(
                acc["max_random_decoder_follow"],
                float(row.get("random_decoder_follow_donor_rate", 1.0)),
            )
            acc["max_random_routed_follow"] = max(
                acc["max_random_routed_follow"],
                float(row.get("random_routed_answer_follow_donor_rate", 1.0)),
            )
    for row in per_op.values():
        row["seed_count_gate"] = row["n_runs"] >= min_seeds
        row["pair_count_gate"] = row["total_pairs"] >= min_pairs_per_op
        row["decoder_follow_gate"] = row["min_decoder_follow"] >= min_follow
        row["routed_follow_gate"] = row["min_routed_follow"] >= min_follow
        row["random_control_gate"] = row["max_random_routed_follow"] <= max_random_routed_follow
        row["rate_gate"] = row["decoder_follow_gate"] and row["routed_follow_gate"] and row["random_control_gate"]
        if row["seed_count_gate"] and row["pair_count_gate"] and row["rate_gate"]:
            row["verdict"] = "CAUSAL_GATE_PASS"
        elif not row["rate_gate"]:
            row["verdict"] = "CAUSAL_FAIL"
        else:
            row["verdict"] = "CAUSAL_UNDERPOWERED"
        row["passes"] = row["verdict"] == "CAUSAL_GATE_PASS"
    gates = {
        "min_seeds": min_seeds,
        "min_pairs_per_op": min_pairs_per_op,
        "min_follow": min_follow,
        "max_random_routed_follow": max_random_routed_follow,
    }
    op_verdicts = {row["verdict"] for row in per_op.values()}
    if per_op and op_verdicts == {"CAUSAL_GATE_PASS"}:
        verdict = "CAUSAL_GATE_PASS"
    elif "CAUSAL_FAIL" in op_verdicts or not per_op:
        verdict = "CAUSAL_FAIL"
    else:
        verdict = "CAUSAL_UNDERPOWERED"
    return {
        "suite": "goalB3_causal_cross_seed_summary",
        "verdict": verdict,
        "gates": gates,
        "n_runs": len(paths),
        "input_paths": [str(path) for path in paths],
        "seed_rows": seed_rows,
        "ops": per_op,
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal B3 Causal Interchange Cross-Seed Summary",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- runs: {payload['n_runs']}",
        "",
        "| op | verdict | runs | total pairs | min pairs | min decoder donor-follow | min routed donor-follow | max random routed-follow |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for op, row in sorted(payload["ops"].items()):
        lines.append(
            f"| `{op}` | `{row['verdict']}` | {row['n_runs']} | {row['total_pairs']} | {row['min_pairs']} | "
            f"{row['min_decoder_follow']:.3f} | {row['min_routed_follow']:.3f} | "
            f"{row['max_random_routed_follow']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This aggregation does not rerun models. It checks that selected L22 operand",
            "chunk patches make decoder and routed answer follow the donor, while random",
            "non-selected chunk patches stay below the frozen random-follow gate.",
            "",
            "Verdicts distinguish rate failures from underpowered evidence: an op is",
            "`CAUSAL_UNDERPOWERED` when observed donor-follow/control rates pass but",
            "the frozen seed-count or pair-count gate is not met.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--min_seeds", type=int, default=3)
    parser.add_argument("--min_pairs_per_op", type=int, default=50)
    parser.add_argument("--min_follow", type=float, default=1.0)
    parser.add_argument("--max_random_routed_follow", type=float, default=0.10)
    args = parser.parse_args()
    payload = aggregate(
        args.inputs,
        min_seeds=args.min_seeds,
        min_pairs_per_op=args.min_pairs_per_op,
        min_follow=args.min_follow,
        max_random_routed_follow=args.max_random_routed_follow,
    )
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, args.out_md)
    print(json.dumps({"verdict": payload["verdict"], "ops": sorted(payload["ops"])}))


if __name__ == "__main__":
    main()
