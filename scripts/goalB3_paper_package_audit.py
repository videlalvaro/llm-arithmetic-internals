#!/usr/bin/env python3
"""Assemble and gate the Goal B3 paper-package evidence.

This is a reproducible manifest, not a model runner. It reads the frozen
artifacts already emitted by the benchmark, causal, adversarial, provenance,
and cross-model scripts and checks them against the paper-facing minimum bar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DOCS = Path("docs")
RESEARCH = DOCS / "research"


DEFAULTS = {
    "broad_cross_seed": DOCS / "goalB3_benchmark_cross_seed_summary.json",
    "gcd_cross_seed": DOCS / "goalB3_gcd_benchmark_cross_seed_summary.json",
    "deepmind_cross_seed": DOCS / "goalB3_final_deepmind_interpolate_recognized_cross_seed.json",
    "deepmind_provenance": DOCS / "goalB3_final_deepmind_provenance.json",
    "retrospective_deepmind_cross_seed": DOCS / "goalB3_deepmind_gcd_div_lcm_cross_seed_summary.json",
    "retrospective_deepmind_provenance": DOCS / "goalB3_deepmind_gcd_div_lcm_provenance_audit.json",
    "deepmind_source_audit": DOCS / "goalB3_deepmind_source_audit.json",
    "adversarial": DOCS / "goalB3_operand_repair.json",
    "qwen_transfer": DOCS / "goalB3_qwen_strict_transfer.json",
    "causal_cross_seed": DOCS / "goalB3_causal_interchange_cross_seed.json",
    "gcd_causal_cross_seed": DOCS / "goalB3_causal_interchange_gcd_cross_seed.json",
}

MIN_BAR = {
    "min_locked_examples": 1000,
    "min_ops": 3,
    "min_lift": 0.20,
    "max_false_fire": 0.01,
    "min_seeds": 3,
    "min_pair_exact": 0.80,
    "min_causal_pairs_per_op": 50,
}

ADVERSARIAL_NEGATIVE_FAMILIES = {
    "quoted_expression_negative",
    "do_not_compute_negative",
    "wrong_op_negative",
    "natural_numeric_negative",
}
ADVERSARIAL_TARGET_FAMILIES = {
    "pre_distractor",
    "between_distractor",
    "post_distractor",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def check_cross_seed(summary: dict[str, Any], required_ops: set[str]) -> dict[str, Any]:
    ops = summary.get("ops", {})
    present = set(ops)
    op_checks = {}
    for op in sorted(required_ops):
        row = ops.get(op, {})
        op_checks[op] = {
            "present": op in ops,
            "n_runs": row.get("n_runs", 0),
            "total_locked": row.get("total_locked", 0),
            "target_locked": row.get("total_target_locked", 0),
            "mean_lift": row.get("mean_lift"),
            "min_lift": row.get("min_lift"),
            "max_false_fire": row.get("max_false_fire"),
            "min_pair_exact_fired": row.get("min_pair_exact_fired"),
            "passes": (
                op in ops
                and row.get("n_runs", 0) >= MIN_BAR["min_seeds"]
                and row.get("min_lift", -1.0) >= MIN_BAR["min_lift"]
                and row.get("max_false_fire", 1.0) <= MIN_BAR["max_false_fire"]
                and row.get("min_pair_exact_fired", 0.0) >= MIN_BAR["min_pair_exact"]
            ),
        }
    return {
        "verdict": summary.get("verdict"),
        "n_runs": summary.get("n_runs"),
        "n_locked_total": summary.get("n_locked_total"),
        "n_target_locked_total": summary.get("n_target_locked_total"),
        "ops_present": sorted(present),
        "ops_required": sorted(required_ops),
        "op_checks": op_checks,
        "passes": (
            summary.get("verdict") == "GOAL_B3_BENCHMARK_CROSS_SEED_PASS"
            and summary.get("n_runs", 0) >= MIN_BAR["min_seeds"]
            and summary.get("n_locked_total", 0) >= MIN_BAR["min_locked_examples"]
            and all(row["passes"] for row in op_checks.values())
        ),
    }


def check_causal_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    per_op: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        for op, row in summary.get("ops", {}).items():
            per_op[op] = {
                "seeds": row.get("n_runs", 0),
                "total_pairs": row.get("total_pairs", 0),
                "min_decoder_follow": row.get("min_decoder_follow", 0.0),
                "min_routed_follow": row.get("min_routed_follow", 0.0),
                "max_random_routed_follow": row.get("max_random_routed_follow", 1.0),
                "passes": bool(row.get("passes")),
            }
    return {
        "summary_verdicts": [summary.get("verdict") for summary in summaries],
        "ops": per_op,
        "passes": (
            bool(per_op)
            and len(per_op) >= 4
            and all(summary.get("verdict") == "CAUSAL_INTERCHANGE_CROSS_SEED_PASS" for summary in summaries)
            and all(row["passes"] for row in per_op.values())
        ),
    }


def check_adversarial(data: dict[str, Any]) -> dict[str, Any]:
    op_checks = {}
    for op, row in data.get("ops", {}).items():
        families = row.get("families", {})
        negative = {
            name: families.get(name, {}).get("fire_rate")
            for name in sorted(ADVERSARIAL_NEGATIVE_FAMILIES)
        }
        target = {
            name: families.get(name, {}).get("routed_exact")
            for name in sorted(ADVERSARIAL_TARGET_FAMILIES)
        }
        op_checks[op] = {
            "negative_fires": negative,
            "distractor_target_exact": target,
            "target_exact": row.get("target_exact"),
            "pair_exact_fired": row.get("pair_exact_fired"),
            "passes": (
                all(v == 0.0 for v in negative.values())
                and all(v is not None and v >= MIN_BAR["min_lift"] for v in target.values())
                and row.get("pair_exact_fired", 0.0) >= MIN_BAR["min_pair_exact"]
            ),
        }
    return {
        "verdict": data.get("verdict"),
        "ops": op_checks,
        "passes": data.get("verdict") == "OPERAND_REPAIR_DISTRACTOR_PASS"
        and len(op_checks) >= MIN_BAR["min_ops"]
        and all(row["passes"] for row in op_checks.values()),
    }


def build_package(paths: dict[str, Path]) -> dict[str, Any]:
    broad = check_cross_seed(read_json(paths["broad_cross_seed"]), {"mul", "div_remainder", "lcm"})
    gcd = check_cross_seed(read_json(paths["gcd_cross_seed"]), {"gcd"})
    deepmind = check_cross_seed(read_json(paths["deepmind_cross_seed"]), {"gcd", "div_remainder", "lcm"})
    causal = check_causal_summaries([read_json(paths["causal_cross_seed"]), read_json(paths["gcd_causal_cross_seed"])])
    adversarial = check_adversarial(read_json(paths["adversarial"]))
    provenance = read_json(paths["deepmind_provenance"])
    qwen = read_json(paths["qwen_transfer"])
    source_audit = read_json(paths["deepmind_source_audit"])

    qwen_passes_as_attempt = qwen.get("verdict") == "STRICT_CROSS_MODEL_READOUT_FAIL" and qwen.get("hard_negative_false_fire") == 0.0
    provenance_passes = (
        provenance.get("verdict") == "PROVENANCE_AUDIT_PASS"
        and provenance.get("n_bad_fired_provenance") == 0
        and provenance.get("n_negative_fires") == 0
    )
    deepmind_mul_limited = source_audit.get("verdict") in {"DEEPMIND_SOURCE_COVERAGE_AUDIT", None}

    gates = {
        "broad_four_op_cross_seed": broad["passes"] and gcd["passes"],
        "recognized_deepmind_three_op_cross_seed": deepmind["passes"],
        "runtime_provenance_records": provenance_passes,
        "scaled_causal_interchange": causal["passes"],
        "adversarial_paraphrase_safety": adversarial["passes"],
        "strict_non_llama_transfer_attempt": qwen_passes_as_attempt,
        "deepmind_mul_source_limitation_documented": bool(deepmind_mul_limited),
    }
    caveats = [
        "DeepMind mul is not a powered recognized-source result under the frozen two-integer route.",
        "Strict Qwen transfer is a falsifier for this answer-site operand route, not a non-Llama positive.",
        "Gcd causal interchange passes the frozen random-control gate, but one seed reaches the max random routed-follow gate exactly (0.10).",
        "The prospective final DeepMind result covers gcd/div_remainder/lcm; the broader four-op result remains an internal synthetic/adversarial benchmark tier.",
    ]
    verdict = "NEURIPS_PACKAGE_MINIMUM_BAR_PASS_WITH_CAVEATS" if all(gates.values()) else "NEURIPS_PACKAGE_INCOMPLETE"

    return {
        "suite": "goalB3_paper_package_audit",
        "verdict": verdict,
        "minimum_bar": MIN_BAR,
        "gates": gates,
        "broad_cross_seed": broad,
        "gcd_cross_seed": gcd,
        "deepmind_cross_seed": deepmind,
        "deepmind_provenance": {
            "verdict": provenance.get("verdict"),
            "n_records": provenance.get("n_records"),
            "n_fired": provenance.get("n_fired"),
            "n_bad_fired_provenance": provenance.get("n_bad_fired_provenance"),
            "n_negative_fires": provenance.get("n_negative_fires"),
            "n_target_nonfires": provenance.get("n_target_nonfires"),
        },
        "causal_interchange": causal,
        "adversarial_paraphrase": adversarial,
        "strict_qwen_transfer": {
            "verdict": qwen.get("verdict"),
            "model_id": qwen.get("model_id"),
            "n_locked": qwen.get("n_locked"),
            "target_exact": qwen.get("target_exact"),
            "pair_exact_on_fired_target": qwen.get("pair_exact_on_fired_target"),
            "hard_negative_false_fire": qwen.get("hard_negative_false_fire"),
        },
        "caveats": caveats,
        "source_paths": {key: str(value) for key, value in paths.items()},
    }


def write_md(package: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal B3 Paper Package Audit",
        "",
        f"- verdict: **{package['verdict']}**",
        "",
        "## Gate Summary",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for gate, passed in package["gates"].items():
        lines.append(f"| `{gate}` | {str(passed)} |")

    lines.extend(
        [
            "",
            "## Benchmark Evidence",
            "",
            "| benchmark | locked | target locked | ops | minimum lift | max false-fire | verdict |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for label, key in [
        ("broad frozen arithmetic/adversarial", "broad_cross_seed"),
        ("gcd broad extension", "gcd_cross_seed"),
        ("DeepMind interpolate recognized", "deepmind_cross_seed"),
    ]:
        section = package[key]
        ops = section["op_checks"]
        min_lift = min(row["min_lift"] for row in ops.values())
        max_false = max(row["max_false_fire"] for row in ops.values())
        lines.append(
            f"| {label} | {section['n_locked_total']} | {section['n_target_locked_total']} | "
            f"{', '.join(section['ops_required'])} | {min_lift:.3f} | {max_false:.3f} | "
            f"`{section['verdict']}` |"
        )

    lines.extend(
        [
            "",
            "## Causal Evidence",
            "",
            "| op | total pairs | min decoder donor-follow | min routed donor-follow | max random routed-follow |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for op, row in sorted(package["causal_interchange"]["ops"].items()):
        lines.append(
            f"| `{op}` | {row['total_pairs']} | {row['min_decoder_follow']:.3f} | "
            f"{row['min_routed_follow']:.3f} | {row['max_random_routed_follow']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Model Attempt",
            "",
            f"- model: `{package['strict_qwen_transfer']['model_id']}`",
            f"- verdict: `{package['strict_qwen_transfer']['verdict']}`",
            f"- target exact: {package['strict_qwen_transfer']['target_exact']:.3f}",
            f"- pair exact fired: {package['strict_qwen_transfer']['pair_exact_on_fired_target']:.3f}",
            f"- hard-negative false-fire: {package['strict_qwen_transfer']['hard_negative_false_fire']:.3f}",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in package["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", type=Path, default=DOCS / "goalB3_paper_package_audit.json")
    parser.add_argument("--out_md", type=Path, default=RESEARCH / "641_goalB3_paper_package_audit_2026-06-02.md")
    args = parser.parse_args()

    package = build_package(DEFAULTS)
    args.out_json.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    write_md(package, args.out_md)
    print(json.dumps({"verdict": package["verdict"], "gates": package["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
