#!/usr/bin/env python3
"""Emit manifest-frozen Goal B3 final rerun commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any


PYTHON = os.environ.get("RUNE_PYTHON", sys.executable)
DOCS = Path("docs")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def q(arg: object) -> str:
    return shlex.quote(str(arg))


def pair_threshold_args(manifest: dict[str, Any]) -> list[str]:
    thresholds = manifest["frozen_thresholds"]["pair_conf_thresholds"]
    return [f"--pair_threshold {q(f'{op}={value:.2f}')}" for op, value in thresholds.items()]


def common_benchmark_args(manifest: dict[str, Any]) -> list[str]:
    f = manifest["frozen_thresholds"]
    g = manifest["acceptance_gates"]
    return [
        f"--op_threshold_min {f['op_threshold_min']}",
        f"--op_neg_margin {f['op_neg_margin']}",
        f"--safe_threshold_min {f['safe_threshold_min']}",
        f"--safe_neg_margin {f['safe_neg_margin']}",
        *pair_threshold_args(manifest),
        f"--chunk_top_k {f['chunk_top_k']}",
        f"--chunk_window {f['chunk_window']}",
        f"--chunk_pos_threshold {f['chunk_pos_threshold']}",
        f"--chunk_value_margin_threshold {f['chunk_value_margin_threshold']}",
        f"--operand_lo {f['operand_lo']}",
        f"--operand_hi {f['operand_hi']}",
        f"--max_new_tokens {f['max_new_tokens']}",
        f"--min_locked {g['min_locked_examples']}",
        "--min_target_per_op 50",
        f"--min_ops {g['min_ops']}",
        f"--min_lift {g['min_lift']}",
        f"--max_false_fire {g['max_false_fire']}",
        f"--min_pair_exact_fired {g['min_pair_exact_fired']}",
    ]


def benchmark_command(
    manifest: dict[str, Any],
    tier_name: str,
    seed: int,
    *,
    dm_dir: str | None,
    output_prefix: str,
) -> tuple[str, dict[str, Any]]:
    tier = manifest["benchmark_tiers"][tier_name]
    out_stem = f"{output_prefix}_{tier_name}_seed{seed}"
    args = [
        q(PYTHON),
        q(tier["runner"]),
        "--ops",
        *[q(op) for op in tier["ops"]],
        f"--seed {seed}",
        f"--split_source {q(tier['split_source'])}",
        f"--n_per_family {tier['n_per_family']}",
        f"--fit_b3_aug_n_per_family {tier['fit_b3_aug_n_per_family']}",
        *common_benchmark_args(manifest),
        f"--out_json {q(DOCS / f'{out_stem}.json')}",
        f"--out_md {q(DOCS / f'{out_stem}.md')}",
        f"--out_records {q(DOCS / f'{out_stem}_records.jsonl')}",
        f"--out_replay_bundles {q(DOCS / f'{out_stem}_replay_bundles.jsonl')}",
    ]
    if tier_name == "broad_frozen_arithmetic_adversarial":
        args.append(f"--n_adversarial_per_family {tier['n_adversarial_per_family']}")
    if tier_name == "deepmind_interpolate_recognized":
        args.extend(
            [
                f"--n_natural {tier['n_natural']}",
                f"--dm_scan_limit {tier['dm_scan_limit']}",
            ]
        )
        if tier.get("include_common_denominator"):
            args.append("--include_common_denominator")
        if dm_dir:
            args.append(f"--dm_dir {q(dm_dir)}")
    return " ".join(args), {
        "kind": "benchmark",
        "tier": tier_name,
        "seed": seed,
        "out_json": str(DOCS / f"{out_stem}.json"),
        "out_md": str(DOCS / f"{out_stem}.md"),
        "out_records": str(DOCS / f"{out_stem}_records.jsonl"),
        "out_replay_bundles": str(DOCS / f"{out_stem}_replay_bundles.jsonl"),
    }


def aggregate_command(
    manifest: dict[str, Any],
    tier_name: str,
    run_specs: list[dict[str, Any]],
    *,
    output_prefix: str,
) -> tuple[str, dict[str, Any]]:
    out_stem = f"{output_prefix}_{tier_name}_cross_seed"
    g = manifest["acceptance_gates"]
    args = [
        q(PYTHON),
        "scripts/goalB3_benchmark_cross_seed_summary.py",
        *[q(spec["out_json"]) for spec in run_specs],
        f"--min_seeds {g['min_seeds']}",
        f"--min_lift {g['min_lift']}",
        f"--max_false_fire {g['max_false_fire']}",
        f"--out_json {q(DOCS / f'{out_stem}.json')}",
        f"--out_md {q(DOCS / f'{out_stem}.md')}",
    ]
    return " ".join(args), {
        "kind": "benchmark_aggregate",
        "tier": tier_name,
        "out_json": str(DOCS / f"{out_stem}.json"),
        "out_md": str(DOCS / f"{out_stem}.md"),
    }


def provenance_command(run_specs: list[dict[str, Any]], *, output_prefix: str) -> tuple[str, dict[str, Any]]:
    out_stem = f"{output_prefix}_deepmind_provenance"
    args = [
        q(PYTHON),
        "scripts/goalB3_provenance_audit.py",
        *[q(spec["out_records"]) for spec in run_specs],
        f"--out_json {q(DOCS / f'{out_stem}.json')}",
        f"--out_md {q(DOCS / f'{out_stem}.md')}",
    ]
    return " ".join(args), {
        "kind": "provenance_audit",
        "out_json": str(DOCS / f"{out_stem}.json"),
        "out_md": str(DOCS / f"{out_stem}.md"),
    }


def replay_provenance_command(run_specs: list[dict[str, Any]], *, tier_name: str, output_prefix: str) -> tuple[str, dict[str, Any]]:
    out_stem = f"{output_prefix}_{tier_name}_replay_provenance_full"
    args = [
        q(PYTHON),
        "scripts/goalB3_replay_provenance_audit.py",
        *[q(spec["out_replay_bundles"]) for spec in run_specs],
        "--require_full",
        "--min_bundles 1",
        f"--out_json {q(DOCS / f'{out_stem}.json')}",
        f"--out_md {q(DOCS / f'{out_stem}.md')}",
    ]
    return " ".join(args), {
        "kind": "replay_provenance_audit",
        "tier": tier_name,
        "out_json": str(DOCS / f"{out_stem}.json"),
        "out_md": str(DOCS / f"{out_stem}.md"),
    }


def build_plan(manifest: dict[str, Any], *, dm_dir: str | None, output_prefix: str) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    tier_run_specs: dict[str, list[dict[str, Any]]] = {}
    for tier_name, tier in manifest["benchmark_tiers"].items():
        tier_run_specs[tier_name] = []
        for seed in tier["seeds"]:
            cmd, spec = benchmark_command(manifest, tier_name, seed, dm_dir=dm_dir, output_prefix=output_prefix)
            commands.append({"name": f"{tier_name}_seed{seed}", "command": cmd, **spec})
            tier_run_specs[tier_name].append(spec)
        cmd, spec = aggregate_command(manifest, tier_name, tier_run_specs[tier_name], output_prefix=output_prefix)
        commands.append({"name": f"{tier_name}_aggregate", "command": cmd, **spec})
        cmd, spec = replay_provenance_command(tier_run_specs[tier_name], tier_name=tier_name, output_prefix=output_prefix)
        commands.append({"name": f"{tier_name}_replay_provenance_full", "command": cmd, **spec})
    deepmind_specs = tier_run_specs.get("deepmind_interpolate_recognized", [])
    if deepmind_specs:
        cmd, spec = provenance_command(deepmind_specs, output_prefix=output_prefix)
        commands.append({"name": "deepmind_provenance", "command": cmd, **spec})
    return {
        "suite": "goalB3_final_rerun_plan",
        "manifest_status": manifest.get("status"),
        "output_prefix": output_prefix,
        "dm_dir_source": "argument_or_env" if dm_dir else "runner_default_or_env_at_execution",
        "dm_dir": dm_dir,
        "commands": commands,
    }


def write_md(plan: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal B3 Final Rerun Plan",
        "",
        f"- output prefix: `{plan['output_prefix']}`",
        f"- DeepMind dir source: `{plan['dm_dir_source']}`",
        "",
        "## Commands",
        "",
    ]
    for idx, item in enumerate(plan["commands"], start=1):
        lines.extend([f"### {idx}. {item['name']}", "", "```bash", item["command"], "```", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DOCS / "goalB3_final_frozen_manifest.json")
    parser.add_argument("--dm_dir", default=os.environ.get("DEEPMIND_MATH_INTERPOLATE_DIR", ""))
    parser.add_argument("--output_prefix", default="goalB3_final")
    parser.add_argument("--out_json", type=Path, default=DOCS / "goalB3_final_rerun_plan.json")
    parser.add_argument("--out_md", type=Path, default=Path("docs/research/644_goalB3_final_rerun_plan_2026-06-02.md"))
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    plan = build_plan(manifest, dm_dir=args.dm_dir or None, output_prefix=args.output_prefix)
    args.out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_md(plan, args.out_md)
    print(json.dumps({"commands": len(plan["commands"]), "out_json": str(args.out_json)}))


if __name__ == "__main__":
    main()
