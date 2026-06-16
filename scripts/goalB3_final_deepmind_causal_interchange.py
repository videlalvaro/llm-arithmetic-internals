#!/usr/bin/env python3
"""Final DeepMind causal-interchange runner/planner for Goal B3.

This runner is intentionally stricter than the earlier synthetic causal
interchange script: candidate pairs must come from final DeepMind recognized
records and must already be target, fired, pair-exact, and carry selected chunk
diagnostics. Full patch execution still delegates to
``goalB3_causal_interchange.py`` because that script owns the L22 activation
patching implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
PYTHON = os.environ.get("RUNE_PYTHON", sys.executable)
OPS = ("gcd", "div_remainder", "lcm")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def selected_groups(diag: dict[str, Any]) -> list[Any]:
    groups = diag.get("selected_groups")
    if groups is None:
        groups = diag.get("chunk_pair_selector", {}).get("selected_groups")
    if groups is None:
        groups = diag.get("chunk_groups")
    if groups is None:
        groups = diag.get("groups")
    return groups if isinstance(groups, list) else []


def causal_candidates(records: list[dict[str, Any]], op: str) -> list[dict[str, Any]]:
    """Return only records eligible for final causal interchange."""
    out = []
    for rec in records:
        if rec.get("target_op") != op and rec.get("op") != op:
            continue
        if not rec.get("is_target"):
            continue
        if not rec.get("fired"):
            continue
        if not rec.get("decoded_pair_exact"):
            continue
        diag = rec.get("readout_diagnostics")
        if not isinstance(diag, dict) or len(selected_groups(diag)) < 2:
            continue
        out.append(rec)
    return out


def candidate_summary(record_paths: list[Path]) -> dict[str, Any]:
    by_seed: dict[str, dict[str, int]] = {}
    totals = {op: 0 for op in OPS}
    for path in record_paths:
        rows = load_jsonl(path)
        seed_summary = {}
        for op in OPS:
            n = len(causal_candidates(rows, op))
            seed_summary[op] = n
            totals[op] += n
        by_seed[path.stem] = seed_summary
    return {"by_record_file": by_seed, "totals": totals}


def build_delegate_command(args: argparse.Namespace, seed: int, records: Path) -> list[str]:
    return [
        PYTHON,
        "scripts/goalB3_causal_interchange.py",
        "--ops",
        *OPS,
        "--fit_seed",
        str(seed),
        "--eval_seed",
        str(seed),
        "--split_source",
        "deepmind_interpolate",
        "--candidate_records",
        str(records),
        "--include_common_denominator",
        "--n_pairs_per_op",
        str(args.n_pairs_per_op),
        "--min_follow",
        str(args.min_follow),
        "--max_random_follow",
        str(args.max_random_follow),
        "--out_json",
        str(DOCS / f"goalB3_final_deepmind_causal_interchange_seed{seed}.json"),
        "--out_md",
        str(DOCS / f"goalB3_final_deepmind_causal_interchange_seed{seed}.md"),
        "--out_records",
        str(DOCS / f"goalB3_final_deepmind_causal_interchange_seed{seed}_records.jsonl"),
    ]


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Final DeepMind Causal Interchange",
        "",
        f"- verdict: **{payload['verdict']}**",
        "- scope: candidate/pairing readiness only; executed causal gate status must be read from the cross-seed powered summary",
        f"- required pairs per op: {payload['required_pairs_per_op']}",
        "",
        "## Candidate Counts",
        "",
        "| record file | gcd | div_remainder | lcm |",
        "|---|---:|---:|---:|",
    ]
    for name, row in payload["candidate_summary"]["by_record_file"].items():
        lines.append(f"| `{name}` | {row['gcd']} | {row['div_remainder']} | {row['lcm']} |")
    lines.extend(["", "## Delegate Commands", ""])
    for cmd in payload["delegate_commands"]:
        lines.extend(["```bash", cmd, "```", ""])
    lines.append(
        "Candidates are restricted to final DeepMind records that are target, fired, "
        "pair-exact, and include selected L22 chunk diagnostics."
    )
    lines.extend(
        [
            "",
            "This plan does not prove the final causal gate. The executed final DeepMind",
            "causal runs are aggregated by",
            "`docs/goalB3_final_deepmind_causal_interchange_cross_seed_powered.{json,md}`,",
            "which enforces the frozen pair-count and donor-follow gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--records",
        nargs="+",
        type=Path,
        default=[
            DOCS / "goalB3_final_deepmind_interpolate_recognized_seed911_records.jsonl",
            DOCS / "goalB3_final_deepmind_interpolate_recognized_seed921_records.jsonl",
            DOCS / "goalB3_final_deepmind_interpolate_recognized_seed931_records.jsonl",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[911, 921, 931])
    p.add_argument("--n_pairs_per_op", type=int, default=50)
    p.add_argument("--min_follow", type=float, default=0.95)
    p.add_argument("--max_random_follow", type=float, default=0.10)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--out_json", type=Path, default=DOCS / "goalB3_final_deepmind_causal_interchange_plan.json")
    p.add_argument("--out_md", type=Path, default=DOCS / "goalB3_final_deepmind_causal_interchange_plan.md")
    args = p.parse_args()

    summary = candidate_summary(args.records)
    coverage_ok = all(n >= args.n_pairs_per_op * 2 for n in summary["totals"].values())
    commands = [
        build_delegate_command(args, seed, records)
        for seed, records in zip(args.seeds, args.records, strict=True)
    ]
    payload = {
        "suite": "goalB3_final_deepmind_causal_interchange",
        "records": [str(pth) for pth in args.records],
        "candidate_summary": summary,
        "required_pairs_per_op": args.n_pairs_per_op,
        "coverage_ok_for_pairing": coverage_ok,
        "delegate_commands": [" ".join(shlex.quote(part) for part in cmd) for cmd in commands],
        "verdict": "FINAL_DEEPMIND_CAUSAL_PAIRING_READY" if coverage_ok else "FINAL_DEEPMIND_CAUSAL_PAIRING_UNDERCOVERED",
    }
    if args.execute:
        completed = []
        for cmd in commands:
            subprocess.run(cmd, cwd=REPO, check=True)
            completed.append(cmd[-1])
        payload["executed"] = completed
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(args.out_md, payload)
    print(json.dumps({"verdict": payload["verdict"], "coverage": summary["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
