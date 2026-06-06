#!/usr/bin/env python3
"""Summarize independent Goal B3 hard-negative families from final records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORY_PATTERNS = {
    "quoted_arithmetic": ("quoted",),
    "do_not_compute": ("do_not",),
    "wrong_op_same_numbers": ("surface", "wrong_op"),
    "tables_logs_code_invoices": ("table", "code", "invoice", "log"),
    "distractor_heavy_number_text": ("natural", "chunk"),
    "decimals_signs_out_of_domain": ("decimal", "negative_operand", "negative_mod"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def category_for_family(family: str) -> str:
    family_l = family.lower()
    for category, needles in CATEGORY_PATTERNS.items():
        if any(needle in family_l for needle in needles):
            return category
    return "other_hard_negative"


def summarize(paths: list[Path], *, max_false_fire: float = 0.01) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "fires": 0, "families": defaultdict(lambda: {"n": 0, "fires": 0})})
    n_records = 0
    n_negative = 0
    for path in paths:
        for rec in load_jsonl(path):
            n_records += 1
            if rec.get("is_target"):
                continue
            n_negative += 1
            family = str(rec.get("family") or "unknown")
            category = category_for_family(family)
            fired = bool(rec.get("fired"))
            by_category[category]["n"] += 1
            by_category[category]["fires"] += int(fired)
            by_category[category]["families"][family]["n"] += 1
            by_category[category]["families"][family]["fires"] += int(fired)
    categories: dict[str, Any] = {}
    for category, row in sorted(by_category.items()):
        rate = row["fires"] / row["n"] if row["n"] else 0.0
        categories[category] = {
            "n": row["n"],
            "fires": row["fires"],
            "false_fire_rate": rate,
            "passes": rate <= max_false_fire,
            "families": {
                family: {
                    "n": sub["n"],
                    "fires": sub["fires"],
                    "false_fire_rate": sub["fires"] / sub["n"] if sub["n"] else 0.0,
                }
                for family, sub in sorted(row["families"].items())
            },
        }
    required_categories = sorted(CATEGORY_PATTERNS)
    missing_categories = [cat for cat in required_categories if cat not in categories]
    all_rates_pass = all(row["passes"] for row in categories.values())
    verdict = "INDEPENDENT_HARD_NEGATIVE_PASS" if n_negative and not missing_categories and all_rates_pass else "INDEPENDENT_HARD_NEGATIVE_INCOMPLETE_OR_FAIL"
    return {
        "suite": "goalB3_independent_hard_negative_summary",
        "verdict": verdict,
        "input_paths": [str(path) for path in paths],
        "max_false_fire": max_false_fire,
        "n_records": n_records,
        "n_negative": n_negative,
        "required_categories": required_categories,
        "missing_categories": missing_categories,
        "categories": categories,
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal B3 Independent Hard-Negative Summary",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- records: {payload['n_records']}",
        f"- negative records: {payload['n_negative']}",
        f"- max false-fire gate: {payload['max_false_fire']}",
        "",
        "| category | n | fires | false-fire rate |",
        "|---|---:|---:|---:|",
    ]
    for category, row in sorted(payload["categories"].items()):
        lines.append(f"| `{category}` | {row['n']} | {row['fires']} | {row['false_fire_rate']:.4f} |")
    if payload["missing_categories"]:
        lines.extend(["", "Missing categories:", ""])
        lines.extend(f"- `{cat}`" for cat in payload["missing_categories"])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", nargs="+", type=Path)
    p.add_argument("--max_false_fire", type=float, default=0.01)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_md", type=Path, required=True)
    args = p.parse_args()
    payload = summarize(args.records, max_false_fire=args.max_false_fire)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, args.out_md)
    print(json.dumps({"verdict": payload["verdict"], "n_negative": payload["n_negative"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
