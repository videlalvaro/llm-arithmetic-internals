#!/usr/bin/env python3
"""Audit DeepMind Mathematics source coverage for Goal B3.

This is not a runtime benchmark and does not make a Goal B claim. It scans
available DeepMind Mathematics files and reports how many examples survive the
same operand/value filters used by the repaired Goal B3 route. The purpose is
to distinguish a model failure from a benchmark-source coverage limitation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DEFAULT_DM_ROOT = Path.home() / "deepmind_math" / "mathematics_dataset-v1.0"


def read_pairs(path: Path, limit: int | None) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        pairs.append((lines[i], lines[i + 1]))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def two_ints(prompt: str) -> tuple[int, int] | None:
    nums = re.findall(r"-?\d+", prompt)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def lcm_operands(prompt: str) -> tuple[int, int] | None:
    common_den = re.search(
        r"common denominator of\s*(-?\d+)\s*/\s*(-?\d+)\s+and\s*(-?\d+)\s*/\s*(-?\d+)",
        prompt,
        re.I,
    )
    if common_den:
        return int(common_den.group(2)), int(common_den.group(4))
    return two_ints(prompt)


def safe_lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def target_value(op: str, a: int, b: int) -> int:
    if op == "mul":
        return a * b
    if op == "div_remainder":
        if b == 0:
            raise ValueError("zero divisor")
        return a % b
    if op == "lcm":
        return safe_lcm(a, b)
    if op == "gcd":
        return math.gcd(a, b)
    raise ValueError(op)


def classify_file(path: Path) -> str | None:
    name = path.name
    if name.startswith("arithmetic__mul"):
        return "mul"
    if name == "numbers__div_remainder.txt":
        return "div_remainder"
    if name == "numbers__lcm.txt":
        return "lcm"
    if name == "numbers__gcd.txt":
        return "gcd"
    return None


def audit_file(path: Path, op: str, args: argparse.Namespace) -> dict[str, Any]:
    pairs = read_pairs(path, args.scan_limit)
    reasons = {
        "non_integer_gold": 0,
        "decimal_prompt": 0,
        "no_two_ints": 0,
        "operand_range": 0,
        "answer_mismatch": 0,
        "zero_divisor": 0,
        "accepted": 0,
    }
    examples: list[dict[str, Any]] = []
    for prompt, gold in pairs:
        if not gold.lstrip("-").isdigit():
            reasons["non_integer_gold"] += 1
            continue
        if op == "mul" and "." in prompt:
            reasons["decimal_prompt"] += 1
            continue
        operands = lcm_operands(prompt) if op == "lcm" else two_ints(prompt)
        if operands is None:
            reasons["no_two_ints"] += 1
            continue
        a, b = operands
        b_lo = max(1, args.operand_lo) if op == "div_remainder" else args.operand_lo
        if not (args.operand_lo <= a <= args.operand_hi and b_lo <= b <= args.operand_hi):
            reasons["operand_range"] += 1
            continue
        try:
            expected = target_value(op, a, b)
        except ValueError:
            reasons["zero_divisor"] += 1
            continue
        answer = int(gold)
        if expected != answer:
            reasons["answer_mismatch"] += 1
            continue
        reasons["accepted"] += 1
        if len(examples) < args.example_limit:
            examples.append({"prompt": prompt, "gold": answer, "a": a, "b": b})
    return {
        "path": str(path),
        "split": path.parent.name,
        "file": path.name,
        "op": op,
        "scanned_pairs": len(pairs),
        "accepted": reasons["accepted"],
        "locked_40pct_estimate": int(math.floor(reasons["accepted"] * 0.40)),
        "reasons": reasons,
        "examples": examples,
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 DeepMind Source Audit",
        "",
        "Eval-only coverage audit. This does not run the activation route and does not make a Goal B claim.",
        "",
        f"- DeepMind root: `{payload['dm_root']}`",
        f"- scan limit per file: {payload['scan_limit']}",
        f"- operand range: [{payload['operand_lo']}, {payload['operand_hi']}]",
        "",
        "## Coverage",
        "",
        "| split | file | op | scanned | accepted | locked 40% estimate | top rejection |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["files"]:
        reject = {k: v for k, v in row["reasons"].items() if k != "accepted"}
        top = max(reject.items(), key=lambda kv: kv[1]) if reject else ("none", 0)
        lines.append(
            f"| `{row['split']}` | `{row['file']}` | `{row['op']}` | "
            f"{row['scanned_pairs']} | {row['accepted']} | {row['locked_40pct_estimate']} | "
            f"`{top[0]}`={top[1]} |"
        )
    lines.extend(["", "## Interpretation", ""])
    enough = [
        row for row in payload["files"] if row["locked_40pct_estimate"] >= payload["min_target_per_op"]
    ]
    if enough:
        lines.append(
            f"- {len(enough)} source files appear to have enough supported examples for the "
            "current target-count gate before fitting/eval."
        )
    else:
        lines.append(
            "- No audited source file appears to have enough supported examples for the current "
            "target-count gate under the frozen operand/value filters."
        )
    lines.append(
        "- If `arithmetic__mul.txt` remains sparse, a claim-bearing DeepMind 3-op pass must "
        "either expand the supported operand/value regime before preregistration or use a "
        "different recognized source with enough target coverage."
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dm_root", default=str(DEFAULT_DM_ROOT))
    p.add_argument("--scan_limit", type=int, default=200000)
    p.add_argument("--operand_lo", type=int, default=0)
    p.add_argument("--operand_hi", type=int, default=9999)
    p.add_argument("--min_target_per_op", type=int, default=50)
    p.add_argument("--example_limit", type=int, default=3)
    p.add_argument("--out_json", default=str(REPO / "docs" / "goalB3_deepmind_source_audit.json"))
    p.add_argument("--out_md", default=str(REPO / "docs" / "goalB3_deepmind_source_audit.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dm_root)
    files = []
    for split in ("interpolate", "extrapolate"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.txt")):
            op = classify_file(path)
            if op is None:
                continue
            files.append(audit_file(path, op, args))
    payload = {
        "audit": "goalB3_deepmind_source_audit",
        "dm_root": str(root),
        "scan_limit": args.scan_limit,
        "operand_lo": args.operand_lo,
        "operand_hi": args.operand_hi,
        "min_target_per_op": args.min_target_per_op,
        "files": files,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({f"{r['split']}/{r['file']}": r["accepted"] for r in files}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
