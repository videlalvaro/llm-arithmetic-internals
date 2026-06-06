#!/usr/bin/env python3
"""Goal B2 NeurIPS adversarial/paraphrase audit for the 3-op suite.

Eval-only: uses already fitted per-op readouts/gates/selectors and runs the
same opaque runtime path as the benchmark suite. Prompt text is used only to
construct locked audit examples and to grade records, never inside runtime.
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


OP_CONFIG = {
    "mul": {
        "stem": "goalB2_mul_chunk_frozen",
        "pair_conf": 0.6137514114379884,
        "families": {
            "clean_symbolic": ("target", "What is {a} * {b}? Answer: "),
            "clean_phrase": ("target", "What is {a} times {b}? Answer: "),
            "semantic": ("target", "Find the product of {a} and {b}. Answer: "),
            "story_literal": (
                "target",
                "A shelf has {a} bins with {b} screws each. How many screws are there? Answer: ",
            ),
            "pre_distractor": (
                "target",
                "Ignore catalog number {d}. What is {a} times {b}? Answer: ",
            ),
            "between_distractor": (
                "target",
                "What is {a} times reference number {d}, actually use {b}? Answer: ",
            ),
            "post_distractor": (
                "target",
                "What is {a} * {b}? The receipt number is {d}. Answer: ",
            ),
            "quoted_expression_negative": (
                "negative",
                "The text says \"{a} * {b}\". Do not solve it; classify it. Answer: ",
            ),
            "do_not_compute_negative": (
                "negative",
                "Do not multiply {a} and {b}; say which number is larger. Answer: ",
            ),
            "wrong_op_negative": (
                "negative",
                "Find the least common multiple of {a} and {b}; do not compute the product. Answer: ",
            ),
            "natural_numeric_negative": (
                "negative",
                "Archive page {a} references section {b} before the paragraph continues. Next word: ",
            ),
        },
    },
    "div_remainder": {
        "stem": "goalB2_div_remainder_frozen",
        "pair_conf": 0.2,
        "families": {
            "clean_symbolic": ("target", "What is {a} mod {b}? Answer: "),
            "clean_phrase": (
                "target",
                "What is the remainder when {a} is divided by {b}? Answer: ",
            ),
            "semantic": ("target", "Compute {a} modulo {b}. Answer: "),
            "story_literal": (
                "target",
                "{a} bolts are packed into boxes of {b}. What is the remainder? Answer: ",
            ),
            "pre_distractor": (
                "target",
                "Ignore catalog number {d}. What is {a} mod {b}? Answer: ",
            ),
            "between_distractor": (
                "target",
                "What is {a} modulo reference number {d}, actually use {b}? Answer: ",
            ),
            "post_distractor": (
                "target",
                "What is {a} mod {b}? The receipt number is {d}. Answer: ",
            ),
            "quoted_expression_negative": (
                "negative",
                "The text says \"{a} mod {b}\". Do not solve it; classify it. Answer: ",
            ),
            "do_not_compute_negative": (
                "negative",
                "Do not compute {a} modulo {b}; say which number is larger. Answer: ",
            ),
            "wrong_op_negative": (
                "negative",
                "Find the product of {a} and {b}; do not compute a remainder. Answer: ",
            ),
            "natural_numeric_negative": (
                "negative",
                "The report lists register {a}, bucket {b}, and no arithmetic request. Continue: ",
            ),
        },
    },
    "lcm": {
        "stem": "goalB2_lcm_chunk_frozen",
        "pair_conf": 0.2,
        "families": {
            "clean_symbolic": ("target", "What is lcm({a}, {b})? Answer: "),
            "clean_phrase": (
                "target",
                "What is the least common multiple of {a} and {b}? Answer: ",
            ),
            "semantic": (
                "target",
                "What is the smallest positive integer divisible by both {a} and {b}? Answer: ",
            ),
            "denominator": (
                "target",
                "What common denominator should be used for fractions with denominators {a} and {b}? Answer: ",
            ),
            "pre_distractor": (
                "target",
                "Ignore catalog number {d}. What is lcm({a}, {b})? Answer: ",
            ),
            "between_distractor": (
                "target",
                "What is the lcm of {a} and reference number {d}, actually use {b}? Answer: ",
            ),
            "post_distractor": (
                "target",
                "What is lcm({a}, {b})? The receipt number is {d}. Answer: ",
            ),
            "quoted_expression_negative": (
                "negative",
                "The text says \"lcm({a}, {b})\". Do not solve it; classify it. Answer: ",
            ),
            "do_not_compute_negative": (
                "negative",
                "Do not find the lcm of {a} and {b}; say which number is larger. Answer: ",
            ),
            "wrong_op_negative": (
                "negative",
                "Find the greatest common divisor of {a} and {b}; do not compute the lcm. Answer: ",
            ),
            "natural_numeric_negative": (
                "negative",
                "Archive page {a} references section {b}; no arithmetic request follows. Next word: ",
            ),
        },
    },
    "gcd": {
        "stem": "goalB2_gcd_chunk_frozen",
        "pair_conf": 0.2,
        "families": {
            "clean_symbolic": ("target", "What is gcd({a}, {b})? Answer: "),
            "clean_phrase": (
                "target",
                "What is the greatest common divisor of {a} and {b}? Answer: ",
            ),
            "semantic": (
                "target",
                "What is the largest positive integer that divides both {a} and {b}? Answer: ",
            ),
            "story_literal": (
                "target",
                "{a} tiles and {b} tiles must be split into equal groups with no leftovers. What is the largest group size? Answer: ",
            ),
            "pre_distractor": (
                "target",
                "Ignore catalog number {d}. What is gcd({a}, {b})? Answer: ",
            ),
            "between_distractor": (
                "target",
                "What is the gcd of {a} and reference number {d}, actually use {b}? Answer: ",
            ),
            "post_distractor": (
                "target",
                "What is gcd({a}, {b})? The receipt number is {d}. Answer: ",
            ),
            "quoted_expression_negative": (
                "negative",
                "The text says \"gcd({a}, {b})\". Do not solve it; classify it. Answer: ",
            ),
            "do_not_compute_negative": (
                "negative",
                "Do not find the gcd of {a} and {b}; say which number is larger. Answer: ",
            ),
            "wrong_op_negative": (
                "negative",
                "Find the least common multiple of {a} and {b}; do not compute the gcd. Answer: ",
            ),
            "natural_numeric_negative": (
                "negative",
                "Archive page {a} references section {b}; no arithmetic request follows. Next word: ",
            ),
        },
    },
}


def compute(op: str, a: int, b: int) -> int:
    return b2.compute_target(op, a, b)


def build_examples(op: str, seed: int, n_per_family: int, tok: Any) -> list[b2.Example]:
    rng = np.random.default_rng(seed + abs(hash(op)) % 10000)
    rows = []
    for family, (kind, template) in OP_CONFIG[op]["families"].items():
        for _ in range(n_per_family):
            a = int(rng.integers(1000, 9999))
            b_lo = 2 if op == "div_remainder" else 1000
            b = int(rng.integers(b_lo, 9999))
            d = int(rng.integers(1000, 9999))
            if d in {a, b}:
                d = (d + 137) % 9999
            is_target = int(kind == "target")
            rows.append(
                {
                    "split": "locked_test",
                    "family": family,
                    "op": op if is_target else "natural",
                    "is_lcm": int(op == "lcm" and is_target),
                    "is_target": is_target,
                    "a": a,
                    "b": b,
                    "answer": compute(op, a, b) if is_target else None,
                    "prompt": template.format(a=a, b=b, d=d),
                    "source": "goalB2_neurips_adversarial_audit",
                }
            )
    return b2.rows_to_examples(rows, seed, "llama", tok)


def work_dir_for_op(base: Path, op: str, seed: int) -> Path:
    return base / f"{op}_seed_{seed}"


def load_runtime_components(base: Path, op: str, seed: int) -> tuple[
    b2.Readouts, b2.SafeGateReadout, b2.ChunkPairSelector
]:
    stem = OP_CONFIG[op]["stem"]
    root = work_dir_for_op(base, op, seed)
    readouts = b2.Readouts.load(root / f"{stem}_readouts.npz")
    readouts.pair_conf_threshold = float(OP_CONFIG[op]["pair_conf"])
    safe_gate = b2.SafeGateReadout.load(root / f"{stem}_safe_gate.npz")
    selector = b2.ChunkPairSelector.load(root / f"{stem}_chunk_selector.npz")
    return readouts, safe_gate, selector


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
                "pair_exact_fired": float(np.mean([r["decoded_pair_exact"] for r in fired]))
                if fired
                else float("nan"),
                "routed_exact": float(np.mean([r["readout_routing_correct"] for r in rows]))
                if rows
                else float("nan"),
            }
        out[op] = {
            "n": len(op_rows),
            "target_exact": float(np.mean([r["readout_routing_correct"] for r in target]))
            if target
            else float("nan"),
            "target_fire_rate": float(np.mean([r["fired"] for r in target]))
            if target
            else float("nan"),
            "negative_fire_rate": float(np.mean([r["fired"] for r in neg]))
            if neg
            else float("nan"),
            "families": by_family,
        }
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B2 NeurIPS Adversarial / Paraphrase Audit",
        "",
        "Eval-only audit using fitted 3-op Goal B2 suite artifacts. Runtime receives only opaque token IDs and activations.",
        "",
        "## Aggregate",
        "",
        "| op | n | target exact | target fire | negative fire |",
        "|---|---:|---:|---:|---:|",
    ]
    for op, row in payload["ops"].items():
        lines.append(
            f"| `{op}` | {row['n']} | {row['target_exact']:.3f} | "
            f"{row['target_fire_rate']:.3f} | {row['negative_fire_rate']:.3f} |"
        )
    for op, row in payload["ops"].items():
        lines.extend(
            [
                "",
                f"## {op}",
                "",
                "| family | target | n | fire | pair exact fired | routed exact |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for family, frow in row["families"].items():
            lines.append(
                f"| `{family}` | {frow['is_target']} | {frow['n']} | "
                f"{frow['fire_rate']:.3f} | {frow['pair_exact_fired']:.3f} | "
                f"{frow['routed_exact']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--audit_seed", type=int, default=651)
    p.add_argument("--n_per_family", type=int, default=20)
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--out_json", default=str(DOCS / "goalB2_neurips_adversarial_audit.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB2_neurips_adversarial_audit.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB2_neurips_adversarial_audit_records.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    records = []
    for op in args.ops:
        readouts, safe_gate, selector = load_runtime_components(base, op, args.artifact_seed)
        guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=op)
        for ex in build_examples(op, args.audit_seed, args.n_per_family, tok):
            runtime = b2.runtime_from_example(ex, args.audit_seed, "llama", model, device)
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
        "audit_seed": args.audit_seed,
        "n_per_family": args.n_per_family,
        "ops": summarize(records),
        "records_path": args.out_records,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({op: payload["ops"][op] for op in args.ops}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
