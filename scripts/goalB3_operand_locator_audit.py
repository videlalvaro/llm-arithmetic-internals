#!/usr/bin/env python3
"""Goal B3 operand-locator audit for distractor robustness.

This is an eval-only audit of the Llama L22 chunk operand locator used by the
Goal B2/B3 runtime. It answers two questions separately:

1. Candidate coverage: do the activation-derived chunk candidates contain the
   grading operand pair at all?
2. Selector correctness: if the pair is present, does the learned pair selector
   choose it?

Prompt text is used only to construct locked audit examples and to grade
candidate/selector correctness. Runtime locator inputs are opaque prompt ids
and captured activations.
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
from goalB2_neurips_adversarial_audit import OP_CONFIG, build_examples, work_dir_for_op  # noqa: E402


def load_selector(base: Path, op: str, seed: int) -> b2.ChunkPairSelector:
    stem = OP_CONFIG[op]["stem"]
    root = work_dir_for_op(base, op, seed)
    return b2.ChunkPairSelector.load(root / f"{stem}_chunk_selector.npz")


def group_value(group: dict[str, Any]) -> int:
    return int(group["value"])


def pair_matches(op: str, left: int, right: int, a: int, b: int) -> bool:
    if op == "div_remainder":
        return left == a and right == b
    return {left, right} == {a, b}


def select_pair(
    groups: list[dict[str, Any]],
    selector: b2.ChunkPairSelector,
    seq_len: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float | None, int]:
    ordered = sorted(groups, key=lambda g: g["positions"][0])
    pair_scores = []
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            base_feat = b2._chunk_pair_feature(ordered[i], ordered[j], seq_len)
            feat = (
                base_feat
                if len(selector.w) == len(base_feat)
                else b2._chunk_pair_feature(ordered[i], ordered[j], seq_len, include_embeddings=True)
            )
            pair_scores.append((selector.score(feat), ordered[i], ordered[j]))
    pair_scores.sort(key=lambda item: -float(item[0]))
    if not pair_scores:
        return None, None, None, 0
    score, left, right = pair_scores[0]
    return left, right, float(score), len(pair_scores)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for op in sorted({r["target_op"] for r in records}):
        op_rows = [r for r in records if r["target_op"] == op]
        target_rows = [r for r in op_rows if r["is_target"]]
        by_family = {}
        for family in OP_CONFIG[op]["families"]:
            rows = [r for r in op_rows if r["family"] == family]
            targets = [r for r in rows if r["is_target"]]
            by_family[family] = {
                "n": len(rows),
                "is_target": bool(rows and rows[0]["is_target"]),
                "mean_n_groups": float(np.mean([r["n_groups"] for r in rows])) if rows else float("nan"),
                "candidate_pair_present": (
                    float(np.mean([r["candidate_pair_present"] for r in targets]))
                    if targets
                    else float("nan")
                ),
                "selector_pair_exact": (
                    float(np.mean([r["selector_pair_exact"] for r in targets]))
                    if targets
                    else float("nan")
                ),
                "selector_fire_like_rate": (
                    float(np.mean([r["selected_pair"] is not None for r in rows]))
                    if rows
                    else float("nan")
                ),
            }
        out[op] = {
            "n": len(op_rows),
            "n_target": len(target_rows),
            "target_candidate_pair_present": (
                float(np.mean([r["candidate_pair_present"] for r in target_rows]))
                if target_rows
                else float("nan")
            ),
            "target_selector_pair_exact": (
                float(np.mean([r["selector_pair_exact"] for r in target_rows]))
                if target_rows
                else float("nan")
            ),
            "families": by_family,
        }
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Operand Locator Audit",
        "",
        "Eval-only audit of the activation-derived L22 chunk locator. Prompt text is used only for construction/grading.",
        "",
        "## Aggregate",
        "",
        "| op | n target | candidate pair present | selector pair exact |",
        "|---|---:|---:|---:|",
    ]
    for op, row in payload["ops"].items():
        lines.append(
            f"| `{op}` | {row['n_target']} | "
            f"{row['target_candidate_pair_present']:.3f} | "
            f"{row['target_selector_pair_exact']:.3f} |"
        )
    for op, row in payload["ops"].items():
        lines.extend(
            [
                "",
                f"## {op}",
                "",
                "| family | target | n | groups | candidate present | selector exact | selectable |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for family, frow in row["families"].items():
            lines.append(
                f"| `{family}` | {frow['is_target']} | {frow['n']} | "
                f"{frow['mean_n_groups']:.2f} | {frow['candidate_pair_present']:.3f} | "
                f"{frow['selector_pair_exact']:.3f} | {frow['selector_fire_like_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "- If candidate pair is absent, the operand carrier/localizer failed before selector scoring.",
            "- If candidate pair is present but selector exact is low, the learned pair selector failed.",
            "- This audit does not claim routed answer improvement; it diagnoses the operand locator used by routing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--audit_seed", type=int, default=671)
    p.add_argument("--n_per_family", type=int, default=20)
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--chunk_top_k", type=int, default=12)
    p.add_argument("--chunk_window", type=int, default=1)
    p.add_argument("--chunk_pos_threshold", type=float, default=0.5)
    p.add_argument("--chunk_value_margin_threshold", type=float, default=0.0)
    p.add_argument("--out_json", default=str(DOCS / "goalB3_operand_locator_audit.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_operand_locator_audit.md"))
    p.add_argument("--out_records", default=str(DOCS / "goalB3_operand_locator_audit_records.jsonl"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    records = []
    for op in args.ops:
        selector = load_selector(base, op, args.artifact_seed)
        for ex in build_examples(op, args.audit_seed, args.n_per_family, tok):
            runtime = b2.runtime_from_example(ex, args.audit_seed, "llama", model, device)
            groups, diag = b2.chunk_group_candidates(
                runtime,
                chunk_probe,
                chunk_top_k=args.chunk_top_k,
                chunk_window=args.chunk_window,
                chunk_pos_threshold=args.chunk_pos_threshold,
                chunk_value_margin_threshold=args.chunk_value_margin_threshold,
            )
            selected_left, selected_right, selected_score, n_pairs = select_pair(
                groups, selector, len(runtime.prompt_ids)
            )
            candidate_pair_present = False
            if ex.is_target:
                vals = [group_value(g) for g in groups]
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        if pair_matches(op, vals[i], vals[j], ex.a, ex.b):
                            candidate_pair_present = True
            selected_pair = None
            selector_pair_exact = False
            if selected_left is not None and selected_right is not None:
                left_val = group_value(selected_left)
                right_val = group_value(selected_right)
                selected_pair = [left_val, right_val]
                selector_pair_exact = bool(
                    ex.is_target and pair_matches(op, left_val, right_val, ex.a, ex.b)
                )
            records.append(
                {
                    "example_id": ex.example_id,
                    "target_op": op,
                    "family": ex.family,
                    "is_target": bool(ex.is_target),
                    "a_for_grading_only": ex.a,
                    "b_for_grading_only": ex.b,
                    "n_groups": len(groups),
                    "n_candidate_pairs": n_pairs,
                    "candidate_values": [group_value(g) for g in groups],
                    "candidate_pair_present": bool(candidate_pair_present),
                    "selected_pair": selected_pair,
                    "selected_score": selected_score,
                    "selector_pair_exact": bool(selector_pair_exact),
                    "runtime_provenance": {
                        "prompt_source": "opaque_token_ids",
                        "operand_candidate_source": "activation_l22_chunk_probe",
                        "selector_source": "activation_pair_features",
                    },
                    "diagnostics": diag,
                }
            )
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
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    print(json.dumps({op: payload["ops"][op] for op in args.ops}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
