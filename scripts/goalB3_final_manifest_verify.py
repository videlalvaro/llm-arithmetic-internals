#!/usr/bin/env python3
"""Verify the frozen Goal B3 final manifest against current artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import goalB3_causal_interchange as causal
import goalB3_repaired_benchmark_suite as benchmark


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def benchmark_defaults() -> dict[str, Any]:
    args = benchmark.parse_args([])
    return {
        "pair_threshold": args.pair_threshold,
        "op_threshold_min": args.op_threshold_min,
        "op_neg_margin": args.op_neg_margin,
        "safe_threshold_min": args.safe_threshold_min,
        "safe_neg_margin": args.safe_neg_margin,
        "chunk_top_k": args.chunk_top_k,
        "chunk_window": args.chunk_window,
        "chunk_pos_threshold": args.chunk_pos_threshold,
        "chunk_value_margin_threshold": args.chunk_value_margin_threshold,
        "operand_lo": args.operand_lo,
        "operand_hi": args.operand_hi,
        "max_new_tokens": args.max_new_tokens,
    }


def causal_defaults() -> dict[str, Any]:
    args = causal.parse_args([])
    return {
        "pair_threshold": args.pair_threshold,
        "min_follow": args.min_follow,
        "max_random_follow": args.max_random_follow,
        "n_pairs_per_op": args.n_pairs_per_op,
    }


def pair_thresholds(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        out[key] = float(value)
    return out


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    package_path = Path(manifest["mandatory_artifacts"]["package_audit"])
    package = read_json(package_path)
    b_defaults = benchmark_defaults()
    c_defaults = causal_defaults()
    errors: list[str] = []

    frozen = manifest["frozen_thresholds"]
    for key in [
        "op_threshold_min",
        "op_neg_margin",
        "safe_threshold_min",
        "safe_neg_margin",
        "chunk_top_k",
        "chunk_window",
        "chunk_pos_threshold",
        "chunk_value_margin_threshold",
        "operand_lo",
        "operand_hi",
        "max_new_tokens",
    ]:
        if b_defaults[key] != frozen[key]:
            errors.append(f"benchmark default {key}={b_defaults[key]} != manifest {frozen[key]}")

    if pair_thresholds(b_defaults["pair_threshold"]) != frozen["pair_conf_thresholds"]:
        errors.append("benchmark pair thresholds differ from manifest")
    if pair_thresholds(c_defaults["pair_threshold"]) != frozen["pair_conf_thresholds"]:
        errors.append("causal pair thresholds differ from manifest")

    gates = manifest["acceptance_gates"]
    if c_defaults["max_random_follow"] != gates["max_random_routed_follow"]:
        errors.append("causal max_random_follow differs from manifest gate")
    if c_defaults["min_follow"] != gates["min_causal_follow"]:
        errors.append("causal min_follow differs from manifest gate")
    if c_defaults["n_pairs_per_op"] != manifest["causal_interchange"]["n_pairs_per_op"]:
        errors.append("causal n_pairs_per_op differs from manifest")
    if package.get("verdict") != "NEURIPS_PACKAGE_MINIMUM_BAR_PASS_WITH_CAVEATS":
        errors.append("package audit verdict is not the expected caveated pass")
    if not all(package.get("gates", {}).values()):
        errors.append("one or more package audit gates are false")

    for name, rel in manifest["mandatory_artifacts"].items():
        if not Path(rel).exists():
            errors.append(f"mandatory artifact missing: {name}: {rel}")

    required_sources = set(manifest["mandatory_artifacts"].values())
    package_sources = set(package.get("source_paths", {}).values())
    for rel in required_sources:
        if rel.endswith(".json") and rel not in package_sources and rel != str(package_path):
            if rel not in {
                manifest["mandatory_artifacts"]["deepmind_source_audit"],
                manifest["mandatory_artifacts"]["package_audit"],
            }:
                errors.append(f"package audit does not reference mandatory source: {rel}")

    verdict = "FINAL_MANIFEST_VERIFY_PASS" if not errors else "FINAL_MANIFEST_VERIFY_FAIL"
    return {
        "suite": "goalB3_final_manifest_verify",
        "manifest": str(manifest_path),
        "package_audit": str(package_path),
        "verdict": verdict,
        "errors": errors,
        "benchmark_defaults": b_defaults,
        "causal_defaults": c_defaults,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("docs/goalB3_final_frozen_manifest.json"))
    parser.add_argument("--out_json", type=Path, default=Path("docs/goalB3_final_manifest_verify.json"))
    parser.add_argument("--out_md", type=Path, default=Path("docs/research/643_goalB3_final_manifest_verify_2026-06-02.md"))
    args = parser.parse_args()
    payload = verify(args.manifest)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Goal B3 Final Manifest Verification",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- manifest: `{payload['manifest']}`",
        f"- package audit: `{payload['package_audit']}`",
        "",
    ]
    if payload["errors"]:
        lines.append("## Errors")
        lines.append("")
        for err in payload["errors"]:
            lines.append(f"- {err}")
    else:
        lines.append("No manifest/default/artifact inconsistencies found.")
    lines.append("")
    args.out_md.write_text("\n".join(lines))
    print(json.dumps({"verdict": payload["verdict"], "n_errors": len(payload["errors"])}))


if __name__ == "__main__":
    main()
