#!/usr/bin/env python3
"""Replay-only provenance audit for Goal B3 runtime bundles.

The replay input is allowed to contain prompt IDs, captured activations,
fitted readout/selector identifiers, thresholds, runtime config, and expected
runtime outputs. It must not contain prompt text, regex results, decoded token
spans, harness operands, CLI operation, or gold answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_KEYS = {
    "prompt",
    "prompt_text",
    "question",
    "source",
    "family",
    "target_op",
    "regex_matches",
    "decoded_token_spans",
    "tokenizer_decoded_operand_spans",
    "harness_operands",
    "gold_operands",
    "gold_answer",
    "gold_answer_for_grading_only",
    "answer",
    "cli_op",
    "cli_operation",
    "source_family_labels",
    "op_labels",
    "native_text",
    "native_pred",
    "native_correct",
    "readout_routing_correct",
}

REQUIRED_RUNTIME_KEYS = {
    "example_id",
    "prompt_ids",
    "activations",
    "readouts",
    "selectors",
    "thresholds",
    "runtime_config",
    "expected",
}

FULL_REPLAY_VERDICT = "REPLAY_PROVENANCE_FULL_PASS"
SMOKE_VERDICT = "REPLAY_SMOKE_ONLY"
FAIL_VERDICT = "REPLAY_PROVENANCE_FAIL"


def find_forbidden(obj: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            child = f"{path}.{key_s}"
            if key_s in FORBIDDEN_KEYS:
                hits.append(child)
            hits.extend(find_forbidden(value, child))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(find_forbidden(value, f"{path}[{idx}]"))
    return hits


def audit_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_RUNTIME_KEYS - set(bundle))
    forbidden = find_forbidden(bundle)
    expected = bundle.get("expected", {})
    replayed_present = "replayed" in bundle
    replayed = bundle.get("replayed", expected)
    comparable = ["fired", "decoded_tuple", "answer_source", "provenance"]
    mismatches = [
        key for key in comparable if expected.get(key) != replayed.get(key)
    ]
    verdict = "REPLAY_PROVENANCE_PASS"
    if missing or forbidden or mismatches:
        verdict = FAIL_VERDICT
    return {
        "verdict": verdict,
        "missing_required_runtime_keys": missing,
        "forbidden_paths": forbidden,
        "replay_mismatches": mismatches,
        "replayed_present": replayed_present,
    }


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("bundles"), list):
        return obj["bundles"]
    return [obj]


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Replay Provenance Audit",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- bundles: {payload['n_bundles']}",
        f"- failed bundles: {payload['n_failed']}",
        f"- bundles with explicit replay output: {payload['n_replayed_present']}",
        f"- full replay required: {payload['require_full']}",
        "",
        "Forbidden runtime fields include prompt text, regex outputs, decoded token spans, "
        "harness operands, CLI op, and gold answers.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundles", nargs="+", type=Path)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--out_md", type=Path, required=True)
    p.add_argument("--require_full", action="store_true")
    p.add_argument("--min_bundles", type=int, default=1)
    args = p.parse_args()
    bundles = [bundle for path in args.bundles for bundle in load_json_or_jsonl(path)]
    results = [audit_bundle(bundle) for bundle in bundles]
    failed = [row for row in results if row["verdict"] != "REPLAY_PROVENANCE_PASS"]
    n_replayed_present = sum(1 for row in results if row["replayed_present"])
    enough_bundles = len(bundles) >= args.min_bundles
    all_explicit_replay = n_replayed_present == len(bundles)
    if failed:
        verdict = FAIL_VERDICT
    elif args.require_full and enough_bundles and all_explicit_replay:
        verdict = FULL_REPLAY_VERDICT
    elif args.require_full:
        verdict = SMOKE_VERDICT
    else:
        verdict = "REPLAY_PROVENANCE_PASS"
    payload = {
        "suite": "goalB3_replay_provenance_audit",
        "verdict": verdict,
        "n_bundles": len(bundles),
        "n_failed": len(failed),
        "n_replayed_present": n_replayed_present,
        "require_full": args.require_full,
        "min_bundles": args.min_bundles,
        "enough_bundles": enough_bundles,
        "all_explicit_replay": all_explicit_replay,
        "results": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(args.out_md, payload)
    print(json.dumps({"verdict": payload["verdict"], "n_failed": payload["n_failed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
