#!/usr/bin/env python3
"""Preregister a recognized-source Goal B3 mul attempt before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "kind": "missing"}
    if path.is_file():
        return {"available": True, "path": str(path), "kind": "file", "sha256": sha256_file(path)}
    return {"available": True, "path": str(path), "kind": "directory", "git_commit": git_commit(path)}


def verify_prereg(payload: dict[str, Any]) -> str:
    if not payload["source"]["available"]:
        return "MUL_SOURCE_UNAVAILABLE"
    gates = payload["acceptance_gates"]
    frozen = payload["frozen_before_eval"]
    required = [
        frozen.get("source"),
        frozen.get("seeds"),
        frozen.get("filters"),
        frozen.get("thresholds"),
        gates.get("min_locked_examples", 0) >= 1000,
        gates.get("min_locked_mul_targets", 0) >= 300,
        gates.get("min_target_per_seed", 0) >= 50,
        gates.get("min_lift", 0) >= 0.20,
        gates.get("max_false_fire", 1) <= 0.01,
        gates.get("min_pair_exact_fired", 0) >= 0.80,
        payload["runtime_contract"].get("forbid_text_parsing") is True,
    ]
    return "MUL_PREREGISTRATION_PASS" if all(required) else "MUL_PREREGISTRATION_INCOMPLETE"


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Recognized-Source Mul Preregistration",
        "",
        f"- verdict: **{payload['verdict']}**",
        f"- source: `{payload['source']['path']}`",
        f"- source available: {payload['source']['available']}",
        f"- seeds: `{payload['seeds']}`",
        "",
        "Frozen filters: two integer operands in `[0, 9999]`, integer product, "
        "and no token/text parsing at runtime.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path.home() / "deepmind_math" / "mathematics_dataset-v1.0")
    p.add_argument("--seeds", nargs="+", type=int, default=[1009, 1019, 1029])
    p.add_argument("--out_json", type=Path, default=DOCS / "goalB3_mul_source_preregistration.json")
    p.add_argument("--out_md", type=Path, default=DOCS / "goalB3_mul_source_preregistration.md")
    args = p.parse_args()
    payload = {
        "suite": "goalB3_mul_source_preregistration",
        "source": resolve_source(args.source),
        "seeds": args.seeds,
        "filters": {
            "op": "mul",
            "operand_lo": 0,
            "operand_hi": 9999,
            "answer_type": "integer_product",
            "normalization": "strip whitespace; exact base-10 integer string",
        },
        "thresholds": {
            "pair_conf_threshold": 0.05,
            "op_threshold_min": 0.65,
            "safe_threshold_min": 0.65,
        },
        "acceptance_gates": {
            "min_locked_examples": 1000,
            "min_locked_mul_targets": 300,
            "min_target_per_seed": 50,
            "min_lift": 0.20,
            "max_false_fire": 0.01,
            "min_pair_exact_fired": 0.80,
        },
        "runtime_contract": {
            "op_source": "activation",
            "operand_source": "activation",
            "answer_source": "python_from_decoded_tuple",
            "forbid_text_parsing": True,
        },
        "frozen_before_eval": {
            "source": True,
            "seeds": True,
            "filters": True,
            "thresholds": True,
        },
    }
    payload["verdict"] = verify_prereg(payload)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(args.out_md, payload)
    print(json.dumps({"verdict": payload["verdict"], "source_available": payload["source"]["available"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
