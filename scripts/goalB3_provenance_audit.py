#!/usr/bin/env python3
"""Audit Goal B3 JSONL runtime records for the honest-routing contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIRED_FIELDS = {
    "op_source": "activation",
    "operand_source": "activation",
    "answer_source": "python_from_decoded_tuple",
}


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_no}: invalid JSON: {exc}")
                    continue
                rec["_audit_path"] = str(path)
                rec["_audit_line"] = line_no
                records.append(rec)
    return records, errors


def audit_records(records: list[dict[str, Any]], parse_errors: list[str]) -> dict[str, Any]:
    bad_fired: list[dict[str, Any]] = []
    negative_fires: list[dict[str, Any]] = []
    target_nonfires: list[dict[str, Any]] = []
    fired = [rec for rec in records if rec.get("fired")]

    for rec in fired:
        bad_fields = {
            key: rec.get(key)
            for key, expected in REQUIRED_FIRED_FIELDS.items()
            if rec.get(key) != expected
        }
        if bad_fields:
            bad_fired.append(
                {
                    "path": rec.get("_audit_path"),
                    "line": rec.get("_audit_line"),
                    "example_id": rec.get("example_id"),
                    "bad_fields": bad_fields,
                }
            )
        if int(rec.get("is_target", 0)) != 1:
            negative_fires.append(
                {
                    "path": rec.get("_audit_path"),
                    "line": rec.get("_audit_line"),
                    "example_id": rec.get("example_id"),
                    "family": rec.get("family"),
                    "op": rec.get("op"),
                    "target_op": rec.get("target_op"),
                }
            )

    for rec in records:
        if int(rec.get("is_target", 0)) == 1 and not rec.get("fired"):
            target_nonfires.append(
                {
                    "path": rec.get("_audit_path"),
                    "line": rec.get("_audit_line"),
                    "example_id": rec.get("example_id"),
                    "family": rec.get("family"),
                    "op": rec.get("op"),
                    "target_op": rec.get("target_op"),
                }
            )

    verdict = "PROVENANCE_AUDIT_PASS"
    if parse_errors or bad_fired or negative_fires:
        verdict = "PROVENANCE_AUDIT_FAIL"

    return {
        "suite": "goalB3_provenance_audit",
        "verdict": verdict,
        "required_fired_fields": REQUIRED_FIRED_FIELDS,
        "n_records": len(records),
        "n_fired": len(fired),
        "n_target_records": sum(1 for rec in records if int(rec.get("is_target", 0)) == 1),
        "n_bad_fired_provenance": len(bad_fired),
        "n_negative_fires": len(negative_fires),
        "n_target_nonfires": len(target_nonfires),
        "parse_errors": parse_errors[:20],
        "bad_fired_provenance_examples": bad_fired[:20],
        "negative_fire_examples": negative_fires[:20],
        "target_nonfire_examples": target_nonfires[:20],
    }


def write_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal B3 Provenance Audit",
        "",
        f"- verdict: **{summary['verdict']}**",
        f"- records: {summary['n_records']}",
        f"- target records: {summary['n_target_records']}",
        f"- fired records: {summary['n_fired']}",
        f"- bad fired provenance: {summary['n_bad_fired_provenance']}",
        f"- negative fires: {summary['n_negative_fires']}",
        f"- target nonfires: {summary['n_target_nonfires']}",
        "",
        "Required fields on fired records:",
    ]
    for key, value in summary["required_fired_fields"].items():
        lines.append(f"- `{key}` = `{value}`")
    lines.extend(
        [
            "",
            "This audit checks emitted runtime records only. It does not prove that the",
            "upstream implementation avoided prompt parsing; that is covered by the",
            "provenance guard tests and source review.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out_json", type=Path, required=True)
    parser.add_argument("--out_md", type=Path, required=True)
    args = parser.parse_args()

    records, parse_errors = load_records(args.records)
    summary = audit_records(records, parse_errors)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_md(summary, args.out_md)
    print(json.dumps({"n_records": summary["n_records"], "verdict": summary["verdict"]}))


if __name__ == "__main__":
    main()
