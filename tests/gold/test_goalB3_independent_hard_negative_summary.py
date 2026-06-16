from __future__ import annotations

import json
from pathlib import Path

from scripts.goalB3_independent_hard_negative_summary import summarize


def test_independent_hard_negative_summary_requires_all_categories(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [
        {"family": "adv_quoted_mul_negative", "is_target": False, "fired": False},
        {"family": "adv_do_not_mul_negative", "is_target": False, "fired": False},
        {"family": "adv_lcm_surface_negative", "is_target": False, "fired": False},
        {"family": "adv_table_negative", "is_target": False, "fired": False},
        {"family": "adv_code_negative", "is_target": False, "fired": False},
        {"family": "adv_natural_chunk_negative", "is_target": False, "fired": False},
        {"family": "adv_decimal_negative", "is_target": False, "fired": False},
        {"family": "adv_negative_operand_negative", "is_target": False, "fired": False},
        {"family": "clean_symbolic", "is_target": True, "fired": True},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    payload = summarize([path])

    assert payload["verdict"] == "INDEPENDENT_HARD_NEGATIVE_PASS"
    assert payload["n_negative"] == 8
    assert payload["missing_categories"] == []


def test_independent_hard_negative_summary_fails_rate_gate(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [{"family": "adv_quoted_mul_negative", "is_target": False, "fired": True}]
    path.write_text(json.dumps(rows[0]) + "\n")

    payload = summarize([path])

    assert payload["verdict"] == "INDEPENDENT_HARD_NEGATIVE_INCOMPLETE_OR_FAIL"
    assert payload["categories"]["quoted_arithmetic"]["false_fire_rate"] == 1.0
