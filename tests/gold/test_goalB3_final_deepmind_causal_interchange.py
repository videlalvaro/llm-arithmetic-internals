from __future__ import annotations

from scripts.goalB3_final_deepmind_causal_interchange import causal_candidates, candidate_summary


def test_final_deepmind_causal_filters_only_fired_pair_exact_with_diagnostics(tmp_path) -> None:
    good = {
        "example_id": "good",
        "target_op": "gcd",
        "is_target": 1,
        "fired": True,
        "decoded_pair_exact": True,
        "readout_diagnostics": {"selected_groups": [{"positions": [1]}, {"positions": [2]}]},
    }
    rows = [
        good,
        {**good, "example_id": "not_fired", "fired": False},
        {**good, "example_id": "not_exact", "decoded_pair_exact": False},
        {**good, "example_id": "missing_diag", "readout_diagnostics": {}},
        {**good, "example_id": "wrong_op", "target_op": "lcm"},
    ]

    assert [row["example_id"] for row in causal_candidates(rows, "gcd")] == ["good"]


def test_final_deepmind_causal_candidate_summary(tmp_path) -> None:
    path = tmp_path / "seed911_records.jsonl"
    path.write_text(
        '{"target_op":"gcd","is_target":1,"fired":true,"decoded_pair_exact":true,'
        '"readout_diagnostics":{"selected_groups":[{},{}]}}\n'
    )

    summary = candidate_summary([path])

    assert summary["totals"]["gcd"] == 1
    assert summary["totals"]["div_remainder"] == 0
