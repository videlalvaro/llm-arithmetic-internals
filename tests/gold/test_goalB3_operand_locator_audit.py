from __future__ import annotations

from scripts.goalB3_operand_locator_audit import pair_matches, summarize


def test_goalB3_operand_locator_summary_separates_candidate_and_selector() -> None:
    records = [
        {
            "target_op": "mul",
            "family": "clean_symbolic",
            "is_target": True,
            "n_groups": 3,
            "candidate_pair_present": True,
            "selector_pair_exact": False,
            "selected_pair": [2, 9],
        },
        {
            "target_op": "mul",
            "family": "clean_symbolic",
            "is_target": True,
            "n_groups": 2,
            "candidate_pair_present": False,
            "selector_pair_exact": False,
            "selected_pair": [4, 5],
        },
        {
            "target_op": "mul",
            "family": "quoted_expression_negative",
            "is_target": False,
            "n_groups": 1,
            "candidate_pair_present": False,
            "selector_pair_exact": False,
            "selected_pair": None,
        },
    ]

    payload = summarize(records)

    assert payload["mul"]["target_candidate_pair_present"] == 0.5
    assert payload["mul"]["target_selector_pair_exact"] == 0.0
    assert payload["mul"]["families"]["quoted_expression_negative"]["is_target"] is False


def test_goalB3_operand_locator_pair_match_respects_order_for_mod() -> None:
    assert pair_matches("mul", 7, 11, 11, 7)
    assert pair_matches("lcm", 7, 11, 11, 7)
    assert not pair_matches("div_remainder", 7, 11, 11, 7)
    assert pair_matches("div_remainder", 7, 11, 7, 11)
