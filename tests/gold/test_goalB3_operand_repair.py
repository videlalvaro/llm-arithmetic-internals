from __future__ import annotations

from scripts.goalB3_operand_repair import parse_pair_thresholds


def test_goalB3_operand_repair_parses_pair_thresholds() -> None:
    parsed = parse_pair_thresholds(["mul=0.05", "div_remainder=0.2", "lcm=0.15"])

    assert parsed == {"mul": 0.05, "div_remainder": 0.2, "lcm": 0.15}
