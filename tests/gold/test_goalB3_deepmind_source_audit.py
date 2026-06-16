from __future__ import annotations

from argparse import Namespace

from scripts.goalB3_deepmind_source_audit import audit_file


def test_goalB3_deepmind_source_audit_counts_supported_mul(tmp_path) -> None:
    path = tmp_path / "arithmetic__mul.txt"
    path.write_text(
        "\n".join(
            [
                "What is 12 times 34?",
                "408",
                "Calculate 1.5*4.",
                "6",
                "What is 10000 times 2?",
                "20000",
                "What is 7 times 8?",
                "57",
            ]
        )
        + "\n"
    )
    args = Namespace(scan_limit=100, operand_lo=0, operand_hi=9999, example_limit=2)

    row = audit_file(path, "mul", args)

    assert row["accepted"] == 1
    assert row["locked_40pct_estimate"] == 0
    assert row["reasons"]["decimal_prompt"] == 1
    assert row["reasons"]["operand_range"] == 1
    assert row["reasons"]["answer_mismatch"] == 1
    assert row["examples"][0]["a"] == 12
    assert row["examples"][0]["b"] == 34


def test_goalB3_deepmind_source_audit_lcm_common_denominator(tmp_path) -> None:
    path = tmp_path / "numbers__lcm.txt"
    path.write_text(
        "\n".join(
            [
                "What is the least common multiple of 12 and 18?",
                "36",
                "What is the common denominator of 1 / 8 and 1 / 12?",
                "24",
            ]
        )
        + "\n"
    )
    args = Namespace(scan_limit=100, operand_lo=0, operand_hi=9999, example_limit=2)

    row = audit_file(path, "lcm", args)

    assert row["accepted"] == 2
    assert row["examples"][1]["a"] == 8
    assert row["examples"][1]["b"] == 12
