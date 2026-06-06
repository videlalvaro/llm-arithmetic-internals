from __future__ import annotations

from pathlib import Path

from scripts import goalB3_qwen_strict_transfer as b3


class Args:
    backend = "synthetic"
    model_id = "synthetic"
    ops = ["mul", "div_remainder", "lcm", "gcd"]
    seed = 703
    n_per_family = 2
    n_negative_per_family = 4
    train_frac = 0.5
    calib_frac = 0.25
    operand_lo = 0
    operand_hi = 999
    lcm_operand_hi = 80
    op_threshold_min = 0.65
    op_neg_margin = 0.05
    min_exact_gate = 0.20
    min_pair_exact_gate = 0.80
    max_false_fire_gate = 0.01
    probes_in = "unused.pt"
    output_stem = "goalB3_test"
    force_prepare = True
    smoke = True


def test_goalB3_prepare_locks_opaque_token_ids(tmp_path: Path) -> None:
    args = Args()
    args.out_dir = str(tmp_path)

    manifest = b3.phase_prepare(args)
    rows = b3.read_examples(tmp_path / "goalB3_test_splits.jsonl")

    assert manifest["runtime_contract"]["prompt"] == "opaque_token_ids"
    assert manifest["runtime_contract"]["op_source"] == "activation"
    assert manifest["runtime_contract"]["answer_source"] == "python_from_decoded_tuple"
    assert any(ex.split == "locked_test" for ex in rows)
    assert all(isinstance(tok, int) for ex in rows for tok in ex.token_ids)


def test_goalB3_runtime_records_required_provenance(tmp_path: Path) -> None:
    args = Args()
    args.out_dir = str(tmp_path)

    summary = b3.run_full(args)
    records = [
        line
        for line in (tmp_path / "goalB3_test_records.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert summary["backend"] == "synthetic"
    assert summary["verdict"] in {"SMOKE_NO_CLAIM", "NO_CLAIM_SYNTHETIC_BACKEND"}
    assert records
    assert "op_source" in records[0]
    assert "python_from_decoded_tuple" in "\n".join(records)
