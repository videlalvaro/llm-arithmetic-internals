from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.goalB2_lcm_benchmark_pipeline import (
    FORBIDDEN_RUNTIME_FIELDS,
    DecodedTuple,
    J16ChunkProbe,
    ProvenanceError,
    ProvenanceGuard,
    Readouts,
    RuntimeInputs,
    decode_operands_j16_chunks,
    phase_eval,
    phase_fit,
    phase_prepare,
    read_examples,
    run_opaque_pipeline,
    runtime_from_example,
)


def _args(tmp_path: Path):
    class A:
        phase = "full"
        backend = "synthetic"
        target_op = "lcm"
        dataset_source = "synthetic"
        out_dir = str(tmp_path)
        seed = 617
        n_per_family = 6
        n_natural = 6
        n_adversarial_per_family = 6
        train_frac = 0.50
        calib_frac = 0.25
        dm_scan_limit = 100
        dm_dir = ""
        require_multitoken_answers = False
        include_common_denominator = False
        smoke = True
        max_new_tokens = 8
        probes_in = "docs/p1_1_internal_value_probes.pt"
        operand_decode_mode = "attention_fourier_l15"
        chunk_probe_in = "docs/j16_multitoken_operand_probe.pt"
        chunk_top_k = 12
        chunk_window = 1
        chunk_pos_threshold = 0.5
        chunk_value_margin_threshold = 0.0
        pair_conf_threshold = 0.20
        op_threshold_min = 0.5
        op_threshold_neg_margin = 1e-4
        operand_lo = 0
        operand_hi = 999

    return A()


@pytest.mark.parametrize("field", FORBIDDEN_RUNTIME_FIELDS)
def test_goalB2_provenance_guard_rejects_forbidden_runtime_fields(field: str) -> None:
    guard = ProvenanceGuard(runtime_mode=True)
    with pytest.raises(ProvenanceError):
        guard.reject_forbidden(**{field: "leak"})


def test_goalB2_calculator_only_accepts_activation_decoded_lcm() -> None:
    guard = ProvenanceGuard(runtime_mode=True)
    ok = DecodedTuple(op="lcm", a=84, b=30, op_score=0.99, pair_confidence=1.0)
    assert guard.calculator(ok) == 420

    bad_source = DecodedTuple(
        op="lcm",
        a=84,
        b=30,
        op_score=0.99,
        pair_confidence=1.0,
        operand_source="harness",
    )
    with pytest.raises(ProvenanceError):
        guard.calculator(bad_source)

    bad_op = DecodedTuple(op="gcd", a=84, b=30, op_score=0.99, pair_confidence=1.0)
    with pytest.raises(ProvenanceError):
        guard.calculator(bad_op)


def test_goalB2_calculator_can_target_div_remainder() -> None:
    guard = ProvenanceGuard(runtime_mode=True, allowed_op="div_remainder")
    ok = DecodedTuple(
        op="div_remainder",
        a=84,
        b=30,
        op_score=0.99,
        pair_confidence=1.0,
    )
    assert guard.calculator(ok) == 24


def test_goalB2_calculator_can_target_mul() -> None:
    guard = ProvenanceGuard(runtime_mode=True, allowed_op="mul")
    ok = DecodedTuple(op="mul", a=84, b=30, op_score=0.99, pair_confidence=1.0)
    assert guard.calculator(ok) == 2520


def test_goalB2_synthetic_smoke_outputs_required_provenance(tmp_path: Path) -> None:
    args = _args(tmp_path)
    manifest = phase_prepare(args)
    fit = phase_fit(args)
    summary = phase_eval(args)

    assert manifest["locked_test_sha256"]
    assert Path(fit["readout_path"]).exists()
    assert summary["verdict"] == "SMOKE_NO_CLAIM"
    assert "run_explanation" in summary

    records = [
        json.loads(line)
        for line in Path(summary["records_path"]).read_text().splitlines()
        if line.strip()
    ]
    assert records
    fired = [r for r in records if r["fired"]]
    assert fired, "synthetic smoke should exercise the fired provenance path"
    for rec in fired:
        assert rec["op_source"] == "activation"
        assert rec["operand_source"] == "activation"
        assert rec["answer_source"] == "python_from_decoded_tuple"


def test_goalB2_lcm_chunk_frozen_smoke_is_supported(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.dataset_source = "lcm_chunk_frozen"
    args.target_op = "lcm"
    args.n_per_family = 4
    args.n_adversarial_per_family = 4

    manifest = phase_prepare(args)
    fit = phase_fit(args)
    summary = phase_eval(args)

    assert manifest["dataset_source"] == "lcm_chunk_frozen"
    assert manifest["target_op"] == "lcm"
    assert manifest["n_target_locked"] > 0
    assert Path(fit["readout_path"]).exists()
    assert summary["verdict"] == "SMOKE_NO_CLAIM"
    assert summary["target_op"] == "lcm"


def test_goalB2_opaque_pipeline_rejects_prompt_text_leak(tmp_path: Path) -> None:
    args = _args(tmp_path)
    phase_prepare(args)
    phase_fit(args)
    examples = read_examples(tmp_path / "goalB2_lcm_benchmark_splits.jsonl")
    readouts = Readouts.load(tmp_path / "goalB2_lcm_benchmark_readouts.npz")
    runtime = runtime_from_example(examples[0], args.seed)

    with pytest.raises(ProvenanceError):
        run_opaque_pipeline(
            runtime,
            readouts,
            ProvenanceGuard(runtime_mode=True),
            injected_forbidden={"prompt_text": examples[0].prompt},
        )


def test_goalB2_chunk_decoder_assembles_operands_from_activations_only() -> None:
    classes = list(range(10))
    eye = np.eye(10, dtype="float32")
    H = np.zeros((6, 10), dtype="float32")
    H[1, 1] = 10.0
    H[2, 2] = 10.0
    H[4, 3] = 10.0
    attention = np.array(
        [-float("inf"), 0.9, 0.8, 0.01, 0.7, -float("inf")],
        dtype="float32",
    )
    probe = J16ChunkProbe(
        layer=22,
        position_w=np.zeros(10, dtype="float32"),
        position_b=10.0,
        value_w=eye,
        value_b=np.zeros(10, dtype="float32"),
        value_classes=np.array(classes, dtype="int64"),
    )
    runtime = RuntimeInputs(
        example_id="toy",
        prompt_ids=(1, 2, 3, 4, 5, 6),
        activations={"all_positions_L22": H, "answer_attention_scores": attention},
    )

    decoded, diagnostics = decode_operands_j16_chunks(
        runtime,
        probe,
        operand_lo=0,
        operand_hi=999,
        chunk_top_k=3,
        chunk_window=0,
        chunk_pos_threshold=0.5,
        chunk_value_margin_threshold=0.0,
    )

    assert decoded is not None
    assert decoded[0] == 12
    assert decoded[1] == 3
    assert decoded[2] > 0.99
    assert diagnostics["operand_decode_mode"] == "attention_j16_l22_chunk"
