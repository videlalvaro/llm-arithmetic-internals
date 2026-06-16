from __future__ import annotations

from argparse import Namespace

import numpy as np

from scripts import goalB2_lcm_benchmark_pipeline as b2
from scripts.goalB3_repaired_benchmark_suite import replay_bundle, verdict


def test_goalB3_gcd_shared_helpers_are_first_class_target() -> None:
    assert b2.compute_target("gcd", 84, 30) == 6

    args = Namespace(
        seed=123,
        n_per_family=4,
        n_adversarial_per_family=4,
        train_frac=0.40,
        calib_frac=0.20,
        operand_lo=0,
        operand_hi=999,
        require_multitoken_answers=False,
    )
    examples = b2.build_frozen_gcd_examples(args, "synthetic")
    locked = [ex for ex in examples if ex.split == "locked_test"]
    target = [ex for ex in locked if ex.is_target]
    negative = [ex for ex in locked if not ex.is_target]

    assert target
    assert negative
    assert all(ex.op == "gcd" for ex in target)
    assert all(ex.answer == b2.compute_target("gcd", ex.a, ex.b) for ex in target)


def test_goalB3_repaired_benchmark_verdict_requires_scale_and_all_gates() -> None:
    args = Namespace(
        min_locked=1000,
        min_target_per_op=50,
        min_ops=3,
        min_lift=0.20,
        max_false_fire=0.01,
        min_pair_exact_fired=0.80,
    )
    payload = {
        "n_locked_total": 1200,
        "ops": {
            "mul": {
                "n_target_locked": 128,
                "exact_score_lift": 0.30,
                "hard_negative_false_fire": 0.0,
                "pair_exact_on_fired_target": 0.90,
            },
            "div_remainder": {
                "n_target_locked": 128,
                "exact_score_lift": 0.25,
                "hard_negative_false_fire": 0.0,
                "pair_exact_on_fired_target": 0.85,
            },
            "lcm": {
                "n_target_locked": 128,
                "exact_score_lift": 0.40,
                "hard_negative_false_fire": 0.005,
                "pair_exact_on_fired_target": 0.95,
            },
        },
    }

    assert verdict(payload, args) == "GOAL_B3_BENCHMARK_LIFT_PASS"

    payload["ops"]["lcm"]["hard_negative_false_fire"] = 0.02
    assert verdict(payload, args) == "GOAL_B3_BENCHMARK_LIFT_FAIL"

    payload["ops"]["lcm"]["hard_negative_false_fire"] = 0.0
    payload["n_locked_total"] = 999
    assert verdict(payload, args) == "BENCHMARK_UNDERPOWERED"

    payload["n_locked_total"] = 1200
    payload["ops"]["mul"]["n_target_locked"] = 27
    assert verdict(payload, args) == "BENCHMARK_TARGET_COVERAGE_FAIL"


def test_goalB3_replay_bundle_uses_runtime_only_schema() -> None:
    runtime = b2.RuntimeInputs(
        example_id="ex",
        prompt_ids=(1, 2, 3),
        activations={"answer_site_L12_L15": np.zeros(4, dtype=np.float32)},
    )
    readouts = b2.Readouts(
        op_w=np.ones(4, dtype=np.float32),
        op_b=0.0,
        op_threshold=0.65,
        operand_W=np.zeros((2, 4), dtype=np.float32),
        operand_b=np.zeros(2, dtype=np.float32),
        operand_rmse=1.0,
        pair_conf_threshold=0.2,
    )
    pipe = {
        "fired": True,
        "decoded": {
            "op": "gcd",
            "a": 12,
            "b": 18,
            "op_source": "activation",
            "operand_source": "activation",
        },
        "provenance": {
            "op_source": "activation",
            "operand_source": "activation",
            "answer_source": "python_from_decoded_tuple",
        },
    }

    bundle = replay_bundle(
        runtime=runtime,
        expected_pipe=pipe,
        replayed_pipe=pipe,
        backend="llama",
        operand_decode_mode="attention_j16_l22_chunk",
        readouts=readouts,
        safe_gate=None,
        selector=None,
        pair_threshold=0.2,
    )

    assert bundle["prompt_ids"] == [1, 2, 3]
    assert "prompt" not in bundle
    assert "gold_answer" not in bundle
    assert bundle["expected"] == bundle["replayed"]
