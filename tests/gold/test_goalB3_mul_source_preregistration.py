from __future__ import annotations

from scripts.goalB3_mul_source_preregistration import resolve_source, verify_prereg


def test_mul_preregistration_freezes_source_and_gates(tmp_path) -> None:
    source = tmp_path / "arithmetic__mul.txt"
    source.write_text("What is 2 times 3?\n6\n")
    payload = {
        "source": resolve_source(source),
        "acceptance_gates": {
            "min_locked_examples": 1000,
            "min_locked_mul_targets": 300,
            "min_target_per_seed": 50,
            "min_lift": 0.20,
            "max_false_fire": 0.01,
            "min_pair_exact_fired": 0.80,
        },
        "runtime_contract": {"forbid_text_parsing": True},
        "frozen_before_eval": {
            "source": True,
            "seeds": True,
            "filters": True,
            "thresholds": True,
        },
    }

    assert payload["source"]["sha256"]
    assert verify_prereg(payload) == "MUL_PREREGISTRATION_PASS"


def test_mul_preregistration_fails_missing_source(tmp_path) -> None:
    payload = {
        "source": resolve_source(tmp_path / "missing"),
        "acceptance_gates": {},
        "runtime_contract": {"forbid_text_parsing": True},
        "frozen_before_eval": {},
    }

    assert verify_prereg(payload) == "MUL_SOURCE_UNAVAILABLE"
