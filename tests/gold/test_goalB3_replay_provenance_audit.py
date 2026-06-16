from __future__ import annotations

from scripts.goalB3_replay_provenance_audit import audit_bundle, FAIL_VERDICT, FULL_REPLAY_VERDICT, SMOKE_VERDICT


def replay_bundle() -> dict:
    expected = {
        "fired": True,
        "decoded_tuple": {"op": "gcd", "a": 12, "b": 18},
        "answer_source": "python_from_decoded_tuple",
        "provenance": {"op_source": "activation", "operand_source": "activation"},
    }
    return {
        "example_id": "ex",
        "prompt_ids": [1, 2, 3],
        "activations": {"L22": "sha256:abc"},
        "readouts": {"op": "sha256:def"},
        "selectors": {"operand": "sha256:ghi"},
        "thresholds": {"pair": 0.2},
        "runtime_config": {"backend": "llama"},
        "expected": expected,
        "replayed": expected,
    }


def test_replay_audit_accepts_replay_only_bundle() -> None:
    assert audit_bundle(replay_bundle())["verdict"] == "REPLAY_PROVENANCE_PASS"
    assert audit_bundle(replay_bundle())["replayed_present"] is True


def test_replay_audit_fails_forbidden_prompt_text_and_gold_answer() -> None:
    bundle = replay_bundle()
    bundle["prompt_text"] = "What is gcd(12, 18)?"
    bundle["gold_answer"] = 6

    summary = audit_bundle(bundle)

    assert summary["verdict"] == "REPLAY_PROVENANCE_FAIL"
    assert "$.prompt_text" in summary["forbidden_paths"]
    assert "$.gold_answer" in summary["forbidden_paths"]


def test_replay_audit_fails_replay_mismatch() -> None:
    bundle = replay_bundle()
    bundle["replayed"] = {**bundle["expected"], "fired": False}

    summary = audit_bundle(bundle)

    assert summary["verdict"] == "REPLAY_PROVENANCE_FAIL"
    assert "fired" in summary["replay_mismatches"]


def test_replay_audit_marks_expected_only_bundle_as_not_explicit_replay() -> None:
    bundle = replay_bundle()
    del bundle["replayed"]

    summary = audit_bundle(bundle)

    assert summary["verdict"] == "REPLAY_PROVENANCE_PASS"
    assert summary["replayed_present"] is False


def test_replay_audit_exports_full_and_smoke_verdict_names() -> None:
    assert FULL_REPLAY_VERDICT == "REPLAY_PROVENANCE_FULL_PASS"
    assert SMOKE_VERDICT == "REPLAY_SMOKE_ONLY"
    assert FAIL_VERDICT == "REPLAY_PROVENANCE_FAIL"
