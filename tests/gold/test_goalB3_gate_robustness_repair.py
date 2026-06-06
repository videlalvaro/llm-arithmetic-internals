from __future__ import annotations

from scripts.goalB3_gate_robustness_repair import summarize, verdict


def test_goalB3_gate_repair_summary_tracks_distractor_and_negative_fire() -> None:
    records = [
        {
            "target_op": "mul",
            "family": "pre_distractor",
            "is_target": 1,
            "fired": True,
            "readout_routing_correct": True,
            "decoded_pair_exact": True,
            "readout_diagnostics": {},
        },
        {
            "target_op": "mul",
            "family": "wrong_op_negative",
            "is_target": 0,
            "fired": False,
            "readout_routing_correct": False,
            "decoded_pair_exact": False,
            "readout_diagnostics": {"abstain_reason": "safe_gate_below_threshold"},
        },
    ]

    payload = {"ops": summarize(records)}

    assert payload["ops"]["mul"]["families"]["pre_distractor"]["fire_rate"] == 1.0
    assert payload["ops"]["mul"]["negative_fire_rate"] == 0.0
    assert verdict(payload, max_false_fire=0.01, min_distractor_fire=0.8) == "GATE_REPAIR_DISTRACTOR_PASS"


def test_goalB3_gate_repair_verdict_fails_on_false_fire() -> None:
    records = [
        {
            "target_op": "mul",
            "family": "pre_distractor",
            "is_target": 1,
            "fired": True,
            "readout_routing_correct": True,
            "decoded_pair_exact": True,
            "readout_diagnostics": {},
        },
        {
            "target_op": "mul",
            "family": "wrong_op_negative",
            "is_target": 0,
            "fired": True,
            "readout_routing_correct": False,
            "decoded_pair_exact": False,
            "readout_diagnostics": {},
        },
    ]

    payload = {"ops": summarize(records)}

    assert verdict(payload, max_false_fire=0.01, min_distractor_fire=0.8) == "GATE_REPAIR_UNSAFE_FALSE_FIRE"
