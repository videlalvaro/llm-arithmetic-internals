from __future__ import annotations

from scripts.goalB3_causal_interchange import OP_CONFIG, parse_args, summarize, verdict


def test_goalB3_causal_interchange_accepts_gcd_and_freezes_threshold() -> None:
    assert "gcd" in OP_CONFIG
    args = parse_args([])
    assert "gcd=0.20" in args.pair_threshold


def test_goalB3_causal_interchange_verdict_requires_controls_to_fail() -> None:
    rows = [
        {
            "target_op": "mul",
            "decoder_followed_donor": True,
            "routed_answer_followed_donor": True,
            "random_decoder_followed_donor": False,
            "random_routed_answer_followed_donor": False,
        }
        for _ in range(10)
    ]
    payload = summarize(rows)

    assert payload["mul"]["decoder_follow_donor_rate"] == 1.0
    assert verdict(payload, min_follow=0.9, max_random_follow=0.1) == "CAUSAL_INTERCHANGE_PASS"

    rows[0]["random_routed_answer_followed_donor"] = True
    payload = summarize(rows)

    assert verdict(payload, min_follow=0.9, max_random_follow=0.05) == "CAUSAL_INTERCHANGE_PARTIAL_OR_FAIL"
