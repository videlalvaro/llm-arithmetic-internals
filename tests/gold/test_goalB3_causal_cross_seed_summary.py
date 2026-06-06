from __future__ import annotations

import json
from pathlib import Path

from scripts.goalB3_causal_cross_seed_summary import aggregate


def _write_run(path: Path, random_follow: float) -> None:
    path.write_text(
        json.dumps(
            {
                "fit_seed": 1,
                "eval_seed": 2,
                "verdict": "CAUSAL_INTERCHANGE_PASS",
                "ops": {
                    "gcd": {
                        "n_pairs": 20,
                        "decoder_follow_donor_rate": 1.0,
                        "routed_answer_follow_donor_rate": 1.0,
                        "random_decoder_follow_donor_rate": 0.0,
                        "random_routed_answer_follow_donor_rate": random_follow,
                    }
                },
            }
        )
    )


def test_goalB3_causal_cross_seed_summary_allows_frozen_random_gate_boundary(tmp_path: Path) -> None:
    paths = [tmp_path / f"{idx}.json" for idx in range(3)]
    for idx, path in enumerate(paths):
        _write_run(path, 0.10 if idx == 2 else 0.0)

    payload = aggregate(paths)

    assert payload["verdict"] == "CAUSAL_GATE_PASS"
    assert payload["ops"]["gcd"]["verdict"] == "CAUSAL_GATE_PASS"
    assert payload["ops"]["gcd"]["max_random_routed_follow"] == 0.10


def test_goalB3_causal_cross_seed_summary_fails_random_control_above_gate(tmp_path: Path) -> None:
    paths = [tmp_path / f"{idx}.json" for idx in range(3)]
    for idx, path in enumerate(paths):
        _write_run(path, 0.11 if idx == 2 else 0.0)

    payload = aggregate(paths)

    assert payload["verdict"] == "CAUSAL_FAIL"
    assert payload["ops"]["gcd"]["verdict"] == "CAUSAL_FAIL"


def test_goalB3_causal_cross_seed_summary_marks_good_rates_underpowered(tmp_path: Path) -> None:
    paths = [tmp_path / f"{idx}.json" for idx in range(3)]
    for path in paths:
        path.write_text(
            json.dumps(
                {
                    "fit_seed": 1,
                    "eval_seed": 2,
                    "verdict": "CAUSAL_INTERCHANGE_PASS",
                    "ops": {
                        "gcd": {
                            "n_pairs": 6,
                            "decoder_follow_donor_rate": 1.0,
                            "routed_answer_follow_donor_rate": 1.0,
                            "random_decoder_follow_donor_rate": 0.0,
                            "random_routed_answer_follow_donor_rate": 0.0,
                        }
                    },
                }
            )
        )

    payload = aggregate(paths)

    assert payload["verdict"] == "CAUSAL_UNDERPOWERED"
    assert payload["ops"]["gcd"]["verdict"] == "CAUSAL_UNDERPOWERED"
    assert payload["ops"]["gcd"]["total_pairs"] == 18
    assert payload["ops"]["gcd"]["pair_count_gate"] is False
