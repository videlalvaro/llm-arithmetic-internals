from __future__ import annotations

import json
from pathlib import Path

from scripts.goalB3_benchmark_cross_seed_summary import aggregate


def _write_run(path: Path, seed: int, lift: float, false_fire: float = 0.0) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_locked_total": 3000,
                "n_target_locked_total": 384,
                "ops": {
                    op: {
                        "n_locked": 1000,
                        "n_target_locked": 128,
                        "native_target_exact": 0.1,
                        "readout_routing_target_exact": 0.1 + lift,
                        "exact_score_lift": lift,
                        "hard_negative_false_fire": false_fire,
                        "pair_exact_on_fired_target": 0.95,
                    }
                    for op in ("mul", "div_remainder", "lcm")
                },
            }
        )
    )


def test_goalB3_benchmark_cross_seed_summary_requires_min_seed_count(tmp_path: Path) -> None:
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for idx, path in enumerate(paths):
        _write_run(path, 800 + idx, 0.30)

    payload = aggregate(paths, min_seeds=3, min_lift=0.20, max_false_fire=0.01)

    assert payload["verdict"] == "GOAL_B3_BENCHMARK_CROSS_SEED_INCOMPLETE_OR_FAIL"


def test_goalB3_benchmark_cross_seed_summary_passes_three_stable_runs(tmp_path: Path) -> None:
    paths = [tmp_path / f"{idx}.json" for idx in range(3)]
    for idx, path in enumerate(paths):
        _write_run(path, 800 + idx, 0.25)

    payload = aggregate(paths, min_seeds=3, min_lift=0.20, max_false_fire=0.01)

    assert payload["verdict"] == "GOAL_B3_BENCHMARK_CROSS_SEED_PASS"
    assert payload["ops"]["mul"]["min_lift"] == 0.25
