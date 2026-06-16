from __future__ import annotations

import numpy as np

from scripts import goalB3_op_gate_repair as op_repair
from scripts.goalB2_lcm_benchmark_pipeline import Readouts


class Ex:
    def __init__(self, is_target: int, family: str = "fam"):
        self.is_target = is_target
        self.family = family


def test_goalB3_op_gate_fit_preserves_operand_readouts(monkeypatch) -> None:
    examples = [Ex(1, "pos") for _ in range(4)] + [Ex(0, "neg") for _ in range(4)]
    base = Readouts(
        op_w=np.array([9.0, 9.0], dtype=np.float32),
        op_b=9.0,
        op_threshold=0.9,
        operand_W=np.ones((2, 2), dtype=np.float32),
        operand_b=np.array([1.0, 2.0], dtype=np.float32),
        operand_rmse=3.0,
        pair_conf_threshold=0.4,
    )

    def fake_x(rows, *_args, **_kwargs):
        return np.stack(
            [
                np.array([1.0, 0.0], dtype=np.float32)
                if row.is_target
                else np.array([0.0, 1.0], dtype=np.float32)
                for row in rows
            ]
        )

    monkeypatch.setattr(op_repair.b2, "_X", fake_x)

    repaired, summary = op_repair.fit_op_readout(
        base,
        examples,
        op="mul",
        seed=1,
        backend="synthetic",
        model=None,
        device=None,
        threshold_min=0.5,
        neg_margin=0.01,
        calib_frac=0.25,
    )

    assert repaired.op_threshold >= 0.5
    assert np.array_equal(repaired.operand_W, base.operand_W)
    assert np.array_equal(repaired.operand_b, base.operand_b)
    assert repaired.pair_conf_threshold == base.pair_conf_threshold
    assert summary["calib_neg_fire"] == 0.0
