from __future__ import annotations

from scripts.goalB3_qwen_operand_diagnostics import (
    build_qwen_examples,
    build_synthetic_examples,
    evaluate_sites,
    stable_split_hash,
    synthetic_capture,
    verdict,
)


def test_qwen_diagnostic_split_hash_is_stable() -> None:
    rows_a = build_synthetic_examples(seed=17, n=20)
    rows_b = build_synthetic_examples(seed=17, n=20)

    assert stable_split_hash(rows_a) == stable_split_hash(rows_b)


def test_qwen_synthetic_backend_finds_high_signal_answer_site() -> None:
    rows = build_synthetic_examples(seed=19, n=80)
    feats = synthetic_capture(rows, layers=[8, 10], positions=["answer_site", "input_a"], seed=19)

    metrics = evaluate_sites(rows, feats)

    assert max(row["ordered_pair_exact"] for row in metrics) >= 0.80
    assert verdict(metrics) == "QWEN_OPERAND_ROUTE_FOUND"


def test_qwen_example_builder_records_operand_token_indices() -> None:
    class ToyTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = True):
            ids = [1] if add_special_tokens else []
            ids.extend(range(2, 2 + len(text.split())))
            return type("Tok", (), {"input_ids": ids})()

    rows = build_qwen_examples(seed=23, n=4, tokenizer=ToyTokenizer())

    assert all(row.a_token_index is not None for row in rows)
    assert all(row.b_token_index is not None for row in rows)
    assert all(row.token_ids for row in rows)
