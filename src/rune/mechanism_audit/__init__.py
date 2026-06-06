"""MUST-FIX mechanism-audit infrastructure.

Modules:
    donors: strict donor matching, near-operand + control strata, wrong-target generation.
    metrics: 7-metric panel + triad scorer (donor / recipient-gold / wrong-target follow).
    stratify: StratifiedTable that refuses pooled writes; bootstrap-CI summaries; verdict.

Built per docs/research/435_mustfix_implementation_plan_2026-05-26.md, satisfying the
7 MUST-FIX concerns from docs/research/ml_intern_post_phase4b_frontier.md.
"""
from rune.mechanism_audit.donors import (
    DonorMatcher,
    DonorPair,
    VALID_STRATA,
    row_features,
)
from rune.mechanism_audit.metrics import (
    TriadScore,
    digits_only,
    exact_parsed_numeric_eq,
    first_chunk_token_eq,
    first_chunk_token_ids,
    full_vocab_kl,
    digit_restricted_logit_margin,
    non_answer_format_kl,
    parse_first_int,
    score_triad,
    token_rank_and_logit,
)
from rune.mechanism_audit.stratify import StratifiedTable, Verdict

__all__ = [
    "DonorMatcher",
    "DonorPair",
    "StratifiedTable",
    "TriadScore",
    "VALID_STRATA",
    "Verdict",
    "digit_restricted_logit_margin",
    "digits_only",
    "exact_parsed_numeric_eq",
    "first_chunk_token_eq",
    "first_chunk_token_ids",
    "full_vocab_kl",
    "non_answer_format_kl",
    "parse_first_int",
    "row_features",
    "score_triad",
    "token_rank_and_logit",
]
