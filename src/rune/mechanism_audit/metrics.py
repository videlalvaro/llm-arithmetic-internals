"""7-metric panel + triad scorer for MUST-FIX MF3 + MF6.

Per ml-intern (docs/research/ml_intern_post_phase4b_frontier.md, MUST-FIX 6):

    Primary metric is canonical numeric equality.

This module provides:

  - exact_parsed_numeric_eq:        canonical integer equality from generated text
  - first_chunk_token_eq:           token-level equality on first 1-3 digit tokens
  - digit_restricted_logit_margin:  margin restricted to digit-only vocab subset
  - full_vocab_kl:                  KL on full softmax
  - non_answer_format_kl:           KL excluding answer-format slots
  - first_chunk_token_ids:          mirrored from hpj_e153 (single-token chunks)
  - token_rank_and_logit:           mirrored from hpj_e153

Triad:

  - score_triad(generated, recipient_gold, donor_gold, wrong_target) returns all
    three follows simultaneously so we can distinguish:
        recipient_recovery > donor_follow   → mechanism transport (PASS)
        donor_follow > recipient_recovery   → answer-state transport only
        wrong_target_follow ≈ donor_follow  → format-only effect, not arithmetic
        all three low                       → generic damage / no transport
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


# ----------------------------- parsing helpers ----------------------------- #


_DIGIT_RUN = re.compile(r"-?\d+")


def digits_only(s: str) -> str:
    """Return only the digit characters of s (drops sign + non-digits)."""
    return "".join(c for c in s if c.isdigit())


def parse_first_int(text: str) -> int | None:
    """Parse the first signed integer span in text. Returns None if none found."""
    if not text:
        return None
    m = _DIGIT_RUN.search(text)
    if m is None:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


# ---------------------------- text-based metrics --------------------------- #


def exact_parsed_numeric_eq(generated_text: str, target: int | None) -> bool:
    """Primary MF6 metric: parse first integer span, equality on canonical int.

    Tolerates surrounding non-digit characters (newlines, prefixes, etc.). NOT a
    substring check — `"199 = "` parsed against target `99` returns False.
    """
    if target is None:
        return False
    got = parse_first_int(generated_text)
    if got is None:
        return False
    return int(got) == int(target)


def first_chunk_str(n: int | None) -> str:
    """First 1-3 digit chunk of |n| (length depends on len(str(|n|)) % 3)."""
    if n is None:
        return ""
    s = str(int(abs(n)))
    rem = len(s) % 3
    return s[:rem] if rem else s[:3]


def first_chunk_token_ids(tok, value: int | None) -> list[int]:
    """Single-token candidates for the first 1-3 digit chunk of value.

    Mirrors hpj_e153.first_chunk_token_ids; duplicated here to avoid scripts
    importing from scripts.
    """
    if value is None:
        return []
    text = str(abs(int(value)))
    out: list[int] = []
    for n in (1, 2, 3):
        if len(text) < n:
            continue
        tids = tok.encode(text[:n], add_special_tokens=False)
        if len(tids) == 1:
            tid = int(tids[0])
            if tid not in out:
                out.append(tid)
    return out


def first_chunk_token_eq(generated_token_ids: list[int], target: int | None, tok) -> bool:
    """True iff the first generated token matches any single-token first-chunk
    candidate for target."""
    if target is None or not generated_token_ids:
        return False
    cands = set(first_chunk_token_ids(tok, target))
    return int(generated_token_ids[0]) in cands


# --------------------------- logit-based metrics --------------------------- #


def token_rank_and_logit(logits, token_ids: list[int]) -> tuple[int | None, float | None]:
    """Best (lowest) rank and highest raw logit across token candidates.

    Mirrors hpj_e153.token_rank_and_logit.
    """
    import torch

    if not token_ids:
        return None, None
    scores = logits.float()
    order = torch.argsort(scores, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device)
    best_rank = min(int(ranks[int(tid)].detach().cpu()) for tid in token_ids)
    best_logit = max(float(scores[int(tid)].detach().cpu()) for tid in token_ids)
    return best_rank, best_logit


def _digit_only_token_ids(tok) -> list[int]:
    """All vocab tokens whose decoded string is composed entirely of digits.

    Cached per tokenizer instance via a private attribute on tok.
    """
    cache_attr = "_rune_digit_token_ids"
    cached = getattr(tok, cache_attr, None)
    if cached is not None:
        return cached
    vocab_size = getattr(tok, "vocab_size", None) or len(tok.get_vocab())
    out: list[int] = []
    for tid in range(vocab_size):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        stripped = s.strip()
        if stripped and stripped.lstrip("-").isdigit():
            out.append(tid)
    try:
        setattr(tok, cache_attr, out)
    except Exception:
        pass
    return out


def digit_restricted_logit_margin(
    logits,
    tok,
    target: int | None,
    donor: int | None = None,
    wrong_target: int | None = None,
) -> dict:
    """Margin of target's first-chunk token over the best non-target digit token.

    Returns a dict with:
      - target_best_logit / target_best_rank
      - donor_best_logit / donor_best_rank
      - wrong_target_best_logit / wrong_target_best_rank
      - margin_target_vs_best_other_digit
    """
    import torch

    digit_ids = _digit_only_token_ids(tok)
    digit_ids_t = torch.tensor(digit_ids, dtype=torch.long, device=logits.device)
    digit_logits = logits.float().index_select(0, digit_ids_t)

    target_cands = set(first_chunk_token_ids(tok, target))
    donor_cands = set(first_chunk_token_ids(tok, donor))
    wrong_cands = set(first_chunk_token_ids(tok, wrong_target))

    target_rank, target_logit = token_rank_and_logit(logits, list(target_cands))
    donor_rank, donor_logit = token_rank_and_logit(logits, list(donor_cands))
    wrong_rank, wrong_logit = token_rank_and_logit(logits, list(wrong_cands))

    best_other = None
    if target_cands:
        mask = torch.tensor(
            [tid not in target_cands for tid in digit_ids],
            dtype=torch.bool, device=logits.device,
        )
        non_target = digit_logits[mask]
        if non_target.numel() > 0:
            best_other = float(non_target.max().detach().cpu())

    margin = None
    if target_logit is not None and best_other is not None:
        margin = float(target_logit - best_other)

    return {
        "target_best_logit": target_logit,
        "target_best_rank": target_rank,
        "donor_best_logit": donor_logit,
        "donor_best_rank": donor_rank,
        "wrong_target_best_logit": wrong_logit,
        "wrong_target_best_rank": wrong_rank,
        "best_other_digit_logit": best_other,
        "margin_target_vs_best_other_digit": margin,
    }


def full_vocab_kl(logits_a, logits_b) -> float:
    """KL(P_a || P_b) on full softmax distributions over vocab."""
    import torch
    import torch.nn.functional as F

    la = logits_a.float()
    lb = logits_b.float()
    log_pa = F.log_softmax(la, dim=-1)
    log_pb = F.log_softmax(lb, dim=-1)
    pa = log_pa.exp()
    return float((pa * (log_pa - log_pb)).sum().detach().cpu())


def non_answer_format_kl(
    logits_a,
    logits_b,
    answer_format_token_ids: list[int],
) -> float:
    """KL on tokens EXCLUDING answer-format slots. Measures whether a patch
    preserves non-answer behavior. answer_format_token_ids should be the
    union of plausible first-chunk tokens for the recipient / donor / wrong-target
    answers (or any other "answer slot" tokens you want excluded).
    """
    import torch
    import torch.nn.functional as F

    la = logits_a.float().clone()
    lb = logits_b.float().clone()
    if answer_format_token_ids:
        idx = torch.tensor(list(set(answer_format_token_ids)), dtype=torch.long,
                           device=la.device)
        la.index_fill_(0, idx, -float("inf"))
        lb.index_fill_(0, idx, -float("inf"))
    log_pa = F.log_softmax(la, dim=-1)
    log_pb = F.log_softmax(lb, dim=-1)
    pa = log_pa.exp()
    # any -inf rows produce 0 * (-inf - -inf) → 0 by torch convention
    return float(torch.nan_to_num(pa * (log_pa - log_pb), nan=0.0, posinf=0.0,
                                  neginf=0.0).sum().detach().cpu())


# -------------------------------- triad scorer -------------------------------- #


@dataclass
class TriadScore:
    """The three simultaneous follows for one (patched recipient, donor, wrong-target) row.

    Per MUST-FIX 3 — must be computed *together* so we can read the pattern, not
    promote one and ignore the others.

    Two metric layers:
      - Exact full-integer equality (primary MF6 — strict, used for negative-paper claims)
      - First-chunk equality (secondary — what L29 residual patches actually transport
        per docs 414 / 433c; useful when strict-match stratum forces first-chunk
        agreement between recipient and donor and exact-numeric becomes structurally
        impossible)
    """
    recipient_recovery: bool          # parsed-int == recipient_gold (exact)
    donor_follow: bool                # parsed-int == donor_gold (exact)
    wrong_target_follow: bool         # parsed-int == wrong_target (exact)
    first_chunk_recipient_recovery: bool
    first_chunk_donor_follow: bool
    first_chunk_wrong_target_follow: bool
    parsed_value: int | None
    recipient_gold: int | None
    donor_gold: int | None
    wrong_target: int | None

    def label(self) -> str:
        """One-word verdict for the row (exact-numeric primary)."""
        if self.recipient_recovery and not self.donor_follow:
            return "MECHANISM"
        if self.donor_follow and not self.recipient_recovery:
            return "ANSWER_TRANSPORT"
        if self.wrong_target_follow:
            return "WRONG_TARGET"
        if self.parsed_value is None:
            return "NO_PARSE"
        if self.first_chunk_donor_follow and not self.first_chunk_recipient_recovery:
            return "FIRST_CHUNK_ANSWER"
        if self.first_chunk_recipient_recovery and not self.first_chunk_donor_follow:
            return "FIRST_CHUNK_MECHANISM"
        return "OTHER"


def score_triad(
    generated_text: str,
    recipient_gold: int | None,
    donor_gold: int | None,
    wrong_target: int | None,
) -> TriadScore:
    """Score a generated text against the (recipient, donor, wrong-target) triad.

    Uses canonical first-integer parsing per MUST-FIX 6. Also computes a
    softer first-chunk equality (the leading 1-3 digit group, matching the
    BPE chunk boundary that prior Rune scoring used in docs 110/414).
    """
    parsed = parse_first_int(generated_text)
    rec_eq = (parsed is not None and recipient_gold is not None
              and int(parsed) == int(recipient_gold))
    don_eq = (parsed is not None and donor_gold is not None
              and int(parsed) == int(donor_gold))
    wrong_eq = (parsed is not None and wrong_target is not None
                and int(parsed) == int(wrong_target))

    parsed_chunk = first_chunk_str(parsed) if parsed is not None else ""
    rec_fc = (parsed_chunk != "" and recipient_gold is not None
              and parsed_chunk == first_chunk_str(recipient_gold))
    don_fc = (parsed_chunk != "" and donor_gold is not None
              and parsed_chunk == first_chunk_str(donor_gold))
    wrong_fc = (parsed_chunk != "" and wrong_target is not None
                and parsed_chunk == first_chunk_str(wrong_target))

    return TriadScore(
        recipient_recovery=bool(rec_eq),
        donor_follow=bool(don_eq),
        wrong_target_follow=bool(wrong_eq),
        first_chunk_recipient_recovery=bool(rec_fc),
        first_chunk_donor_follow=bool(don_fc),
        first_chunk_wrong_target_follow=bool(wrong_fc),
        parsed_value=parsed,
        recipient_gold=recipient_gold,
        donor_gold=donor_gold,
        wrong_target=wrong_target,
    )
