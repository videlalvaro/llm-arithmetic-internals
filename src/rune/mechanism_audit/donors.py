"""Strict donor matching + near-operand + control strata for MUST-FIX MF1 + MF3 + MF7.

Extends `scripts/hpj_e160_phase36_recovery_scoring.py:select_same_op_pairs` (lines
121-158) with:
- match on answer-token-count, first-chunk BPE class, decimal first-chunk, prompt-length bucket
- near-operand stratum (A±1,B) / (A,B±1)
- strict_distinct_chunk stratum: distinct *decimal-string* first-chunks across
  recipient / donor / wrong-target so the triad scorer is non-degenerate
- right×right and wrong×wrong control strata
- wrong-target generation: same format, mathematically wrong, decimal-chunk
  controlled, ≥5 away from gold so it is not a plausible carry-neighbor;
  FAIL-CLOSED when no valid candidate exists (caller skips the pair)

Pure Python (numpy + tokenizer); no torch dependency.

v1.2 fix (per ml-intern v1.1 review BLOCKING #1+#2): donor matching now uses the
SAME decimal-string first-chunk definition as `metrics.first_chunk_str` for the
`strict_distinct_chunk` distinctness constraint, so the triad scorer's
first-chunk metrics are truly non-degenerate. Tokenizer first-chunk-id match is
retained as an ADDITIONAL constraint for `strict_match` (format-conservative)
but is no longer the sole chunk-distinctness predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rune.mechanism_audit.metrics import first_chunk_str


Row = dict


VALID_STRATA = (
    "strict_match",            # primary: tight feature match, recipient wrong / donor right
    "strict_distinct_chunk",   # like strict_match BUT donor's first_chunk_token DIFFERS
                                # from recipient's — makes the first-chunk triad non-degenerate
                                # (per ml-intern review 2026-05-26 BLOCKING #2)
    "near_operand",            # operand differs by ±1 / ±10, otherwise strict
    "right_right_control",     # both native-right (donor injects "true answer transport")
    "wrong_wrong_control",     # both native-wrong (donor injects "wrong answer transport")
)


@dataclass
class DonorPair:
    recipient: Row
    donor: Row
    wrong_target: int           # mathematically wrong, format-matched
    stratum: str
    match_distance: float       # tie-break composite (lower = tighter)
    recipient_features: dict
    donor_features: dict


def _log_mag(x: int | None) -> float | None:
    return float(np.log10(abs(int(x)) + 1)) if x is not None else None


def _digit_count(x: int | None) -> int | None:
    return len(str(abs(int(x)))) if x is not None else None


def _first_chunk_token_id(tok, value: int | None) -> int | None:
    """Single-token id for the longest 1-3 digit prefix of value.

    Mirrors the spirit of `hpj_e153.first_chunk_token_ids` but returns the
    longest single-token chunk rather than the union — used as an equality
    key for format matching.
    """
    if value is None:
        return None
    text = str(abs(int(value)))
    for n in (3, 2, 1):
        if len(text) < n:
            continue
        tids = tok.encode(text[:n], add_special_tokens=False)
        if len(tids) == 1:
            return int(tids[0])
    return None


def _answer_token_count(tok, value: int | None) -> int | None:
    if value is None:
        return None
    return len(tok.encode(str(abs(int(value))), add_special_tokens=False))


def _prompt_length_bucket(seq_len: int) -> int:
    if seq_len <= 20:
        return 0
    if seq_len <= 30:
        return 1
    if seq_len <= 50:
        return 2
    if seq_len <= 80:
        return 3
    return 4


def row_features(row: Row, tok) -> dict:
    """Extract all match-relevant features for one corpus row.

    Required keys consumed: op, phrasing (default "symbolic"), band, digit_a,
    digit_b, operand_shape, gold, seq_len, native_correct.

    Two first-chunk definitions are recorded:
      - first_chunk_token_id: single-token BPE id of the longest 1-3 digit prefix
        that tokenizes as one token (used for strict_match's tokenization control)
      - first_chunk_decimal_str: decimal-string chunk per metrics.first_chunk_str
        (used for strict_distinct_chunk's non-degeneracy + matches the scorer)
    """
    gold = row.get("gold")
    return {
        "op": row.get("op"),
        "phrasing": row.get("phrasing", "symbolic"),
        "band": row.get("band"),
        "digit_a": row.get("digit_a"),
        "digit_b": row.get("digit_b"),
        "operand_shape": row.get("operand_shape"),
        "gold_digit_count": _digit_count(gold),
        "gold_log_mag": _log_mag(gold),
        "answer_token_count": _answer_token_count(tok, gold),
        "first_chunk_token_id": _first_chunk_token_id(tok, gold),
        "first_chunk_decimal_str": first_chunk_str(gold) if gold is not None else "",
        "prompt_length_bucket": _prompt_length_bucket(int(row.get("seq_len", 0))),
        "native_correct": bool(row.get("native_correct", False)),
    }


_STRICT_MATCH_KEYS_FULL = (
    "op", "phrasing", "digit_a", "digit_b",
    "gold_digit_count", "answer_token_count",
    "first_chunk_token_id", "first_chunk_decimal_str", "prompt_length_bucket",
)

# strict_distinct_chunk uses these (no first-chunk equality; decimal distinctness
# is enforced separately and is the load-bearing constraint for the scorer)
_STRICT_MATCH_KEYS_NO_CHUNK = tuple(
    k for k in _STRICT_MATCH_KEYS_FULL
    if k not in ("first_chunk_token_id", "first_chunk_decimal_str")
)


def _features_match_strict(rec_f: dict, don_f: dict,
                          require_chunk_match: bool = True) -> tuple[bool, float]:
    """Strict feature match. Returns (passes, distance).

    If require_chunk_match is True (default), donor and recipient must share
    both first_chunk_token_id AND first_chunk_decimal_str (strict_match
    stratum — format-conservative).

    If False, BOTH chunk-equality constraints are dropped (strict_distinct_chunk
    stratum); we instead require BOTH to have a well-defined
    first_chunk_decimal_str AND those strings must DIFFER (this is the
    load-bearing non-degeneracy constraint for the triad scorer, which uses
    decimal-string chunks).

    Distance is the log-magnitude gap (tie-break only).
    """
    keys = _STRICT_MATCH_KEYS_FULL if require_chunk_match else _STRICT_MATCH_KEYS_NO_CHUNK
    for k in keys:
        rv, dv = rec_f.get(k), don_f.get(k)
        if rv is None or dv is None or rv != dv:
            return False, float("inf")
    rec_decimal = rec_f.get("first_chunk_decimal_str")
    don_decimal = don_f.get("first_chunk_decimal_str")
    if not rec_decimal or not don_decimal:
        return False, float("inf")
    if not require_chunk_match:
        # strict_distinct_chunk: enforce DECIMAL-STRING distinctness so the
        # triad scorer can actually distinguish donor / recipient / wrong-target.
        if rec_decimal == don_decimal:
            return False, float("inf")
    return True, abs((rec_f["gold_log_mag"] or 0) - (don_f["gold_log_mag"] or 0))


def _band_bounds(digit_count: int | None) -> tuple[int, int]:
    if digit_count is None or digit_count < 1:
        return 1, 10
    return 10 ** (digit_count - 1), 10 ** digit_count - 1


def _is_near_operand(rec_row: Row, don_row: Row) -> bool:
    """One operand differs by ≤10, the other identical."""
    try:
        ra, rb = int(rec_row["a"]), int(rec_row["b"])
        da, db = int(don_row["a"]), int(don_row["b"])
    except (KeyError, TypeError, ValueError):
        return False
    if ra == da and rb == db:
        return False  # identical, not "near"
    if ra == da and 0 < abs(rb - db) <= 10:
        return True
    if rb == db and 0 < abs(ra - da) <= 10:
        return True
    return False


class DonorMatcher:
    """Builds DonorPairs across 4 strata + wrong-target rows.

    Usage:
        matcher = DonorMatcher(tok, seed=1603)
        pairs = matcher.find_pairs(meta, stratum="strict_match", max_per_op=8)
    """

    def __init__(self, tok, seed: int = 1603):
        self.tok = tok
        self.rng = np.random.default_rng(seed)
        self._feat_cache: dict[int, dict] = {}

    def features(self, row: Row) -> dict:
        rid = row.get("row_id")
        if rid is not None and rid in self._feat_cache:
            return self._feat_cache[rid]
        f = row_features(row, self.tok)
        if rid is not None:
            self._feat_cache[rid] = f
        return f

    def _make_wrong_target(self, gold: int, digit_count: int,
                           exclude: set[int],
                           required_decimal_chunk: str | None = None,
                           excluded_decimal_chunks: set[str] | None = None) -> int | None:
        """Pick a wrong-target int in same digit band, format-controlled.

        - In same digit band as gold
        - Not equal to gold; not in exclude
        - Not within ±5 of gold (avoid carry-neighbor confusion)
        - If required_decimal_chunk is not None: first_chunk_decimal_str must equal it
          (strict_match: required = recipient's decimal chunk)
        - If excluded_decimal_chunks is non-empty: first_chunk_decimal_str must
          NOT be in excluded set (strict_distinct_chunk: exclude both recipient
          and donor decimal chunks so the wrong-target gets a third distinct
          decimal chunk — matches the triad scorer's metric)

        v1.2 fix per ml-intern v1.1 review BLOCKING #2: chunk control now uses
        the SAME decimal-string definition as `metrics.first_chunk_str`. Returns
        None on failure (no silent fallback that violates the distinctness
        guarantee — caller must skip the pair).
        """
        lo, hi = _band_bounds(digit_count)
        excluded_decimal_chunks = excluded_decimal_chunks or set()
        for _ in range(500):
            cand = int(self.rng.integers(lo, hi + 1))
            if cand == gold or cand in exclude:
                continue
            if abs(cand - gold) <= 5:
                continue
            cand_chunk = first_chunk_str(cand)
            if required_decimal_chunk is not None and cand_chunk != required_decimal_chunk:
                continue
            if cand_chunk in excluded_decimal_chunks:
                continue
            return cand
        # Fail-closed: no silent fallback that could violate the distinctness
        # contract. Caller must skip the pair.
        return None

    def find_pairs(self, meta: Sequence[Row], stratum: str,
                   max_per_op: int = 8,
                   ops: Sequence[str] = ("add", "sub", "mul", "mod")) -> list[DonorPair]:
        if stratum not in VALID_STRATA:
            raise ValueError(f"stratum must be in {VALID_STRATA}, got {stratum!r}")

        # Strata define recipient/donor native-correctness requirements
        if stratum in ("strict_match", "strict_distinct_chunk", "near_operand"):
            recip_native_required = False  # recipient must be native-wrong
            donor_native_required = True   # donor must be native-right
        elif stratum == "right_right_control":
            recip_native_required = True
            donor_native_required = True
        else:  # wrong_wrong_control
            recip_native_required = False
            donor_native_required = False

        require_chunk_match = (stratum != "strict_distinct_chunk")

        pairs: list[DonorPair] = []
        used_donor_ids: set[int] = set()

        for op in ops:
            recipients = [r for r in meta
                          if r.get("op") == op
                          and bool(r.get("native_correct", False)) == recip_native_required]
            donors = [r for r in meta
                      if r.get("op") == op
                      and bool(r.get("native_correct", False)) == donor_native_required]

            order = list(range(len(recipients)))
            self.rng.shuffle(order)

            picked_for_op = 0
            for ri in order:
                if picked_for_op >= max_per_op:
                    break
                recip = recipients[ri]
                rec_f = self.features(recip)
                if any(rec_f.get(k) is None for k in _STRICT_MATCH_KEYS_NO_CHUNK):
                    continue

                best: DonorPair | None = None
                for donor in donors:
                    if donor.get("row_id") in used_donor_ids:
                        continue
                    if donor.get("row_id") == recip.get("row_id"):
                        continue
                    if donor.get("a") == recip.get("a") and donor.get("b") == recip.get("b"):
                        continue
                    # Skip donors whose gold matches recipient's — collapses the
                    # triad (donor_follow becomes equivalent to recipient_recovery).
                    if donor.get("gold") == recip.get("gold"):
                        continue
                    don_f = self.features(donor)
                    ok, dist = _features_match_strict(rec_f, don_f,
                                                     require_chunk_match=require_chunk_match)
                    if not ok:
                        continue
                    if stratum == "near_operand" and not _is_near_operand(recip, donor):
                        continue
                    if best is None or dist < best.match_distance:
                        recip_gold = int(recip.get("gold"))
                        donor_gold = int(donor.get("gold"))
                        # Wrong-target chunk control depends on stratum (decimal-string
                        # chunk, matching the triad scorer's metric):
                        # - strict_match: wrong-target's decimal chunk MATCHES recipient
                        #   (format-class controls; triad is intentionally degenerate)
                        # - strict_distinct_chunk: wrong-target's decimal chunk DIFFERS
                        #   from both recipient and donor (3-way distinct chunks →
                        #   non-degenerate triad)
                        # - other strata: don't enforce chunk match (best-effort)
                        if stratum == "strict_match":
                            required_decimal_chunk = rec_f["first_chunk_decimal_str"]
                            excluded_decimal_chunks = None
                        elif stratum == "strict_distinct_chunk":
                            required_decimal_chunk = None
                            excluded_decimal_chunks = {rec_f["first_chunk_decimal_str"],
                                                       don_f["first_chunk_decimal_str"]}
                        else:
                            required_decimal_chunk = None
                            excluded_decimal_chunks = None
                        wrong_t = self._make_wrong_target(
                            recip_gold, rec_f["gold_digit_count"],
                            exclude={recip_gold, donor_gold},
                            required_decimal_chunk=required_decimal_chunk,
                            excluded_decimal_chunks=excluded_decimal_chunks,
                        )
                        # Fail-closed per v1.2 fix: skip pair if no valid
                        # wrong-target could be generated (otherwise triad
                        # distinctness contract violated).
                        if wrong_t is None:
                            continue
                        best = DonorPair(
                            recipient=recip,
                            donor=donor,
                            wrong_target=wrong_t,
                            stratum=stratum,
                            match_distance=dist,
                            recipient_features=rec_f,
                            donor_features=don_f,
                        )
                if best is not None:
                    pairs.append(best)
                    used_donor_ids.add(best.donor.get("row_id"))
                    picked_for_op += 1
        return pairs
