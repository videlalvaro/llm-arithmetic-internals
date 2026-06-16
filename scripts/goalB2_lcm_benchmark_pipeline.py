#!/usr/bin/env python3
"""Goal B2 honest multi-token LCM benchmark pipeline.

Phases:

  prepare -> lock train/calibration/test metadata
  fit     -> activation-only LCM gate/readouts
  eval    -> opaque runtime path + exact-answer benchmark scoring

The default synthetic backend is for engineering tests only and always reports
NO_CLAIM_SYNTHETIC_BACKEND. A claim-bearing run must use --backend llama and a
locked DeepMind interpolate split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

from p1_1_internal_value_decoder import load_probe_bank  # noqa: E402

MODEL_ID = "unsloth/Meta-Llama-3.1-8B"
DTYPE = torch.bfloat16
DEFAULT_SEED = 617
ACT_DIM = 32
LLAMA_FEATURE_LAYERS = (12, 15)
SAFE_GATE_LAYER = 5
ATTN_OPERAND_LAYERS = (22, 24, 26, 28)
FOURIER_OPERAND_LAYER = 15
J16_CHUNK_LAYER = 22
ANSWER_SUFFIX = " Answer: "
EXCLUDE_TRAILING_ATTENTION_POSITIONS = 3
DEFAULT_TARGET_OP = "lcm"
DEFAULT_OPERAND_DECODE_MODE = "attention_fourier_l15"
FORBIDDEN_RUNTIME_FIELDS = (
    "prompt_text",
    "regex_matches",
    "tokenizer_decoded_spans",
    "cli_op",
    "harness_operands",
    "gold_answer",
    "gold_label",
)


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    family: str
    op: str
    is_lcm: int
    is_target: int
    a: int
    b: int
    answer: int | None
    prompt: str
    token_ids: list[int]
    prompt_tokens: int
    operand_band: str
    answer_band: str
    answer_tokens: int
    source: str


@dataclass(frozen=True)
class RuntimeInputs:
    example_id: str
    prompt_ids: tuple[int, ...]
    activations: dict[str, np.ndarray]


@dataclass(frozen=True)
class DecodedTuple:
    op: str
    a: int
    b: int
    op_score: float
    pair_confidence: float
    op_source: str = "activation"
    operand_source: str = "activation"


class ProvenanceError(RuntimeError):
    pass


class ProvenanceGuard:
    def __init__(self, runtime_mode: bool = True, allowed_op: str = DEFAULT_TARGET_OP):
        self.runtime_mode = runtime_mode
        self.allowed_op = allowed_op

    def reject_forbidden(self, **values: Any) -> None:
        if not self.runtime_mode:
            return
        present = [name for name, value in values.items() if value is not None]
        forbidden = [name for name in present if name in FORBIDDEN_RUNTIME_FIELDS]
        if forbidden:
            raise ProvenanceError(
                "Forbidden runtime provenance field(s): " + ", ".join(sorted(forbidden))
            )

    def assert_runtime_inputs(self, runtime: RuntimeInputs) -> None:
        if not isinstance(runtime.prompt_ids, tuple):
            raise ProvenanceError("runtime prompt IDs must be an opaque tuple")
        if not isinstance(runtime.activations, dict):
            raise ProvenanceError("runtime activations must be a dict of arrays")
        for forbidden in FORBIDDEN_RUNTIME_FIELDS:
            if hasattr(runtime, forbidden):
                raise ProvenanceError(f"RuntimeInputs exposes forbidden field: {forbidden}")

    def assert_decoded_tuple(self, decoded: DecodedTuple) -> None:
        if decoded.op_source != "activation" or decoded.operand_source != "activation":
            raise ProvenanceError("decoded op/operands must be activation-sourced")
        if decoded.op != self.allowed_op:
            raise ProvenanceError(f"calculator is only allowed for decoded {self.allowed_op}")

    def calculator(self, decoded: DecodedTuple) -> int:
        self.assert_decoded_tuple(decoded)
        if decoded.op == "gcd":
            return math.gcd(int(decoded.a), int(decoded.b))
        if decoded.op == "lcm":
            return safe_lcm(int(decoded.a), int(decoded.b))
        if decoded.op == "div_remainder":
            return int(decoded.a) % int(decoded.b)
        if decoded.op == "mul":
            return int(decoded.a) * int(decoded.b)
        raise ProvenanceError(f"unsupported decoded op: {decoded.op}")


@dataclass
class Readouts:
    op_w: np.ndarray
    op_b: float
    op_threshold: float
    operand_W: np.ndarray
    operand_b: np.ndarray
    operand_rmse: float
    pair_conf_threshold: float

    def save(self, path: Path) -> None:
        np.savez(
            path,
            op_w=self.op_w,
            op_b=np.array([self.op_b], dtype=np.float32),
            op_threshold=np.array([self.op_threshold], dtype=np.float32),
            operand_W=self.operand_W,
            operand_b=self.operand_b,
            operand_rmse=np.array([self.operand_rmse], dtype=np.float32),
            pair_conf_threshold=np.array([self.pair_conf_threshold], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> Readouts:
        z = np.load(path)
        return cls(
            op_w=z["op_w"].astype(np.float32),
            op_b=float(z["op_b"][0]),
            op_threshold=float(z["op_threshold"][0]),
            operand_W=z["operand_W"].astype(np.float32),
            operand_b=z["operand_b"].astype(np.float32),
            operand_rmse=float(z["operand_rmse"][0]),
            pair_conf_threshold=float(z["pair_conf_threshold"][0]),
        )


@dataclass
class SafeGateReadout:
    w: np.ndarray
    b: float
    threshold: float
    mode: str = "l5_mean"

    def save(self, path: Path) -> None:
        np.savez(
            path,
            w=self.w.astype(np.float32),
            b=np.array([self.b], dtype=np.float32),
            threshold=np.array([self.threshold], dtype=np.float32),
            mode=np.array([self.mode]),
        )

    @classmethod
    def load(cls, path: Path) -> SafeGateReadout:
        z = np.load(path)
        mode_arr = z.get("mode")
        return cls(
            w=z["w"].astype(np.float32),
            b=float(z["b"][0]),
            threshold=float(z["threshold"][0]),
            mode=str(mode_arr[0]) if mode_arr is not None else "l5_mean",
        )


@dataclass
class ChunkGroupSelector:
    w: np.ndarray
    b: float
    threshold: float = 0.0

    def save(self, path: Path) -> None:
        np.savez(
            path,
            w=self.w.astype(np.float32),
            b=np.array([self.b], dtype=np.float32),
            threshold=np.array([self.threshold], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> ChunkGroupSelector:
        z = np.load(path)
        return cls(
            w=z["w"].astype(np.float32),
            b=float(z["b"][0]),
            threshold=float(z["threshold"][0]),
        )

    def score(self, features: np.ndarray) -> float:
        return float(sigmoid(float(features.astype(np.float32) @ self.w + self.b)))


@dataclass
class ChunkPairSelector:
    w: np.ndarray
    b: float
    threshold: float = 0.0

    def save(self, path: Path) -> None:
        np.savez(
            path,
            w=self.w.astype(np.float32),
            b=np.array([self.b], dtype=np.float32),
            threshold=np.array([self.threshold], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> ChunkPairSelector:
        z = np.load(path)
        return cls(
            w=z["w"].astype(np.float32),
            b=float(z["b"][0]),
            threshold=float(z["threshold"][0]),
        )

    def score(self, features: np.ndarray) -> float:
        return float(sigmoid(float(features.astype(np.float32) @ self.w + self.b)))


@dataclass
class J16ChunkProbe:
    """Activation-only multi-token operand chunk probe from J16.

    The checkpoint contains a binary position probe and a 1000-way per-token
    value probe. Runtime uses only residual activations and attention scores;
    token strings in the checkpoint are historical metadata and are not used.
    """

    layer: int
    position_w: np.ndarray
    position_b: float
    value_w: np.ndarray
    value_b: np.ndarray
    value_classes: np.ndarray

    @classmethod
    def load(cls, path: Path) -> J16ChunkProbe:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        pos = payload["position_probe"]
        val = payload["value_probe"]
        return cls(
            layer=int(payload["L_probe"]),
            position_w=np.asarray(pos["W"], dtype=np.float32).reshape(-1),
            position_b=float(np.asarray(pos["b"], dtype=np.float32).reshape(-1)[0]),
            value_w=np.asarray(val["W"], dtype=np.float32),
            value_b=np.asarray(val["b"], dtype=np.float32),
            value_classes=np.asarray(val["classes"], dtype=np.int64),
        )

    def position_probs(self, H: np.ndarray) -> np.ndarray:
        logits = H.astype(np.float32) @ self.position_w + self.position_b
        return sigmoid(logits).astype(np.float32)

    def decode_value(self, h: np.ndarray) -> tuple[int, float, float]:
        logits = self.value_w @ h.astype(np.float32) + self.value_b
        if logits.shape[0] == 1:
            return int(self.value_classes[0]), 1.0, float("inf")
        top2 = np.argpartition(logits, -2)[-2:]
        top2 = top2[np.argsort(logits[top2])[::-1]]
        best, second = int(top2[0]), int(top2[1])
        margin = float(logits[best] - logits[second])
        shifted = logits - float(np.max(logits))
        probs = np.exp(shifted)
        probs /= float(np.sum(probs))
        return int(self.value_classes[best]), float(probs[best]), margin


def safe_lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def compute_target(op: str, a: int, b: int) -> int:
    if op == "gcd":
        return math.gcd(a, b)
    if op == "lcm":
        return safe_lcm(a, b)
    if op == "div_remainder":
        if b == 0:
            raise ValueError("div_remainder requires nonzero divisor")
        return a % b
    if op == "mul":
        return a * b
    raise ValueError(op)


def target_family(op: str) -> str:
    if op == "gcd":
        return "gcd_deepmind"
    if op == "lcm":
        return "lcm_deepmind"
    if op == "div_remainder":
        return "div_remainder_deepmind"
    if op == "mul":
        return "mul_deepmind"
    raise ValueError(op)


def stem_for_op(op: str) -> str:
    if op == "gcd":
        return "goalB2_gcd_benchmark"
    if op == "lcm":
        return "goalB2_lcm_benchmark"
    if op == "div_remainder":
        return "goalB2_div_remainder_benchmark"
    if op == "mul":
        return "goalB2_mul_benchmark"
    raise ValueError(op)


def stem_for_args(args: argparse.Namespace) -> str:
    if getattr(args, "dataset_source", "") == "lcm_chunk_frozen":
        return "goalB2_lcm_chunk_frozen"
    if getattr(args, "dataset_source", "") == "mul_chunk_frozen":
        return "goalB2_mul_chunk_frozen"
    if getattr(args, "dataset_source", "") == "div_remainder_frozen":
        return "goalB2_div_remainder_frozen"
    return stem_for_op(args.target_op)


def prereg_for_op(op: str) -> str:
    if op == "gcd":
        return "docs/research/615_goalB_gcd_prereg_2026-06-01.md"
    if op == "lcm":
        return "docs/research/617_goalB2_lcm_prereg_2026-06-01.md"
    if op == "div_remainder":
        return "docs/research/621_goalB2_div_remainder_prereg_2026-06-01.md"
    if op == "mul":
        return "docs/research/623_goalB2_mul_prereg_2026-06-01.md"
    raise ValueError(op)


def band(x: int | None) -> str:
    if x is None:
        return "na"
    if abs(x) < 10:
        return "1d"
    if abs(x) < 100:
        return "2d"
    if abs(x) < 1000:
        return "3d"
    if abs(x) < 10000:
        return "4d"
    return "5d_plus"


def stable_token_ids(prompt: str) -> list[int]:
    toks = re.findall(r"\S+", prompt)
    return [int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % 32000 for t in toks]


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode_prompt(prompt: str, backend: str, tok: Any | None = None) -> list[int]:
    if backend == "llama":
        if tok is None:
            tok = load_tokenizer()
        return [int(x) for x in tok.encode(prompt, add_special_tokens=True)]
    return stable_token_ids(prompt)


def answer_token_count(answer: int | None, backend: str, tok: Any | None = None) -> int:
    if answer is None:
        return 0
    if backend == "llama":
        if tok is None:
            tok = load_tokenizer()
        return len(tok.encode(str(int(answer)), add_special_tokens=False))
    return max(1, len(str(abs(int(answer)))) // 3 + 1)


def load_llama() -> tuple[Any, Any, torch.device]:
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device.type != "cpu":
        model.to(device)
    return model, tok, device


def dm_interpolate_dir(args: argparse.Namespace) -> Path:
    if args.dm_dir:
        return Path(args.dm_dir)
    return Path(
        os.environ.get(
            "DEEPMIND_MATH_INTERPOLATE_DIR",
            str(Path.home() / "deepmind_math" / "mathematics_dataset-v1.0" / "interpolate"),
        )
    )


def read_dm_pairs(path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    pairs = []
    for i in range(0, len(lines) - 1, 2):
        pairs.append((lines[i], lines[i + 1]))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


_LCM_COMMON_DENOM = re.compile(
    r"common denominator of\s*(-?\d+)\s*/\s*(-?\d+)\s+and\s*(-?\d+)\s*/\s*(-?\d+)",
    re.I,
)


def extract_lcm_operands(prompt: str) -> tuple[int, int] | None:
    m = _LCM_COMMON_DENOM.search(prompt)
    if m is not None:
        return int(m.group(2)), int(m.group(4))
    nums = re.findall(r"-?\d+", prompt)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def extract_two_ints(prompt: str) -> tuple[int, int] | None:
    nums = re.findall(r"-?\d+", prompt)
    if len(nums) < 2:
        return None
    return int(nums[0]), int(nums[1])


def render_synthetic(family: str, a: int, b: int) -> tuple[str, str, int | None]:
    if family == "lcm_deepmind_style":
        return (
            f"What is the least common multiple of {a} and {b}?{ANSWER_SUFFIX}",
            "lcm",
            safe_lcm(a, b),
        )
    if family == "lcm_semantic":
        return (
            f"What is the smallest positive integer divisible by both {a} and {b}?{ANSWER_SUFFIX}",
            "lcm",
            safe_lcm(a, b),
        )
    if family == "gcd_hard_negative":
        return (
            f"What is the greatest common divisor of {a} and {b}?{ANSWER_SUFFIX}",
            "gcd",
            math.gcd(a, b),
        )
    if family == "mul_hard_negative":
        return f"What is {a} times {b}?{ANSWER_SUFFIX}", "mul", a * b
    if family == "mod_hard_negative":
        b = max(2, b)
        return f"What is the remainder when {a} is divided by {b}?{ANSWER_SUFFIX}", "mod", a % b
    if family == "natural_number_control":
        return f"The report lists batch {a} and section {b} before continuing: ", "natural", None
    raise ValueError(family)


MUL_TARGET_TEMPLATES = {
    "mul_symbolic_target": [
        "What is {a} * {b}?",
        "Calculate {a} * {b}.",
    ],
    "mul_times_target": [
        "What is {a} times {b}?",
        "Calculate {a} times {b}.",
    ],
    "mul_product_target": [
        "Find the product of {a} and {b}.",
        "Compute the product of {a} and {b}.",
    ],
    "mul_story_target": [
        "A warehouse has {a} boxes with {b} parts in each box. How many parts are there?",
        "There are {a} rows with {b} seats in each row. How many seats are there?",
    ],
}


MUL_ADVERSARIAL_NEGATIVE_TEMPLATES = {
    "adv_quoted_mul_negative": [
        "The note says \"{a} * {b}\" but asks only for a summary. Answer with the summary:",
        "Quote the expression \"{a} times {b}\" without solving it. Answer:",
    ],
    "adv_do_not_mul_negative": [
        "Do not multiply {a} and {b}; say which number is larger. Answer:",
        "Without computing the product of {a} and {b}, identify the smaller number. Answer:",
    ],
    "adv_table_negative": [
        "Table row: item={a}, count={b}, product column blank. Continue the table label:",
        "Invoice line has SKU {a} and batch {b}. Do not compute totals. Category:",
    ],
    "adv_code_negative": [
        "In code, x = {a} * {b}; do not evaluate it. What operator appears?",
        "The snippet return {a} * {b} is shown as text. Name the symbol used:",
    ],
    "adv_lcm_surface_negative": [
        "Find the least common multiple of {a} and {b}; product is only a distractor. Answer:",
        "What is the common multiple requested for {a} and {b}, not their product? Answer:",
    ],
    "adv_gcd_surface_negative": [
        "Find the greatest common divisor of {a} and {b}; do not multiply. Answer:",
        "What is the highest common factor of {a} and {b}? The word product is irrelevant. Answer:",
    ],
    "adv_mod_surface_negative": [
        "What is the remainder when {a} is divided by {b}? Do not compute the product. Answer:",
        "Compute {a} mod {b}; ignore any multiplication distractor. Answer:",
    ],
    "adv_decimal_negative": [
        "What is {a}.5 times {b}.25? Unsupported decimal calculation. Answer:",
        "The decimal expression {a}.1 * {b}.2 appears in a log. Classify it:",
    ],
    "adv_negative_operand_negative": [
        "What is -{a} times {b}? Unsupported signed multiplication. Answer:",
        "The expression {a} * -{b} is outside the supported route. Classify it:",
    ],
    "adv_natural_chunk_negative": [
        "Archive page {a} references section {b} before the paragraph continues. Next word:",
        "The report lists station {a}, route {b}, and no arithmetic request. Continue:",
    ],
}


DIV_TARGET_TEMPLATES = {
    "div_mod_symbolic_target": [
        "What is {a} mod {b}?",
        "Calculate {a} % {b}.",
    ],
    "div_remainder_target": [
        "What is the remainder when {a} is divided by {b}?",
        "Find the remainder of {a} divided by {b}.",
    ],
    "div_story_target": [
        "{a} cards are dealt into piles of {b}. How many cards are left over?",
        "{a} bolts are packed into boxes of {b}. What is the remainder?",
    ],
    "div_mod_word_target": [
        "Compute {a} modulo {b}.",
        "Evaluate {a} mod {b}.",
    ],
}


DIV_ADVERSARIAL_NEGATIVE_TEMPLATES = {
    "adv_quoted_mod_negative": [
        "The note says \"{a} mod {b}\" but asks only for a summary. Answer:",
        "Quote the expression \"{a} % {b}\" without solving it. Answer:",
    ],
    "adv_do_not_mod_negative": [
        "Do not compute {a} modulo {b}; say which number is larger. Answer:",
        "Without finding the remainder of {a} divided by {b}, identify the smaller number. Answer:",
    ],
    "adv_mul_surface_negative": [
        "What is {a} times {b}? The word remainder is a distractor. Answer:",
        "Find the product of {a} and {b}; do not compute a remainder. Answer:",
    ],
    "adv_gcd_mod_surface_negative": [
        "Find the gcd of {a} and {b}; ignore modulo notation in the notes. Answer:",
        "What is the greatest common divisor of {a} and {b}? Answer:",
    ],
    "adv_lcm_mod_surface_negative": [
        "Find the least common multiple of {a} and {b}; do not compute modulo. Answer:",
        "What is the lowest common multiple of {a} and {b}? Answer:",
    ],
    "adv_table_mod_negative": [
        "Table row: dividend={a}, divisor={b}, remainder column blank. Continue label:",
        "Log entry has code {a} and shard {b}; no arithmetic request. Category:",
    ],
    "adv_code_mod_negative": [
        "In code, x = {a} % {b}; do not evaluate it. What operator appears?",
        "The snippet return {a} % {b} is shown as text. Name the symbol used:",
    ],
    "adv_decimal_mod_negative": [
        "What is {a}.5 mod {b}.25? Unsupported decimal modulo. Answer:",
        "The decimal expression {a}.1 % {b}.2 appears in a log. Classify it:",
    ],
    "adv_negative_mod_negative": [
        "What is -{a} mod {b}? Unsupported signed modulo. Answer:",
        "The expression {a} % -{b} is outside the supported route. Classify it:",
    ],
    "adv_natural_mod_chunk_negative": [
        "Archive page {a} references divisor field {b}, then prose continues. Next word:",
        "The report lists register {a}, bucket {b}, and no arithmetic request. Continue:",
    ],
}


LCM_TARGET_TEMPLATES = {
    "lcm_symbolic_target": [
        "What is lcm({a}, {b})?",
        "Calculate lcm({a}, {b}).",
    ],
    "lcm_phrase_target": [
        "What is the least common multiple of {a} and {b}?",
        "Find the lowest common multiple of {a} and {b}.",
    ],
    "lcm_semantic_target": [
        "What is the smallest positive integer divisible by both {a} and {b}?",
        "Find the smallest whole number that is a multiple of both {a} and {b}.",
    ],
    "lcm_denominator_target": [
        "What common denominator should be used for fractions with denominators {a} and {b}?",
        "Find a least common denominator for denominators {a} and {b}.",
    ],
}


LCM_ADVERSARIAL_NEGATIVE_TEMPLATES = {
    "adv_quoted_lcm_negative": [
        "The note says \"lcm({a}, {b})\" but asks only for a summary. Answer:",
        "Quote the phrase \"least common multiple of {a} and {b}\" without solving it. Answer:",
    ],
    "adv_do_not_lcm_negative": [
        "Do not find the lcm of {a} and {b}; say which number is larger. Answer:",
        "Without computing the least common multiple of {a} and {b}, identify the smaller number. Answer:",
    ],
    "adv_gcd_surface_lcm_negative": [
        "Find the greatest common divisor of {a} and {b}; do not compute the lcm. Answer:",
        "What is the highest common factor of {a} and {b}? Answer:",
    ],
    "adv_mul_surface_lcm_negative": [
        "What is {a} times {b}? The phrase common multiple is a distractor. Answer:",
        "Find the product of {a} and {b}; do not compute a least common multiple. Answer:",
    ],
    "adv_mod_surface_lcm_negative": [
        "What is {a} mod {b}? Do not compute the lcm. Answer:",
        "Find the remainder when {a} is divided by {b}; ignore lcm notes. Answer:",
    ],
    "adv_table_lcm_negative": [
        "Table row: denominator={a}, code={b}, lcm column blank. Continue label:",
        "Archive page {a} references section {b}; no arithmetic request follows. Next word:",
    ],
    "adv_code_lcm_negative": [
        "In code, y = lcm({a}, {b}); do not evaluate it. What function name appears?",
        "The snippet call lcm({a}, {b}) is shown as text. Classify it:",
    ],
}


GCD_TARGET_TEMPLATES = {
    "gcd_symbolic_target": [
        "What is gcd({a}, {b})?",
        "Calculate gcd({a}, {b}).",
    ],
    "gcd_phrase_target": [
        "What is the greatest common divisor of {a} and {b}?",
        "Find the highest common factor of {a} and {b}.",
    ],
    "gcd_semantic_target": [
        "What is the largest positive integer that divides both {a} and {b}?",
        "Find the biggest whole number that is a factor of both {a} and {b}.",
    ],
    "gcd_story_target": [
        "{a} tiles and {b} tiles must be split into equal groups with no leftovers. What is the largest group size?",
        "Two bundles contain {a} and {b} items. What largest equal packet size divides both bundles?",
    ],
}


GCD_ADVERSARIAL_NEGATIVE_TEMPLATES = {
    "adv_quoted_gcd_negative": [
        "The note says \"gcd({a}, {b})\" but asks only for a summary. Answer:",
        "Quote the phrase \"greatest common divisor of {a} and {b}\" without solving it. Answer:",
    ],
    "adv_do_not_gcd_negative": [
        "Do not find the gcd of {a} and {b}; say which number is larger. Answer:",
        "Without computing the greatest common divisor of {a} and {b}, identify the smaller number. Answer:",
    ],
    "adv_lcm_surface_gcd_negative": [
        "Find the least common multiple of {a} and {b}; do not compute the gcd. Answer:",
        "What is the lowest common multiple of {a} and {b}? Answer:",
    ],
    "adv_mul_surface_gcd_negative": [
        "What is {a} times {b}? The phrase common divisor is a distractor. Answer:",
        "Find the product of {a} and {b}; do not compute a greatest common divisor. Answer:",
    ],
    "adv_mod_surface_gcd_negative": [
        "What is {a} mod {b}? Do not compute the gcd. Answer:",
        "Find the remainder when {a} is divided by {b}; ignore gcd notes. Answer:",
    ],
    "adv_table_gcd_negative": [
        "Table row: factor={a}, code={b}, gcd column blank. Continue label:",
        "Archive page {a} references section {b}; no arithmetic request follows. Next word:",
    ],
    "adv_code_gcd_negative": [
        "In code, y = gcd({a}, {b}); do not evaluate it. What function name appears?",
        "The snippet call gcd({a}, {b}) is shown as text. Classify it:",
    ],
}


SAFE_GATE_POSITIVE_TEMPLATES = {
    "mul": [
        "What is {a} * {b}?",
        "What is {a} times {b}?",
        "Find the product of {a} and {b}.",
        "A shelf has {a} bins with {b} screws each. How many screws are there?",
        "In a report, section {d} asks: what is the product of {a} and {b}?",
    ],
    "div_remainder": [
        "What is {a} mod {b}?",
        "What is the remainder when {a} is divided by {b}?",
        "Find the remainder of {a} divided by {b}.",
        "{a} bolts are packed into boxes of {b}. What is the remainder?",
    ],
    "lcm": [
        "What is the least common multiple of {a} and {b}?",
        "Find the lowest common multiple of {a} and {b}.",
        "What is the smallest positive integer divisible by both {a} and {b}?",
    ],
    "gcd": [
        "What is the greatest common divisor of {a} and {b}?",
        "Find the highest common factor of {a} and {b}.",
        "What is the largest positive integer that divides both {a} and {b}?",
    ],
}


SAFE_GATE_NEGATIVE_TEMPLATES = {
    "mul": [
        "The text says \"{a} * {b}\". Do not solve it; classify it.",
        "Quote the expression \"{a} times {b}\" without solving it.",
        "Do not multiply {a} and {b}; say which number is larger.",
        "In code, x = {a} * {b}; do not evaluate it. What operator appears?",
        "Table row: item={a}, count={b}, product column blank. Continue the label.",
        "The receipt number is {a} and batch number is {b}. Do not compute totals.",
        "Find the gcd of {a} and {b}; do not multiply.",
        "What is {a} mod {b}? The word product is irrelevant.",
    ],
    "div_remainder": [
        "The text says \"{a} mod {b}\". Do not solve it; classify it.",
        "Quote the expression \"{a} % {b}\" without solving it.",
        "Do not compute {a} modulo {b}; say which number is larger.",
        "In code, x = {a} % {b}; do not evaluate it. What operator appears?",
        "Table row: dividend={a}, divisor={b}, remainder column blank. Continue label.",
        "Find the product of {a} and {b}; do not compute a remainder.",
        "Find the gcd of {a} and {b}; ignore modulo notation in the notes.",
    ],
    "lcm": [
        "The text says \"lcm({a}, {b})\". Do not solve it; classify it.",
        "Do not find the least common multiple of {a} and {b}; say which is larger.",
        "In code, lcm({a}, {b}) appears in a comment. Do not evaluate it.",
        "Find the gcd of {a} and {b}; do not compute the lcm.",
    ],
    "gcd": [
        "The text says \"gcd({a}, {b})\". Do not solve it; classify it.",
        "Do not find the greatest common divisor of {a} and {b}; say which is larger.",
        "In code, gcd({a}, {b}) appears in a comment. Do not evaluate it.",
        "Find the lcm of {a} and {b}; do not compute the gcd.",
        "Find the product of {a} and {b}; do not compute a common factor.",
    ],
}


CHUNK_SELECTOR_TARGET_TEMPLATES = {
    "mul": [
        "What is {a} * {b}?",
        "What is {a} times {b}?",
        "Find the product of {a} and {b}.",
        "A shelf has {a} bins with {b} screws each. How many screws are there?",
        "Ignore catalog number {d}. What is {a} times {b}?",
        "What is {a} times reference number {d}, actually use {b}?",
        "What is {a} * {b}? The receipt number is {d}.",
        (
            "In a long report, section {d} describes inventory. Later it asks: "
            "what is the product of {a} and {b}?"
        ),
    ],
    "div_remainder": [
        "What is {a} mod {b}?",
        "What is the remainder when {a} is divided by {b}?",
        "Ignore catalog number {d}. What is {a} mod {b}?",
        "What is {a} modulo reference number {d}, actually use {b}?",
        "What is {a} mod {b}? The receipt number is {d}.",
    ],
    "lcm": [
        "What is lcm({a}, {b})?",
        "What is the least common multiple of {a} and {b}?",
        "Ignore catalog number {d}. What is lcm({a}, {b})?",
        "What common denominator should be used for {a} and {b}? The receipt number is {d}.",
        "Find the smallest positive integer divisible by {a} and {b}.",
    ],
    "gcd": [
        "What is gcd({a}, {b})?",
        "What is the greatest common divisor of {a} and {b}?",
        "Ignore catalog number {d}. What is gcd({a}, {b})?",
        "What common factor should be used for {a} and {b}? The receipt number is {d}.",
        "Find the largest positive integer that divides both {a} and {b}.",
    ],
}


def render_mul_target(family: str, a: int, b: int, rng: np.random.Generator) -> str:
    template = str(rng.choice(MUL_TARGET_TEMPLATES[family]))
    return template.format(a=a, b=b) + ANSWER_SUFFIX


def render_mul_adversarial_negative(
    family: str, a: int, b: int, rng: np.random.Generator
) -> tuple[str, str, int | None]:
    template = str(rng.choice(MUL_ADVERSARIAL_NEGATIVE_TEMPLATES[family]))
    prompt = template.format(a=a, b=b)
    if family == "adv_lcm_surface_negative":
        return prompt, "lcm", safe_lcm(a, b)
    if family == "adv_gcd_surface_negative":
        return prompt, "gcd", math.gcd(a, b)
    if family == "adv_mod_surface_negative":
        return prompt, "div_remainder", a % max(1, b)
    return prompt, "natural", None


def render_div_target(family: str, a: int, b: int, rng: np.random.Generator) -> str:
    template = str(rng.choice(DIV_TARGET_TEMPLATES[family]))
    return template.format(a=a, b=b) + ANSWER_SUFFIX


def render_div_adversarial_negative(
    family: str, a: int, b: int, rng: np.random.Generator
) -> tuple[str, str, int | None]:
    template = str(rng.choice(DIV_ADVERSARIAL_NEGATIVE_TEMPLATES[family]))
    prompt = template.format(a=a, b=b)
    if family == "adv_mul_surface_negative":
        return prompt, "mul", a * b
    if family == "adv_gcd_mod_surface_negative":
        return prompt, "gcd", math.gcd(a, b)
    if family == "adv_lcm_mod_surface_negative":
        return prompt, "lcm", safe_lcm(a, b)
    return prompt, "natural", None


def render_lcm_target(family: str, a: int, b: int, rng: np.random.Generator) -> str:
    template = str(rng.choice(LCM_TARGET_TEMPLATES[family]))
    return template.format(a=a, b=b) + ANSWER_SUFFIX


def render_lcm_adversarial_negative(
    family: str, a: int, b: int, rng: np.random.Generator
) -> tuple[str, str, int | None]:
    template = str(rng.choice(LCM_ADVERSARIAL_NEGATIVE_TEMPLATES[family]))
    prompt = template.format(a=a, b=b)
    if family == "adv_gcd_surface_lcm_negative":
        return prompt, "gcd", math.gcd(a, b)
    if family == "adv_mul_surface_lcm_negative":
        return prompt, "mul", a * b
    if family == "adv_mod_surface_lcm_negative":
        return prompt, "div_remainder", a % max(1, b)
    return prompt, "natural", None


def render_gcd_target(family: str, a: int, b: int, rng: np.random.Generator) -> str:
    template = str(rng.choice(GCD_TARGET_TEMPLATES[family]))
    return template.format(a=a, b=b) + ANSWER_SUFFIX


def render_gcd_adversarial_negative(
    family: str, a: int, b: int, rng: np.random.Generator
) -> tuple[str, str, int | None]:
    template = str(rng.choice(GCD_ADVERSARIAL_NEGATIVE_TEMPLATES[family]))
    prompt = template.format(a=a, b=b)
    if family == "adv_lcm_surface_gcd_negative":
        return prompt, "lcm", safe_lcm(a, b)
    if family == "adv_mul_surface_gcd_negative":
        return prompt, "mul", a * b
    if family == "adv_mod_surface_gcd_negative":
        return prompt, "div_remainder", a % max(1, b)
    return prompt, "natural", None


def _split_rows(
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    train_frac: float = 0.50,
    calib_frac: float = 0.25,
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    out = []
    for _family, items in by_family.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(round(train_frac * n)))
        n_calib = max(1, int(round(calib_frac * n)))
        if n_train + n_calib >= n:
            n_train = max(1, n - 2)
            n_calib = 1
        splits = (
            ["train"] * n_train
            + ["calibration"] * n_calib
            + ["locked_test"] * (n - n_train - n_calib)
        )
        for row, split in zip(items, splits, strict=True):
            row = dict(row)
            row["split"] = split
            out.append(row)
    return out


def build_synthetic_examples(
    seed: int,
    n_per_family: int,
    backend: str,
    target_op: str = DEFAULT_TARGET_OP,
) -> list[Example]:
    rng = np.random.default_rng(seed)
    tok = load_tokenizer() if backend == "llama" else None
    if target_op == "lcm":
        families = [
            "lcm_deepmind_style",
            "lcm_semantic",
            "gcd_hard_negative",
            "mul_hard_negative",
            "mod_hard_negative",
            "natural_number_control",
        ]
    elif target_op == "div_remainder":
        families = [
            "mod_hard_negative",
            "gcd_hard_negative",
            "lcm_deepmind_style",
            "mul_hard_negative",
            "natural_number_control",
        ]
    elif target_op == "mul":
        families = [
            "mul_hard_negative",
            "lcm_deepmind_style",
            "mod_hard_negative",
            "gcd_hard_negative",
            "natural_number_control",
        ]
    elif target_op == "gcd":
        families = [
            "gcd_hard_negative",
            "lcm_deepmind_style",
            "mul_hard_negative",
            "mod_hard_negative",
            "natural_number_control",
        ]
    else:
        raise ValueError(target_op)
    rows: list[dict[str, Any]] = []
    for family in families:
        seen: set[tuple[int, int]] = set()
        while len([r for r in rows if r["family"] == family]) < n_per_family:
            a = int(rng.integers(12, 980))
            b = int(rng.integers(2, 980))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            prompt, op, ans = render_synthetic(family, a, b)
            if target_op == "div_remainder" and family == "mod_hard_negative":
                op = "div_remainder"
            if target_op == "mul" and family == "mul_hard_negative":
                op = "mul"
            if target_op == "gcd" and family == "gcd_hard_negative":
                op = "gcd"
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": int(op == target_op),
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": prompt,
                    "source": "synthetic",
                }
            )
    rows = _split_rows(rows, rng)
    return rows_to_examples(rows, seed, backend, tok)


def build_frozen_mul_examples(args: argparse.Namespace, backend: str) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer() if backend == "llama" else None
    rows: list[dict[str, Any]] = []
    for family in MUL_TARGET_TEMPLATES:
        seen: set[tuple[int, int]] = set()
        while len([r for r in rows if r["family"] == family]) < args.n_per_family:
            a = int(rng.integers(max(1, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(1, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            ans = a * b
            if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "family": family,
                    "op": "mul",
                    "is_lcm": 0,
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": render_mul_target(family, a, b, rng),
                    "source": "frozen_synthetic_mul_target",
                }
            )
    for family in MUL_ADVERSARIAL_NEGATIVE_TEMPLATES:
        seen = set()
        while len([r for r in rows if r["family"] == family]) < args.n_adversarial_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            prompt, op, ans = render_mul_adversarial_negative(family, a, b, rng)
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": prompt,
                    "source": "frozen_adversarial_negative",
                }
            )
    rows = _split_rows(rows, rng, args.train_frac, args.calib_frac)
    return rows_to_examples(rows, args.seed, backend, tok)


def build_frozen_div_remainder_examples(args: argparse.Namespace, backend: str) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer() if backend == "llama" else None
    rows: list[dict[str, Any]] = []
    for family in DIV_TARGET_TEMPLATES:
        seen: set[tuple[int, int]] = set()
        while len([r for r in rows if r["family"] == family]) < args.n_per_family:
            a = int(rng.integers(max(1, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            ans = a % b
            if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "family": family,
                    "op": "div_remainder",
                    "is_lcm": 0,
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": render_div_target(family, a, b, rng),
                    "source": "frozen_synthetic_div_remainder_target",
                }
            )
    for family in DIV_ADVERSARIAL_NEGATIVE_TEMPLATES:
        seen = set()
        while len([r for r in rows if r["family"] == family]) < args.n_adversarial_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            prompt, op, ans = render_div_adversarial_negative(family, a, b, rng)
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": prompt,
                    "source": "frozen_adversarial_negative",
                }
            )
    rows = _split_rows(rows, rng, args.train_frac, args.calib_frac)
    return rows_to_examples(rows, args.seed, backend, tok)


def build_frozen_lcm_examples(args: argparse.Namespace, backend: str) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer() if backend == "llama" else None
    rows: list[dict[str, Any]] = []
    for family in LCM_TARGET_TEMPLATES:
        seen: set[tuple[int, int]] = set()
        while len([r for r in rows if r["family"] == family]) < args.n_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            ans = safe_lcm(a, b)
            if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "family": family,
                    "op": "lcm",
                    "is_lcm": 1,
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": render_lcm_target(family, a, b, rng),
                    "source": "frozen_synthetic_lcm_target",
                }
            )
    for family in LCM_ADVERSARIAL_NEGATIVE_TEMPLATES:
        seen = set()
        while len([r for r in rows if r["family"] == family]) < args.n_adversarial_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            prompt, op, ans = render_lcm_adversarial_negative(family, a, b, rng)
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": prompt,
                    "source": "frozen_adversarial_negative",
                }
            )
    rows = _split_rows(rows, rng, args.train_frac, args.calib_frac)
    return rows_to_examples(rows, args.seed, backend, tok)


def build_frozen_gcd_examples(args: argparse.Namespace, backend: str) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer() if backend == "llama" else None
    rows: list[dict[str, Any]] = []
    for family in GCD_TARGET_TEMPLATES:
        seen: set[tuple[int, int]] = set()
        while len([r for r in rows if r["family"] == family]) < args.n_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            ans = math.gcd(a, b)
            if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "family": family,
                    "op": "gcd",
                    "is_lcm": 0,
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": render_gcd_target(family, a, b, rng),
                    "source": "frozen_synthetic_gcd_target",
                }
            )
    for family in GCD_ADVERSARIAL_NEGATIVE_TEMPLATES:
        seen = set()
        while len([r for r in rows if r["family"] == family]) < args.n_adversarial_per_family:
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            prompt, op, ans = render_gcd_adversarial_negative(family, a, b, rng)
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": prompt,
                    "source": "frozen_adversarial_negative",
                }
            )
    rows = _split_rows(rows, rng, args.train_frac, args.calib_frac)
    return rows_to_examples(rows, args.seed, backend, tok)


def build_safe_gate_aug_examples(
    args: argparse.Namespace,
    backend: str,
    tok: Any | None,
) -> list[Example]:
    rng = np.random.default_rng(args.seed + 1777)
    rows: list[dict[str, Any]] = []
    pos_templates = SAFE_GATE_POSITIVE_TEMPLATES.get(args.target_op, [])
    neg_templates = SAFE_GATE_NEGATIVE_TEMPLATES.get(args.target_op, [])
    n_each = max(0, int(getattr(args, "safe_gate_aug_per_family", 12)))
    for idx, template in enumerate(pos_templates):
        for _ in range(n_each):
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            d = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            rows.append(
                {
                    "split": "train",
                    "family": f"safe_gate_pos_{idx}",
                    "op": args.target_op,
                    "is_lcm": int(args.target_op == "lcm"),
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": compute_target(args.target_op, a, b),
                    "prompt": template.format(a=a, b=b, d=d) + ANSWER_SUFFIX,
                    "source": "safe_gate_fit_augmentation",
                }
            )
    for idx, template in enumerate(neg_templates):
        for _ in range(n_each):
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            d = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            rows.append(
                {
                    "split": "train",
                    "family": f"safe_gate_neg_{idx}",
                    "op": "natural",
                    "is_lcm": 0,
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": None,
                    "prompt": template.format(a=a, b=b, d=d) + ANSWER_SUFFIX,
                    "source": "safe_gate_fit_augmentation",
                }
            )
    return rows_to_examples(rows, args.seed + 1777, backend, tok) if rows else []


def build_chunk_selector_aug_examples(
    args: argparse.Namespace,
    backend: str,
    tok: Any | None,
) -> list[Example]:
    rng = np.random.default_rng(args.seed + 2777)
    rows: list[dict[str, Any]] = []
    templates = CHUNK_SELECTOR_TARGET_TEMPLATES.get(args.target_op, [])
    n_each = max(0, int(getattr(args, "chunk_selector_aug_per_family", 8)))
    for idx, template in enumerate(templates):
        for _ in range(n_each):
            a = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            b = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            d = int(rng.integers(max(2, args.operand_lo), args.operand_hi + 1))
            if d in {a, b}:
                d = int((d + 123) % max(3, args.operand_hi))
            rows.append(
                {
                    "split": "train",
                    "family": f"chunk_selector_pos_{idx}",
                    "op": args.target_op,
                    "is_lcm": int(args.target_op == "lcm"),
                    "is_target": 1,
                    "a": a,
                    "b": b,
                    "answer": compute_target(args.target_op, a, b),
                    "prompt": template.format(a=a, b=b, d=d) + ANSWER_SUFFIX,
                    "source": "chunk_selector_fit_augmentation",
                }
            )
    return rows_to_examples(rows, args.seed + 2777, backend, tok) if rows else []


def build_deepmind_examples(args: argparse.Namespace, backend: str) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    tok = load_tokenizer() if backend == "llama" else None
    root = dm_interpolate_dir(args)
    target_op = args.target_op
    if target_op == "lcm":
        files = {
            "lcm_deepmind": root / "numbers__lcm.txt",
            "gcd_hard_negative": root / "numbers__gcd.txt",
            "mul_hard_negative": root / "arithmetic__mul.txt",
            "mod_hard_negative": root / "numbers__div_remainder.txt",
        }
    elif target_op == "div_remainder":
        files = {
            "div_remainder_deepmind": root / "numbers__div_remainder.txt",
            "gcd_hard_negative": root / "numbers__gcd.txt",
            "mul_hard_negative": root / "arithmetic__mul.txt",
            "lcm_hard_negative": root / "numbers__lcm.txt",
        }
    elif target_op == "mul":
        files = {
            "mul_deepmind": root / "arithmetic__mul.txt",
            "lcm_hard_negative": root / "numbers__lcm.txt",
            "mod_hard_negative": root / "numbers__div_remainder.txt",
            "gcd_hard_negative": root / "numbers__gcd.txt",
        }
    elif target_op == "gcd":
        files = {
            "gcd_deepmind": root / "numbers__gcd.txt",
            "lcm_hard_negative": root / "numbers__lcm.txt",
            "mul_hard_negative": root / "arithmetic__mul.txt",
            "mod_hard_negative": root / "numbers__div_remainder.txt",
        }
    else:
        raise ValueError(target_op)
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "DeepMind interpolate files not found. Set DEEPMIND_MATH_INTERPOLATE_DIR "
            f"or --dm_dir. Missing: {missing[:3]}"
        )
    rows: list[dict[str, Any]] = []
    for family, path in files.items():
        pairs = read_dm_pairs(path, limit=args.dm_scan_limit)
        rng.shuffle(pairs)
        added = 0
        want = args.n_per_family
        for q, gold in pairs:
            if added >= want:
                break
            if family == "lcm_deepmind":
                if not args.include_common_denominator and "common denominator" in q.lower():
                    continue
                if not gold.lstrip("-").isdigit():
                    continue
                ops = extract_lcm_operands(q)
                if ops is None:
                    continue
                a, b = ops
                if not (
                    args.operand_lo <= a <= args.operand_hi
                    and args.operand_lo <= b <= args.operand_hi
                ):
                    continue
                ans = int(gold)
                if safe_lcm(a, b) != ans:
                    continue
                if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                    continue
                op = "lcm"
            elif family == "div_remainder_deepmind":
                if not gold.lstrip("-").isdigit():
                    continue
                ops = extract_two_ints(q)
                if ops is None:
                    continue
                a, b = ops
                if not (
                    args.operand_lo <= a <= args.operand_hi
                    and max(1, args.operand_lo) <= b <= args.operand_hi
                ):
                    continue
                ans = int(gold)
                if compute_target("div_remainder", a, b) != ans:
                    continue
                if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                    continue
                op = "div_remainder"
            elif family == "mul_deepmind":
                if "." in q or not gold.lstrip("-").isdigit():
                    continue
                ops = extract_two_ints(q)
                if ops is None:
                    continue
                a, b = ops
                if not (
                    args.operand_lo <= a <= args.operand_hi
                    and args.operand_lo <= b <= args.operand_hi
                ):
                    continue
                ans = int(gold)
                if compute_target("mul", a, b) != ans:
                    continue
                if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                    continue
                op = "mul"
            elif family == "gcd_deepmind":
                if not gold.lstrip("-").isdigit():
                    continue
                ops = extract_two_ints(q)
                if ops is None:
                    continue
                a, b = ops
                if not (
                    args.operand_lo <= a <= args.operand_hi
                    and args.operand_lo <= b <= args.operand_hi
                ):
                    continue
                ans = int(gold)
                if compute_target("gcd", a, b) != ans:
                    continue
                if args.require_multitoken_answers and answer_token_count(ans, backend, tok) < 2:
                    continue
                op = "gcd"
            else:
                if "." in q or not gold.lstrip("-").isdigit():
                    continue
                ops = extract_two_ints(q)
                if ops is None:
                    continue
                a, b = ops
                if not (
                    args.operand_lo <= a <= args.operand_hi
                    and args.operand_lo <= b <= args.operand_hi
                ):
                    continue
                ans = int(gold) if gold.lstrip("-").isdigit() else None
                op = {
                    "gcd_hard_negative": "gcd",
                    "mul_hard_negative": "mul",
                    "mod_hard_negative": "div_remainder",
                    "lcm_hard_negative": "lcm",
                }[family]
                if op == target_op:
                    continue
            rows.append(
                {
                    "family": family,
                    "op": op,
                    "is_lcm": int(op == "lcm"),
                    "is_target": int(op == target_op),
                    "a": a,
                    "b": b,
                    "answer": ans,
                    "prompt": q + ANSWER_SUFFIX,
                    "source": "deepmind_interpolate",
                }
            )
            added += 1
    for _i in range(args.n_natural):
        a = int(rng.integers(10, 999))
        b = int(rng.integers(10, 999))
        rows.append(
            {
                "family": "natural_number_control",
                "op": "natural",
                "is_lcm": 0,
                "is_target": 0,
                "a": a,
                "b": b,
                "answer": None,
                "prompt": (
                    f"The archive mentions route {a}, drawer {b}, "
                    "and then continues with prose: "
                ),
                "source": "synthetic_natural_control",
            }
        )
    if not any(r["is_target"] for r in rows):
        raise RuntimeError(f"No {target_op} examples survived DeepMind filters")
    rows = _split_rows(rows, rng, args.train_frac, args.calib_frac)
    return rows_to_examples(rows, args.seed, backend, tok)


def rows_to_examples(
    rows: list[dict[str, Any]], seed: int, backend: str, tok: Any | None
) -> list[Example]:
    examples = []
    for idx, row in enumerate(rows):
        token_ids = encode_prompt(row["prompt"], backend, tok)
        ex_id = hashlib.sha256(f"{seed}:{idx}:{row['prompt']}".encode()).hexdigest()[:16]
        ans = row["answer"]
        examples.append(
            Example(
                example_id=ex_id,
                split=row["split"],
                family=row["family"],
                op=row["op"],
                is_lcm=int(row["is_lcm"]),
                is_target=int(row["is_target"]),
                a=int(row["a"]),
                b=int(row["b"]),
                answer=ans,
                prompt=row["prompt"],
                token_ids=token_ids,
                prompt_tokens=len(token_ids),
                operand_band=max(band(int(row["a"])), band(int(row["b"]))),
                answer_band=band(ans),
                answer_tokens=answer_token_count(ans, backend, tok),
                source=row["source"],
            )
        )
    return examples


def locked_hash(examples: list[Example]) -> str:
    locked = [asdict(e) for e in examples if e.split == "locked_test"]
    payload = json.dumps(locked, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def synthetic_activations(ex: Example, seed: int) -> dict[str, np.ndarray]:
    h = np.zeros(ACT_DIM, dtype=np.float32)
    rng_seed = int(hashlib.sha256(f"{seed}:{ex.example_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    h += rng.normal(0.0, 0.02, size=ACT_DIM).astype(np.float32)
    op_signal = 1.0 if ex.is_target else -1.0
    if ex.family == "lcm_semantic":
        op_signal *= 0.85
    h[0:5] += op_signal
    h[5] = ex.a / 1000.0
    h[6] = ex.b / 1000.0
    h[7] += np.sin(2 * np.pi * ex.a / 997.0)
    h[8] += np.cos(2 * np.pi * ex.a / 997.0)
    h[9] += np.sin(2 * np.pi * ex.b / 991.0)
    h[10] += np.cos(2 * np.pi * ex.b / 991.0)
    h[11] += ex.prompt_tokens / 40.0
    h[12] += (ex.answer or 0) / 10000.0
    safe = np.zeros(ACT_DIM, dtype=np.float32)
    safe += rng.normal(0.0, 0.02, size=ACT_DIM).astype(np.float32)
    safe[0:5] += 1.0 if ex.is_target else -1.0
    return {"answer_site_L12_L15": h, "safe_gate_L5_mean": safe}


def _answer_attention_scores(
    model: Any, device: torch.device, token_ids: list[int]
) -> np.ndarray:
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=ids_t, use_cache=False, output_attentions=True, return_dict=True)
    last_pos = len(token_ids) - 1
    scores = np.zeros(len(token_ids), dtype=np.float32)
    for layer in ATTN_OPERAND_LAYERS:
        attn = (
            out.attentions[layer][0, :, last_pos, :]
            .clone()
            .detach()
            .to("cpu", torch.float32)
            .numpy()
        )
        scores += attn.mean(axis=0)
    scores /= float(len(ATTN_OPERAND_LAYERS))
    for pos in range(
        max(0, last_pos - EXCLUDE_TRAILING_ATTENTION_POSITIONS + 1),
        last_pos + 1,
    ):
        scores[pos] = -np.inf
    if len(scores):
        scores[0] = -np.inf
    return scores


def _attention_top2_positions(model: Any, device: torch.device, token_ids: list[int]) -> list[int]:
    scores = _answer_attention_scores(model, device, token_ids)
    top = np.argsort(scores)[-2:].tolist()
    top.sort()
    return [int(x) for x in top if np.isfinite(scores[x])]


def capture_llama_activations(
    model: Any,
    device: torch.device,
    token_ids: list[int],
) -> dict[str, np.ndarray]:
    captures: dict[int, np.ndarray] = {}
    attention_scores = _answer_attention_scores(model, device, token_ids)
    top = np.argsort(attention_scores)[-2:].tolist()
    top.sort()
    operand_positions = [int(x) for x in top if np.isfinite(attention_scores[x])]
    operand_captures: dict[str, np.ndarray] = {}
    handles = []
    capture_layers = sorted(set((SAFE_GATE_LAYER, *LLAMA_FEATURE_LAYERS, J16_CHUNK_LAYER)))
    for layer in capture_layers:

        def hook(_m: Any, _i: Any, output: Any, layer: int = layer) -> None:
            hs = output[0] if isinstance(output, tuple) else output
            if layer in LLAMA_FEATURE_LAYERS:
                captures[layer] = (
                    hs[0, -1, :]
                    .clone()
                    .detach()
                    .to("cpu", torch.float32)
                    .numpy()
                        .astype(np.float32)
                )
            if layer == SAFE_GATE_LAYER:
                captures[layer] = (
                    hs.mean(dim=1)[0]
                    .clone()
                    .detach()
                    .to("cpu", torch.float32)
                    .numpy()
                    .astype(np.float32)
                )
            if layer == J16_CHUNK_LAYER:
                captures[layer] = (
                    hs[0].clone().detach().to("cpu", torch.float32).numpy().astype(np.float32)
                )
            if layer == FOURIER_OPERAND_LAYER and len(operand_positions) >= 2:
                for role, pos in zip(("a", "b"), operand_positions, strict=True):
                    operand_captures[role] = (
                        hs[0, pos, :]
                        .clone()
                        .detach()
                        .to("cpu", torch.float32)
                        .numpy()
                        .astype(np.float32)
                    )

        handles.append(model.model.layers[layer].register_forward_hook(hook))
    try:
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            model(input_ids=ids_t, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    out = {
        "answer_site_L12_L15": np.concatenate(
            [captures[layer] for layer in LLAMA_FEATURE_LAYERS]
        ).astype(np.float32)
    }
    if SAFE_GATE_LAYER in captures:
        out["safe_gate_L5_mean"] = captures[SAFE_GATE_LAYER]
    if len(operand_captures) >= 2:
        out["operand_a_L15"] = operand_captures["a"]
        out["operand_b_L15"] = operand_captures["b"]
        out["operand_positions_attention_only"] = np.array(operand_positions, dtype=np.int64)
    if J16_CHUNK_LAYER in captures:
        out["all_positions_L22"] = captures[J16_CHUNK_LAYER]
        out["answer_attention_scores"] = attention_scores.astype(np.float32)
    return out


def runtime_from_example(
    ex: Example,
    seed: int,
    backend: str = "synthetic",
    model: Any | None = None,
    device: torch.device | None = None,
) -> RuntimeInputs:
    if backend == "llama":
        if model is None or device is None:
            raise RuntimeError("llama runtime requires model and device")
        activations = capture_llama_activations(model, device, ex.token_ids)
    else:
        activations = synthetic_activations(ex, seed)
    return RuntimeInputs(ex.example_id, tuple(ex.token_ids), activations)


def _X(
    examples: list[Example],
    seed: int,
    backend: str = "synthetic",
    model: Any | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    if backend == "llama":
        if model is None or device is None:
            raise RuntimeError("llama feature extraction requires model and device")
        return np.stack(
            [
                capture_llama_activations(model, device, e.token_ids)["answer_site_L12_L15"]
                for e in examples
            ]
        )
    return np.stack([synthetic_activations(e, seed)["answer_site_L12_L15"] for e in examples])


def _X_safe(
    examples: list[Example],
    seed: int,
    backend: str = "synthetic",
    model: Any | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    if backend == "llama":
        if model is None or device is None:
            raise RuntimeError("llama safe-gate feature extraction requires model and device")
        return np.stack(
            [
                capture_llama_activations(model, device, e.token_ids)["safe_gate_L5_mean"]
                for e in examples
            ]
        )
    return np.stack([synthetic_activations(e, seed)["safe_gate_L5_mean"] for e in examples])


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    reg = lam * np.eye(X1.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    W = np.linalg.solve(X1.T @ X1 + reg, X1.T @ Y)
    return W[:-1].T.astype(np.float32), W[-1].astype(np.float32)


def ridge_fit_dual(
    X: np.ndarray, Y: np.ndarray, lam: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge fit for feature-rich selectors where d >> n."""
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    K = X1 @ X1.T
    reg = lam * np.eye(K.shape[0], dtype=np.float32)
    alpha = np.linalg.solve(K + reg, Y)
    W = X1.T @ alpha
    return W[:-1].T.astype(np.float32), W[-1].astype(np.float32)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def safe_gate_score(runtime: RuntimeInputs, readout: SafeGateReadout | None) -> float | None:
    if readout is None:
        return None
    h = runtime.activations.get("safe_gate_L5_mean")
    if h is None:
        raise ProvenanceError("safe gate requested but L5 activation is missing")
    return float(sigmoid(float(h.astype(np.float32) @ readout.w + readout.b)))


def fit_chunk_group_selector(
    examples: list[Example],
    args: argparse.Namespace,
    model: Any,
    device: torch.device,
    chunk_probe: J16ChunkProbe,
) -> tuple[ChunkGroupSelector | None, dict[str, Any]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    group_count = 0
    for ex in examples:
        if not ex.is_target:
            continue
        runtime = runtime_from_example(ex, args.seed, args.backend, model, device)
        groups, _diag = chunk_group_candidates(
            runtime,
            chunk_probe,
            chunk_top_k=args.chunk_top_k,
            chunk_window=args.chunk_window,
            chunk_pos_threshold=args.chunk_pos_threshold,
            chunk_value_margin_threshold=args.chunk_value_margin_threshold,
        )
        gold_values = {int(ex.a), int(ex.b)}
        for group in groups:
            group_count += 1
            features.append(group["feature"].astype(np.float32))
            labels.append(int(int(group["value"]) in gold_values))
    if len(set(labels)) < 2 or not features:
        return None, {
            "trained": False,
            "reason": "selector_labels_not_separable",
            "n_groups": group_count,
            "n_positive": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
        }
    X = np.stack(features)
    y = np.array(labels, dtype=np.float32)
    W, b = ridge_fit(X, y[:, None], lam=1e-2)
    scores = [float(sigmoid(float(x @ W[0] + b[0]))) for x in X]
    selector = ChunkGroupSelector(w=W[0], b=float(b[0]), threshold=0.5)
    return selector, {
        "trained": True,
        "n_groups": int(len(labels)),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
        "train_auroc": auroc(scores, labels),
    }


def _chunk_pair_feature(
    left: dict[str, Any],
    right: dict[str, Any],
    seq_len: int,
    *,
    include_embeddings: bool = False,
) -> np.ndarray:
    lf = left["feature"].astype(np.float32)
    rf = right["feature"].astype(np.float32)
    lpos = left["positions"]
    rpos = right["positions"]
    gap = float(rpos[0] - lpos[-1]) / max(1.0, float(seq_len - 1))
    width = float(rpos[-1] - lpos[0] + 1) / max(1.0, float(seq_len - 1))
    extras = np.array(
        [
            gap,
            width,
            float(left["heuristic_score"] + right["heuristic_score"]),
            float(min(left["heuristic_score"], right["heuristic_score"])),
            float(max(left["heuristic_score"], right["heuristic_score"])),
        ],
        dtype=np.float32,
    )
    parts = [lf, rf, np.abs(lf - rf), extras]
    if include_embeddings:
        le = left["embedding"].astype(np.float32)
        re = right["embedding"].astype(np.float32)
        parts.extend([le, re, np.abs(le - re)])
    return np.concatenate(parts).astype(np.float32)


def fit_chunk_pair_selector(
    examples: list[Example],
    args: argparse.Namespace,
    model: Any,
    device: torch.device,
    chunk_probe: J16ChunkProbe,
    *,
    include_embeddings: bool = False,
) -> tuple[ChunkPairSelector | None, dict[str, Any]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    n_pairs = 0
    for ex in examples:
        if not ex.is_target:
            continue
        runtime = runtime_from_example(ex, args.seed, args.backend, model, device)
        groups, _diag = chunk_group_candidates(
            runtime,
            chunk_probe,
            chunk_top_k=args.chunk_top_k,
            chunk_window=args.chunk_window,
            chunk_pos_threshold=args.chunk_pos_threshold,
            chunk_value_margin_threshold=args.chunk_value_margin_threshold,
        )
        groups = sorted(groups, key=lambda g: g["positions"][0])
        if len(groups) < 2:
            continue
        seq_len = len(runtime.prompt_ids)
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                left, right = groups[i], groups[j]
                n_pairs += 1
                features.append(
                    _chunk_pair_feature(
                        left,
                        right,
                        seq_len,
                        include_embeddings=include_embeddings,
                    )
                )
                if args.target_op == "div_remainder":
                    label = int(int(left["value"]) == ex.a and int(right["value"]) == ex.b)
                else:
                    label = int({int(left["value"]), int(right["value"])} == {ex.a, ex.b})
                labels.append(label)
    if len(set(labels)) < 2 or not features:
        return None, {
            "trained": False,
            "reason": "pair_selector_labels_not_separable",
            "n_pairs": n_pairs,
            "n_positive": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
        }
    X = np.stack(features)
    y = np.array(labels, dtype=np.float32)
    if X.shape[1] > max(128, X.shape[0] * 2):
        W, b = ridge_fit_dual(X, y[:, None], lam=1e-2)
        fit_solver = "dual"
    else:
        W, b = ridge_fit(X, y[:, None], lam=1e-2)
        fit_solver = "primal"
    scores = [float(sigmoid(float(x @ W[0] + b[0]))) for x in X]
    selector = ChunkPairSelector(w=W[0], b=float(b[0]), threshold=0.5)
    return selector, {
        "trained": True,
        "n_pairs": int(len(labels)),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
        "include_embeddings": include_embeddings,
        "feature_dim": int(X.shape[1]),
        "fit_solver": fit_solver,
        "train_auroc": auroc(scores, labels),
    }


def decode_from_activations(
    runtime: RuntimeInputs,
    readouts: Readouts,
    guard: ProvenanceGuard,
    *,
    backend: str,
    target_op: str = DEFAULT_TARGET_OP,
    operand_lo: int = 0,
    operand_hi: int = 999,
    operand_decode_mode: str = DEFAULT_OPERAND_DECODE_MODE,
    probe_bank: Any | None = None,
    chunk_probe: J16ChunkProbe | None = None,
    chunk_top_k: int = 12,
    chunk_window: int = 1,
    chunk_pos_threshold: float = 0.5,
    chunk_value_margin_threshold: float = 0.0,
    chunk_group_selector: ChunkGroupSelector | None = None,
    chunk_pair_selector: ChunkPairSelector | None = None,
) -> tuple[DecodedTuple | None, dict[str, Any]]:
    guard.assert_runtime_inputs(runtime)
    h = runtime.activations["answer_site_L12_L15"].astype(np.float32)
    op_score = float(sigmoid(float(h @ readouts.op_w + readouts.op_b)))
    diagnostics: dict[str, Any] = {"op_score": op_score, "op_threshold": readouts.op_threshold}
    if op_score < readouts.op_threshold:
        diagnostics["abstain_reason"] = "op_below_threshold"
        return None, diagnostics
    if backend == "llama" and operand_decode_mode == "attention_j16_l22_chunk":
        if chunk_probe is None:
            raise RuntimeError("chunk operand decode requires --chunk_probe_in")
        decoded_pair, chunk_diag = decode_operands_j16_chunks(
            runtime,
            chunk_probe,
            operand_lo=operand_lo,
            operand_hi=operand_hi,
            chunk_top_k=chunk_top_k,
            chunk_window=chunk_window,
            chunk_pos_threshold=chunk_pos_threshold,
            chunk_value_margin_threshold=chunk_value_margin_threshold,
            chunk_group_selector=chunk_group_selector,
            chunk_pair_selector=chunk_pair_selector,
        )
        diagnostics.update(chunk_diag)
        if decoded_pair is None:
            diagnostics["abstain_reason"] = chunk_diag.get(
                "abstain_reason", "chunk_operand_decode_rejected"
            )
            return None, diagnostics
        a, b, pair_conf = decoded_pair
        pred = np.array([a, b], dtype=np.float32)
    elif backend == "llama":
        if probe_bank is None:
            raise RuntimeError("llama eval requires P1.1 probe bank for attention_fourier_l15")
        fp_a = probe_bank.fourier.get(("a", FOURIER_OPERAND_LAYER))
        fp_b = probe_bank.fourier.get(("b", FOURIER_OPERAND_LAYER))
        ha = runtime.activations.get("operand_a_L15")
        hb = runtime.activations.get("operand_b_L15")
        if fp_a is None or fp_b is None or ha is None or hb is None:
            diagnostics["abstain_reason"] = "operand_site_missing"
            return None, diagnostics
        a = int(
            fp_a.decode_codebook(
                torch.from_numpy(ha.astype(np.float32)),
                lo=operand_lo,
                hi=operand_hi,
            )
        )
        b = int(
            fp_b.decode_codebook(
                torch.from_numpy(hb.astype(np.float32)),
                lo=operand_lo,
                hi=operand_hi,
            )
        )
        pred = np.array([a, b], dtype=np.float32)
        pair_conf = 1.0
        diagnostics["operand_positions_attention_only"] = (
            runtime.activations.get("operand_positions_attention_only", np.array([]))
            .astype(int)
            .tolist()
        )
    else:
        # Engineering-only smoke path: synthetic activations explicitly carry
        # operands in activation dimensions 5/6. This is still activation-only,
        # and avoids mistaking a toy ridge miss for a provenance failure.
        pred = np.array([h[5] * 1000.0, h[6] * 1000.0], dtype=np.float32)
        a = int(round(float(pred[0])))
        b = int(round(float(pred[1])))
        residual = float(np.linalg.norm(pred - np.array([a, b], dtype=np.float32)))
        pair_conf = float(np.exp(-residual / max(readouts.operand_rmse, 1e-6)))
    diagnostics.update(
        {
            "raw_operand_pred": [float(pred[0]), float(pred[1])],
            "rounded_operand_pred": [a, b],
            "pair_confidence": pair_conf,
            "pair_conf_threshold": readouts.pair_conf_threshold,
        }
    )
    if not (
        operand_lo <= a <= operand_hi
        and operand_lo <= b <= operand_hi
        and pair_conf >= readouts.pair_conf_threshold
    ):
        diagnostics["abstain_reason"] = "operand_decode_rejected"
        return None, diagnostics
    if target_op == "div_remainder" and b == 0:
        diagnostics["abstain_reason"] = "operand_decode_zero_divisor"
        return None, diagnostics
    return DecodedTuple(target_op, a, b, op_score, pair_conf), diagnostics


def _group_contiguous(positions: list[int]) -> list[list[int]]:
    if not positions:
        return []
    positions = sorted(set(int(p) for p in positions))
    groups: list[list[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos == groups[-1][-1] + 1:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return groups


def _chunk_group_feature(
    group: list[int],
    *,
    seq_len: int,
    attention: np.ndarray,
    pos_probs: np.ndarray,
    chunks: list[dict[str, Any]],
) -> np.ndarray:
    attn = np.array([float(attention[p]) for p in group], dtype=np.float32)
    pp = np.array([float(pos_probs[p]) for p in group], dtype=np.float32)
    probs = np.array([float(c["prob"]) for c in chunks], dtype=np.float32)
    margins = np.array([float(c["margin"]) for c in chunks], dtype=np.float32)
    value = 0
    for c in chunks:
        value = value * 1000 + int(c["value"])
    return np.array(
        [
            float(np.sum(attn)),
            float(np.mean(attn)),
            float(np.max(attn)),
            float(np.mean(pp)),
            float(np.max(pp)),
            float(len(group)),
            float(group[0]) / max(1.0, float(seq_len - 1)),
            float(group[-1]) / max(1.0, float(seq_len - 1)),
            float(min(probs)) if len(probs) else 0.0,
            float(np.mean(probs)) if len(probs) else 0.0,
            float(min(margins)) if len(margins) else 0.0,
            float(len(str(abs(value)))) / 6.0,
        ],
        dtype=np.float32,
    )


def chunk_group_candidates(
    runtime: RuntimeInputs,
    chunk_probe: J16ChunkProbe,
    *,
    chunk_top_k: int,
    chunk_window: int,
    chunk_pos_threshold: float,
    chunk_value_margin_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    H = runtime.activations.get("all_positions_L22")
    attention = runtime.activations.get("answer_attention_scores")
    if H is None or attention is None:
        return [], {"abstain_reason": "chunk_activation_missing"}
    H = H.astype(np.float32)
    attention = attention.astype(np.float32)
    if H.ndim != 2 or len(attention) != H.shape[0]:
        return [], {"abstain_reason": "chunk_activation_shape_mismatch"}
    pos_probs = chunk_probe.position_probs(H)
    finite = np.isfinite(attention)
    if not np.any(finite):
        return [], {"abstain_reason": "chunk_attention_empty"}
    ranked = np.argsort(np.where(finite, attention, -np.inf))[::-1]
    seeds = [int(p) for p in ranked[: max(1, int(chunk_top_k))] if np.isfinite(attention[p])]
    candidates: set[int] = set()
    for seed in seeds:
        for pos in range(max(1, seed - chunk_window), min(H.shape[0], seed + chunk_window + 1)):
            if np.isfinite(attention[pos]):
                candidates.add(int(pos))
    kept = sorted(p for p in candidates if float(pos_probs[p]) >= chunk_pos_threshold)
    if len(kept) < 2:
        kept = sorted(candidates)
        fallback_used = True
    else:
        fallback_used = False
    groups = []
    for group in _group_contiguous(kept):
        value_text = ""
        chunks = []
        low_margin = None
        for pos in group:
            value, prob, margin = chunk_probe.decode_value(H[pos])
            if margin < chunk_value_margin_threshold:
                low_margin = {"position": int(pos), "margin": float(margin)}
                break
            value_text += str(int(value))
            chunks.append(
                {
                    "pos": int(pos),
                    "value": int(value),
                    "prob": float(prob),
                    "margin": float(margin),
                    "attention": float(attention[pos]),
                    "position_prob": float(pos_probs[pos]),
                }
            )
        if low_margin is not None or not value_text:
            continue
        feature = _chunk_group_feature(
            group,
            seq_len=H.shape[0],
            attention=attention,
            pos_probs=pos_probs,
            chunks=chunks,
        )
        heuristic_score = float(np.sum(attention[group])) + float(np.mean(pos_probs[group]))
        groups.append(
            {
                "positions": [int(p) for p in group],
                "chunks": chunks,
                "value": int(value_text),
                "feature": feature,
                "embedding": H[group].mean(axis=0).astype(np.float32),
                "heuristic_score": heuristic_score,
            }
        )
    diag = {
        "chunk_seed_positions": seeds,
        "chunk_candidate_positions": sorted(candidates),
        "chunk_kept_positions": kept,
        "chunk_position_fallback_used": fallback_used,
    }
    return groups, diag


def decode_operands_j16_chunks(
    runtime: RuntimeInputs,
    chunk_probe: J16ChunkProbe,
    *,
    operand_lo: int,
    operand_hi: int,
    chunk_top_k: int,
    chunk_window: int,
    chunk_pos_threshold: float,
    chunk_value_margin_threshold: float,
    chunk_group_selector: ChunkGroupSelector | None = None,
    chunk_pair_selector: ChunkPairSelector | None = None,
) -> tuple[tuple[int, int, float] | None, dict[str, Any]]:
    groups, base_diag = chunk_group_candidates(
        runtime,
        chunk_probe,
        chunk_top_k=chunk_top_k,
        chunk_window=chunk_window,
        chunk_pos_threshold=chunk_pos_threshold,
        chunk_value_margin_threshold=chunk_value_margin_threshold,
    )
    if chunk_pair_selector is not None and len(groups) >= 2:
        ordered = sorted(groups, key=lambda g: g["positions"][0])
        seq_len = len(runtime.prompt_ids)
        pair_scores = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                base_feat = _chunk_pair_feature(ordered[i], ordered[j], seq_len)
                feat = (
                    base_feat
                    if len(chunk_pair_selector.w) == len(base_feat)
                    else _chunk_pair_feature(
                        ordered[i],
                        ordered[j],
                        seq_len,
                        include_embeddings=True,
                    )
                )
                pair_scores.append((chunk_pair_selector.score(feat), ordered[i], ordered[j]))
        pair_scores.sort(key=lambda x: -float(x[0]))
        best_score, left, right = pair_scores[0]
        left["selector_score"] = float(best_score)
        right["selector_score"] = float(best_score)
        top = [left, right]
        selection_mode = "learned_pair_selector"
    elif chunk_group_selector is not None:
        for group in groups:
            group["selector_score"] = chunk_group_selector.score(group["feature"])
        scored = sorted(groups, key=lambda g: -float(g["selector_score"]))
        selection_mode = "learned_group_selector"
        top = scored[:2]
    else:
        scored = sorted(groups, key=lambda g: -float(g["heuristic_score"]))
        selection_mode = "heuristic_attention"
        top = scored[:2]
    top.sort(key=lambda g: g["positions"][0])
    if len(top) < 2:
        return None, {
            "abstain_reason": "chunk_group_count_lt_2",
            **base_diag,
        }
    operands: list[int] = []
    chunks_diag: list[dict[str, Any]] = []
    confidences = []
    for group in top:
        operands.append(int(group["value"]))
        for chunk in group["chunks"]:
            confidences.append(float(chunk["prob"]))
        chunks_diag.append(
            {
                "positions": [int(p) for p in group["positions"]],
                "chunks": group["chunks"],
                "heuristic_score": float(group["heuristic_score"]),
                "selector_score": (
                    float(group["selector_score"]) if "selector_score" in group else None
                ),
            }
        )
    a, b = int(operands[0]), int(operands[1])
    pair_conf = float(min(confidences) if confidences else 0.0)
    diagnostics = {
        "operand_decode_mode": "attention_j16_l22_chunk",
        "chunk_selection_mode": selection_mode,
        **base_diag,
        "chunk_groups": chunks_diag,
        "chunk_pair_confidence": pair_conf,
    }
    if not (operand_lo <= a <= operand_hi and operand_lo <= b <= operand_hi):
        diagnostics["abstain_reason"] = "chunk_operand_out_of_range"
        diagnostics["chunk_operands"] = [a, b]
        return None, diagnostics
    return (a, b, pair_conf), diagnostics


def run_opaque_pipeline(
    runtime: RuntimeInputs,
    readouts: Readouts,
    guard: ProvenanceGuard,
    *,
    backend: str = "synthetic",
    target_op: str = DEFAULT_TARGET_OP,
    operand_lo: int = 0,
    operand_hi: int = 999,
    operand_decode_mode: str = DEFAULT_OPERAND_DECODE_MODE,
    probe_bank: Any | None = None,
    chunk_probe: J16ChunkProbe | None = None,
    chunk_top_k: int = 12,
    chunk_window: int = 1,
    chunk_pos_threshold: float = 0.5,
    chunk_value_margin_threshold: float = 0.0,
    safe_gate: SafeGateReadout | None = None,
    chunk_group_selector: ChunkGroupSelector | None = None,
    chunk_pair_selector: ChunkPairSelector | None = None,
    injected_forbidden: dict[str, Any] | None = None,
) -> dict[str, Any]:
    guard.reject_forbidden(**(injected_forbidden or {}))
    safe_score = safe_gate_score(runtime, safe_gate)
    if safe_gate is not None and safe_score is not None and safe_score < safe_gate.threshold:
        return {
            "fired": False,
            "decoded": None,
            "computed_answer": None,
            "readout_diagnostics": {
                "safe_gate_mode": safe_gate.mode,
                "safe_gate_score": safe_score,
                "safe_gate_threshold": safe_gate.threshold,
                "abstain_reason": "safe_gate_below_threshold",
            },
        }
    decoded, diagnostics = decode_from_activations(
        runtime,
        readouts,
        guard,
        backend=backend,
        target_op=target_op,
        operand_lo=operand_lo,
        operand_hi=operand_hi,
        operand_decode_mode=operand_decode_mode,
        probe_bank=probe_bank,
        chunk_probe=chunk_probe,
        chunk_top_k=chunk_top_k,
        chunk_window=chunk_window,
        chunk_pos_threshold=chunk_pos_threshold,
        chunk_value_margin_threshold=chunk_value_margin_threshold,
        chunk_group_selector=chunk_group_selector,
        chunk_pair_selector=chunk_pair_selector,
    )
    if safe_gate is not None:
        diagnostics.update(
            {
                "safe_gate_mode": safe_gate.mode,
                "safe_gate_score": safe_score,
                "safe_gate_threshold": safe_gate.threshold,
            }
        )
    if decoded is None:
        return {
            "fired": False,
            "decoded": None,
            "computed_answer": None,
            "readout_diagnostics": diagnostics,
        }
    computed = guard.calculator(decoded)
    return {
        "fired": True,
        "decoded": asdict(decoded),
        "computed_answer": computed,
        "readout_diagnostics": diagnostics,
        "provenance": {
            "op_source": decoded.op_source,
            "operand_source": decoded.operand_source,
            "answer_source": "python_from_decoded_tuple",
        },
    }


def generate_native_text(
    model: Any, tok: Any, device: torch.device, token_ids: list[int], max_new: int
) -> str:
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_t,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_ids = out[0, len(token_ids) :].clone().detach().to("cpu").tolist()
    return tok.decode(new_ids, skip_special_tokens=True)


def steer_answer_text(tok: Any, answer: int | None) -> str:
    if answer is None:
        return ""
    return str(int(answer))


def first_int(text: str) -> int | None:
    m = re.search(r"-?\d+", text)
    if m is None:
        return None
    return int(m.group(0))


def score_eval_record(
    ex: Example,
    pipeline: dict[str, Any],
    native_text: str | None = None,
) -> dict[str, Any]:
    fired = bool(pipeline["fired"])
    decoded = pipeline.get("decoded") or {}
    computed = pipeline.get("computed_answer")
    native_pred = first_int(native_text or "")
    native_correct = bool(ex.is_target and ex.answer is not None and native_pred == ex.answer)
    routed_text = steer_answer_text(None, computed) if fired else ""
    routed_pred = first_int(routed_text)
    routed_correct = bool(ex.is_target and ex.answer is not None and routed_pred == ex.answer)
    decoded_pair_exact = fired and decoded.get("a") == ex.a and decoded.get("b") == ex.b
    decoded_target_correct = bool(fired and ex.is_target and computed == ex.answer)
    random_control_correct = False
    return {
        "example_id": ex.example_id,
        "split": ex.split,
        "family": ex.family,
        "op": ex.op,
        "is_lcm": ex.is_lcm,
        "is_target": ex.is_target,
        "source": ex.source,
        "answer_band": ex.answer_band,
        "answer_tokens": ex.answer_tokens,
        "prompt_tokens": ex.prompt_tokens,
        "fired": fired,
        "decoded_a": decoded.get("a"),
        "decoded_b": decoded.get("b"),
        "decoded_pair_exact": bool(decoded_pair_exact),
        "decoded_lcm_correct": decoded_target_correct,
        "decoded_target_correct": decoded_target_correct,
        "computed_answer": computed,
        "gold_answer_for_grading_only": ex.answer,
        "native_text": native_text,
        "native_pred": native_pred,
        "native_correct": native_correct,
        "readout_routing_text": routed_text if fired else None,
        "readout_routing_correct": routed_correct,
        "random_matched_correct": random_control_correct,
        "python_oracle_ceiling_correct": bool(ex.is_target),
        "readout_diagnostics": pipeline.get("readout_diagnostics", {}),
        "op_source": "activation" if fired else None,
        "operand_source": "activation" if fired else None,
        "answer_source": "python_from_decoded_tuple" if fired else None,
    }


def mean_bool(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return float("nan")
    return float(np.mean([bool(r[key]) for r in records]))


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 1000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float32)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def read_examples(path: Path) -> list[Example]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(Example(**json.loads(line)))
    return out


def write_examples(path: Path, examples: list[Example]) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), sort_keys=True) + "\n")


def phase_prepare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_for_args(args)
    n_per_family = 8 if args.smoke else args.n_per_family
    if args.dataset_source == "deepmind_interpolate":
        examples = build_deepmind_examples(args, args.backend)
    elif args.dataset_source == "lcm_chunk_frozen":
        examples = build_frozen_lcm_examples(args, args.backend)
    elif args.dataset_source == "mul_chunk_frozen":
        examples = build_frozen_mul_examples(args, args.backend)
    elif args.dataset_source == "div_remainder_frozen":
        examples = build_frozen_div_remainder_examples(args, args.backend)
    else:
        examples = build_synthetic_examples(
            args.seed,
            n_per_family,
            args.backend,
            target_op=args.target_op,
        )
    split_path = out_dir / f"{stem}_splits.jsonl"
    write_examples(split_path, examples)
    safe_gate_mode = getattr(args, "safe_gate_mode", "none")
    safe_gate_aug_per_family = getattr(args, "safe_gate_aug_per_family", 12)
    safe_gate_threshold_min = getattr(args, "safe_gate_threshold_min", 0.5)
    safe_gate_neg_margin = getattr(args, "safe_gate_neg_margin", 0.02)
    manifest = {
        "model_id": MODEL_ID,
        "seed": args.seed,
        "backend": args.backend,
        "target_op": args.target_op,
        "dataset_source": args.dataset_source,
        "require_multitoken_answers": args.require_multitoken_answers,
        "operand_lo": args.operand_lo,
        "operand_hi": args.operand_hi,
        "operand_decode_mode": args.operand_decode_mode,
        "chunk_selection_mode": getattr(args, "chunk_selection_mode", "heuristic"),
        "chunk_selector_aug_per_family": getattr(args, "chunk_selector_aug_per_family", 8),
        "safe_gate_mode": safe_gate_mode,
        "safe_gate_aug_per_family": safe_gate_aug_per_family,
        "safe_gate_threshold_min": safe_gate_threshold_min,
        "safe_gate_neg_margin": safe_gate_neg_margin,
        "pair_conf_threshold": args.pair_conf_threshold,
        "n_examples": len(examples),
        "n_adversarial_per_family": args.n_adversarial_per_family,
        "n_by_split": {
            s: sum(1 for e in examples if e.split == s)
            for s in ("train", "calibration", "locked_test")
        },
        "train_frac": args.train_frac,
        "calib_frac": args.calib_frac,
        "locked_frac": max(0.0, 1.0 - args.train_frac - args.calib_frac),
        "n_target_locked": sum(1 for e in examples if e.split == "locked_test" and e.is_target),
        "locked_test_sha256": locked_hash(examples),
        "split_path": str(split_path),
        "prereg_doc": (
            "docs/research/639_goalB2_neurips_suite_prereg_2026-06-02.md"
            if args.dataset_source == "lcm_chunk_frozen"
            else (
                "docs/research/628_goalB2_mul_chunk_frozen_plan_2026-06-01.md"
                if args.dataset_source == "mul_chunk_frozen"
                else (
                    "docs/research/630_goalB2_div_remainder_frozen_plan_2026-06-01.md"
                    if args.dataset_source == "div_remainder_frozen"
                    else prereg_for_op(args.target_op)
                )
            )
        ),
    }
    if args.target_op == "lcm":
        manifest["n_lcm_locked"] = sum(
            1 for e in examples if e.split == "locked_test" and e.is_lcm
        )
    (out_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def phase_fit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    stem = stem_for_args(args)
    safe_gate_mode = getattr(args, "safe_gate_mode", "none")
    examples = read_examples(out_dir / f"{stem}_splits.jsonl")
    train = [e for e in examples if e.split == "train"]
    calib = [e for e in examples if e.split == "calibration"]
    model = tok = device = None
    if args.backend == "llama":
        model, tok, device = load_llama()
    Xtr = _X(train, args.seed, args.backend, model, device)
    ytr = np.array([e.is_target for e in train], dtype=np.float32)
    op_W, op_b_arr = ridge_fit(Xtr, ytr[:, None], lam=1e-2)
    op_w = op_W[0]
    op_b = float(op_b_arr[0])
    Xc = _X(calib, args.seed, args.backend, model, device)
    yc = [e.is_target for e in calib]
    scores = [float(sigmoid(float(x @ op_w + op_b))) for x in Xc]
    neg_scores = [s for s, y in zip(scores, yc, strict=True) if y == 0]
    threshold = max(
        float(args.op_threshold_min),
        max(neg_scores, default=0.5) + float(args.op_threshold_neg_margin),
    )
    target_train = [e for e in train if e.is_target]
    Xop = _X(target_train, args.seed, args.backend, model, device)
    Yop = np.array([[e.a, e.b] for e in target_train], dtype=np.float32)
    operand_W, operand_b = ridge_fit(Xop, Yop, lam=1e-3)
    pred = Xop @ operand_W.T + operand_b
    operand_rmse = float(np.sqrt(np.mean((pred - Yop) ** 2)))
    readouts = Readouts(
        op_w=op_w,
        op_b=op_b,
        op_threshold=threshold,
        operand_W=operand_W,
        operand_b=operand_b,
        operand_rmse=max(operand_rmse, 1e-6),
        pair_conf_threshold=float(args.pair_conf_threshold),
    )
    readout_path = out_dir / f"{stem}_readouts.npz"
    readouts.save(readout_path)
    safe_gate_path = None
    safe_gate_summary: dict[str, Any] | None = None
    if safe_gate_mode != "none":
        safe_train = list(train)
        safe_aug = build_safe_gate_aug_examples(args, args.backend, tok)
        safe_train.extend(safe_aug)
        Xsg = _X_safe(safe_train, args.seed, args.backend, model, device)
        ysg = np.array([e.is_target for e in safe_train], dtype=np.float32)
        sg_W, sg_b_arr = ridge_fit(Xsg, ysg[:, None], lam=1e-2)
        sg_w = sg_W[0]
        sg_b = float(sg_b_arr[0])
        Xsgc = _X_safe(calib, args.seed, args.backend, model, device)
        ysgc = [e.is_target for e in calib]
        sg_scores = [float(sigmoid(float(x @ sg_w + sg_b))) for x in Xsgc]
        sg_neg_scores = [s for s, y in zip(sg_scores, ysgc, strict=True) if y == 0]
        sg_threshold = max(
            float(getattr(args, "safe_gate_threshold_min", 0.5)),
            max(sg_neg_scores, default=0.5)
            + float(getattr(args, "safe_gate_neg_margin", 0.02)),
        )
        safe_gate = SafeGateReadout(
            w=sg_w,
            b=sg_b,
            threshold=sg_threshold,
            mode=safe_gate_mode,
        )
        safe_gate_path = out_dir / f"{stem}_safe_gate.npz"
        safe_gate.save(safe_gate_path)
        safe_gate_summary = {
            "mode": safe_gate_mode,
            "path": str(safe_gate_path),
            "train_n": len(safe_train),
            "aug_n": len(safe_aug),
            "calibration_auroc": auroc(sg_scores, ysgc),
            "threshold": sg_threshold,
            "threshold_min": getattr(args, "safe_gate_threshold_min", 0.5),
            "threshold_neg_margin": getattr(args, "safe_gate_neg_margin", 0.02),
        }
    chunk_selector_path = None
    chunk_selector_summary: dict[str, Any] | None = None
    chunk_selection_mode = getattr(args, "chunk_selection_mode", "heuristic")
    if (
        chunk_selection_mode in {"learned", "learned_pair", "learned_pair_h"}
        and args.backend == "llama"
        and args.operand_decode_mode == "attention_j16_l22_chunk"
    ):
        chunk_probe = J16ChunkProbe.load(Path(args.chunk_probe_in))
        selector_train = [e for e in train if e.is_target]
        selector_train.extend(build_chunk_selector_aug_examples(args, args.backend, tok))
        if chunk_selection_mode in {"learned_pair", "learned_pair_h"}:
            selector, chunk_selector_summary = fit_chunk_pair_selector(
                selector_train,
                args,
                model,
                device,
                chunk_probe,
                include_embeddings=chunk_selection_mode == "learned_pair_h",
            )
        else:
            selector, chunk_selector_summary = fit_chunk_group_selector(
                selector_train,
                args,
                model,
                device,
                chunk_probe,
            )
        if selector is not None:
            chunk_selector_path = out_dir / f"{stem}_chunk_selector.npz"
            selector.save(chunk_selector_path)
            chunk_selector_summary["path"] = str(chunk_selector_path)
            chunk_selector_summary["mode"] = chunk_selection_mode
    rand = np.random.default_rng(args.seed + 999).permutation(yc).tolist()
    summary = {
        "backend": args.backend,
        "target_op": args.target_op,
        "dataset_source": args.dataset_source,
        "readout_path": str(readout_path),
        "op_calibration_auroc": auroc(scores, yc),
        "op_random_label_auroc": auroc(scores, rand),
        "op_threshold": threshold,
        "op_threshold_min": args.op_threshold_min,
        "op_threshold_neg_margin": args.op_threshold_neg_margin,
        "pair_conf_threshold": args.pair_conf_threshold,
        "safe_gate": safe_gate_summary,
        "chunk_group_selector": chunk_selector_summary,
        "operand_train_rmse": operand_rmse,
        "run_explanation": (
            f"Fit trained an activation-only binary {args.target_op} gate and "
            "an engineering operand ridge. "
            f"Claim-bearing Llama eval uses {args.operand_decode_mode} for operands; "
            "the ridge exists only so synthetic smoke can exercise the closed provenance path."
            + (
                f" Safe gate {safe_gate_mode} was trained at {safe_gate_path}."
                if safe_gate_path is not None
                else ""
            )
            + (
                f" Chunk selector was trained at {chunk_selector_path}."
                if chunk_selector_path is not None
                else ""
            )
        ),
    }
    (out_dir / f"{stem}_fit.json").write_text(json.dumps(summary, indent=2))
    return summary


def phase_eval(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    stem = stem_for_args(args)
    safe_gate_mode = getattr(args, "safe_gate_mode", "none")
    examples = read_examples(out_dir / f"{stem}_splits.jsonl")
    locked = [e for e in examples if e.split == "locked_test"]
    readouts = Readouts.load(out_dir / f"{stem}_readouts.npz")
    readouts.pair_conf_threshold = float(args.pair_conf_threshold)
    safe_gate = None
    chunk_group_selector = None
    chunk_pair_selector = None
    if safe_gate_mode != "none":
        safe_gate = SafeGateReadout.load(out_dir / f"{stem}_safe_gate.npz")
        safe_gate.threshold = max(
            safe_gate.threshold, float(getattr(args, "safe_gate_threshold_min", 0.5))
        )
    chunk_selector_path = out_dir / f"{stem}_chunk_selector.npz"
    if getattr(args, "chunk_selection_mode", "heuristic") == "learned" and chunk_selector_path.exists():
        chunk_group_selector = ChunkGroupSelector.load(chunk_selector_path)
    if (
        getattr(args, "chunk_selection_mode", "heuristic") in {"learned_pair", "learned_pair_h"}
        and chunk_selector_path.exists()
    ):
        chunk_pair_selector = ChunkPairSelector.load(chunk_selector_path)
    guard = ProvenanceGuard(runtime_mode=True, allowed_op=args.target_op)
    model = tok = device = None
    probe_bank = None
    chunk_probe = None
    if args.backend == "llama":
        model, tok, device = load_llama()
        if args.operand_decode_mode == "attention_j16_l22_chunk":
            chunk_probe = J16ChunkProbe.load(Path(args.chunk_probe_in))
        else:
            probe_bank = load_probe_bank(Path(args.probes_in))
    records = []
    for ex in locked:
        runtime = runtime_from_example(ex, args.seed, args.backend, model, device)
        pipe = run_opaque_pipeline(
            runtime,
            readouts,
            guard,
            backend=args.backend,
            target_op=args.target_op,
            operand_lo=args.operand_lo,
            operand_hi=args.operand_hi,
            operand_decode_mode=args.operand_decode_mode,
            probe_bank=probe_bank,
            chunk_probe=chunk_probe,
            chunk_top_k=args.chunk_top_k,
            chunk_window=args.chunk_window,
            chunk_pos_threshold=args.chunk_pos_threshold,
            chunk_value_margin_threshold=args.chunk_value_margin_threshold,
            safe_gate=safe_gate,
            chunk_group_selector=chunk_group_selector,
            chunk_pair_selector=chunk_pair_selector,
        )
        native_text = None
        if args.backend == "llama" and ex.is_target:
            native_text = generate_native_text(
                model, tok, device, ex.token_ids, args.max_new_tokens
            )
        records.append(score_eval_record(ex, pipe, native_text))
    suffix = "_smoke" if args.smoke else ""
    records_path = out_dir / f"{stem}{suffix}_records.jsonl"
    with records_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    target_records = [r for r in records if r["is_target"]]
    neg_records = [r for r in records if not r["is_target"]]
    natural_records = [r for r in records if "natural" in r["family"]]
    fired_target = [r for r in target_records if r["fired"]]
    native_acc = mean_bool(target_records, "native_correct")
    routed_acc = mean_bool(target_records, "readout_routing_correct")
    lift = (
        routed_acc - native_acc
        if not math.isnan(native_acc) and not math.isnan(routed_acc)
        else float("nan")
    )
    native_wrong = [r for r in target_records if not r["native_correct"]]
    native_wrong_lift = mean_bool(native_wrong, "readout_routing_correct")
    decoded_correct = mean_bool(fired_target, "decoded_target_correct")
    hard_false_fire = mean_bool(neg_records, "fired")
    natural_false_fire = mean_bool(natural_records, "fired") if natural_records else 0.0
    lift_values = [
        float(r["readout_routing_correct"]) - float(r["native_correct"]) for r in target_records
    ]
    if args.smoke:
        verdict = "SMOKE_NO_CLAIM"
    elif args.backend == "synthetic":
        verdict = "NO_CLAIM_SYNTHETIC_BACKEND"
    elif not target_records or not fired_target or decoded_correct < 0.80:
        verdict = "READOUT_FAIL"
    elif hard_false_fire > 0.02 or natural_false_fire > 0.01:
        verdict = "ROUTING_UNSAFE"
    elif lift >= 0.15 and native_wrong_lift >= 0.30:
        verdict = "GOAL_B2_DEMO_PASS"
    else:
        verdict = "NO_LIFT"
    next_step = (
        "a claim-bearing full Llama run on the locked DeepMind split"
        if args.backend == "synthetic"
        else "review the records and false-fire strata before making any public claim"
    )
    run_explanation = (
        f"Eval ran {len(records)} locked examples with backend={args.backend}. "
        f"{args.target_op} native exact={native_acc:.3f}, "
        f"routed exact={routed_acc:.3f}, lift={lift:.3f}, "
        f"decoded correctness on fired target={decoded_correct:.3f}, "
        f"hard-negative false-fire={hard_false_fire:.3f}. "
        f"Verdict is {verdict}; next step is {next_step}."
    )
    summary = {
        "backend": args.backend,
        "dataset_source": args.dataset_source,
        "target_op": args.target_op,
        "operand_decode_mode": args.operand_decode_mode,
        "chunk_selection_mode": getattr(args, "chunk_selection_mode", "heuristic"),
        "safe_gate_mode": safe_gate_mode,
        "safe_gate_threshold": safe_gate.threshold if safe_gate is not None else None,
        "model_id": MODEL_ID,
        "smoke": args.smoke,
        "locked_test_sha256": locked_hash(examples),
        "n_locked": len(records),
        "n_target_locked": len(target_records),
        "target_fire_rate": mean_bool(target_records, "fired"),
        "hard_negative_false_fire": hard_false_fire,
        "natural_false_fire": natural_false_fire,
        "pair_exact_on_fired_target": mean_bool(fired_target, "decoded_pair_exact"),
        "decoded_target_correct_on_fired_target": decoded_correct,
        "native_target_exact": native_acc,
        "readout_routing_target_exact": routed_acc,
        "exact_score_lift": lift,
        "exact_score_lift_bootstrap_ci": bootstrap_ci(lift_values, args.seed),
        "native_wrong_routed_exact": native_wrong_lift,
        "records_path": str(records_path),
        "verdict": verdict,
        "run_explanation": run_explanation,
        "limitations": [
            "Synthetic backend validates plumbing only and cannot support a Llama claim"
            if args.backend == "synthetic"
            else (
                "Routing intervention renders the Python answer after internal readout; "
                "residual writes are not the headline claim"
            ),
            "Prompt text and regex are used only in prepare/scoring, not runtime inference",
        ],
    }
    if args.target_op == "lcm":
        summary.update(
            {
                "n_lcm_locked": sum(1 for r in records if r["is_lcm"]),
                "lcm_fire_rate": summary["target_fire_rate"],
                "pair_exact_on_fired_lcm": summary["pair_exact_on_fired_target"],
                "decoded_lcm_correct_on_fired_lcm": decoded_correct,
                "native_lcm_exact": native_acc,
                "readout_routing_lcm_exact": routed_acc,
            }
        )
    (out_dir / f"{stem}{suffix}.json").write_text(json.dumps(summary, indent=2))
    md = [
        f"# Goal B2 {args.target_op} benchmark pipeline",
        "",
        f"- backend: `{args.backend}`",
        f"- dataset source: `{args.dataset_source}`",
        f"- operand decode mode: `{args.operand_decode_mode}`",
        f"- locked examples: {len(records)}",
        f"- locked-test hash: `{summary['locked_test_sha256']}`",
        f"- verdict: **{verdict}**",
        "",
        "## Locked-test metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| {args.target_op} fire rate | {summary['target_fire_rate']:.3f} |",
        f"| hard-negative false-fire | {hard_false_fire:.3f} |",
        f"| natural false-fire | {natural_false_fire:.3f} |",
        f"| pair exact on fired target | {summary['pair_exact_on_fired_target']:.3f} |",
        f"| decoded target correct on fired target | {decoded_correct:.3f} |",
        f"| native target exact | {native_acc:.3f} |",
        f"| readout-routing target exact | {routed_acc:.3f} |",
        f"| exact-score lift | {lift:.3f} |",
        f"| native-wrong routed exact | {native_wrong_lift:.3f} |",
        "",
        "## Run explanation",
        "",
        run_explanation,
        "",
        "## Provenance",
        "",
        "Runtime records use `op_source=activation`, `operand_source=activation`, "
        "and `answer_source=python_from_decoded_tuple` whenever the pipeline fires.",
    ]
    (out_dir / f"{stem}{suffix}.md").write_text("\n".join(md) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["prepare", "fit", "eval", "full"], default="full")
    p.add_argument("--backend", choices=["synthetic", "llama"], default="synthetic")
    p.add_argument(
        "--target_op",
        choices=["lcm", "div_remainder", "mul", "gcd"],
        default=DEFAULT_TARGET_OP,
    )
    p.add_argument(
        "--dataset_source",
        choices=[
            "synthetic",
            "deepmind_interpolate",
            "lcm_chunk_frozen",
            "mul_chunk_frozen",
            "div_remainder_frozen",
        ],
        default="synthetic",
    )
    p.add_argument("--out_dir", default=str(DOCS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_per_family", type=int, default=80)
    p.add_argument("--n_natural", type=int, default=40)
    p.add_argument("--n_adversarial_per_family", type=int, default=250)
    p.add_argument("--train_frac", type=float, default=0.50)
    p.add_argument("--calib_frac", type=float, default=0.25)
    p.add_argument("--dm_scan_limit", type=int, default=10000)
    p.add_argument("--dm_dir", default="")
    p.add_argument("--require_multitoken_answers", action="store_true")
    p.add_argument("--include_common_denominator", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--probes_in", default=str(DOCS / "p1_1_internal_value_probes.pt"))
    p.add_argument(
        "--operand_decode_mode",
        choices=["attention_fourier_l15", "attention_j16_l22_chunk"],
        default=DEFAULT_OPERAND_DECODE_MODE,
    )
    p.add_argument("--chunk_probe_in", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--chunk_top_k", type=int, default=12)
    p.add_argument("--chunk_window", type=int, default=1)
    p.add_argument("--chunk_pos_threshold", type=float, default=0.5)
    p.add_argument("--chunk_value_margin_threshold", type=float, default=0.0)
    p.add_argument(
        "--chunk_selection_mode",
        choices=["heuristic", "learned", "learned_pair", "learned_pair_h"],
        default="heuristic",
    )
    p.add_argument("--chunk_selector_aug_per_family", type=int, default=8)
    p.add_argument("--pair_conf_threshold", type=float, default=0.20)
    p.add_argument("--op_threshold_min", type=float, default=0.5)
    p.add_argument("--op_threshold_neg_margin", type=float, default=1e-4)
    p.add_argument("--safe_gate_mode", choices=["none", "l5_mean"], default="none")
    p.add_argument("--safe_gate_aug_per_family", type=int, default=12)
    p.add_argument("--safe_gate_threshold_min", type=float, default=0.5)
    p.add_argument("--safe_gate_neg_margin", type=float, default=0.02)
    p.add_argument("--operand_lo", type=int, default=0)
    p.add_argument("--operand_hi", type=int, default=999)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    if args.phase in ("prepare", "full"):
        print(f"[goalB2-{args.target_op}] prepare", flush=True)
        phase_prepare(args)
    if args.phase in ("fit", "full"):
        print(f"[goalB2-{args.target_op}] fit", flush=True)
        fit_summary = phase_fit(args)
        print(fit_summary["run_explanation"], flush=True)
    if args.phase in ("eval", "full"):
        print(f"[goalB2-{args.target_op}] eval", flush=True)
        summary = phase_eval(args)
        print(summary["run_explanation"], flush=True)
        print(
            json.dumps(
                {"verdict": summary["verdict"], "wall_s": round(time.perf_counter() - t0, 2)}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
