#!/usr/bin/env python3
"""Goal B3 strict non-Llama transfer attempt.

This is a deliberately hard cross-model test for the current Goal B2/B3
runtime contract:

opaque prompt ids + captured activations
  -> activation op gate
  -> answer-site activation operand decoder
  -> Python op(decoded_a, decoded_b)
  -> exact scoring

No prompt text, regex, decoded token spans, harness operands, or gold labels are
available to the runtime path. Prompt text and metadata are used only in
prepare/fit labels and grading.

The claim-bearing backend is Qwen/Qwen2.5-7B using the existing Qwen P1.3
internal value probes. The synthetic backend exists only for tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

from goalB2_lcm_benchmark_pipeline import (  # noqa: E402
    ProvenanceGuard,
    RuntimeInputs,
    bootstrap_ci,
    compute_target,
    mean_bool,
)
from p1_1_internal_value_decoder import load_probe_bank  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B"
DTYPE = torch.bfloat16
OPS = ("mul", "div_remainder", "lcm", "gcd")
OP_TO_ID = {op: i for i, op in enumerate(OPS)}
ID_TO_OP = {i: op for op, i in OP_TO_ID.items()}
ANSWER_SUFFIX = " Answer: "
FEATURE_LAYERS = (8, 10, 12, 15)
OP_LAYER = 10
OPERAND_LAYER = 8


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    family: str
    op: str
    is_target: int
    a: int
    b: int
    answer: int | None
    prompt: str
    token_ids: list[int]
    source: str


@dataclass
class OpReadout:
    scaler: StandardScaler
    clf: LogisticRegression
    threshold: float


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def target_answer(op: str, a: int, b: int) -> int:
    if op == "gcd":
        return gcd(a, b)
    return compute_target(op, a, b)


def stable_hash(rows: list[Example]) -> str:
    h = hashlib.sha256()
    for ex in rows:
        h.update(
            json.dumps(
                {
                    "id": ex.example_id,
                    "split": ex.split,
                    "family": ex.family,
                    "op": ex.op,
                    "token_ids": ex.token_ids,
                },
                sort_keys=True,
            ).encode()
        )
        h.update(b"\n")
    return h.hexdigest()


def encode_prompt(prompt: str, backend: str, tokenizer: Any | None) -> list[int]:
    if backend == "synthetic":
        return [ord(c) % 251 for c in prompt]
    return tokenizer.encode(prompt, add_special_tokens=False)


TARGET_TEMPLATES = {
    "mul": [
        "What is {a} times {b}?",
        "Compute the product of {a} and {b}.",
        "Find {a} * {b}.",
        "A grid has {a} rows and {b} columns; how many cells?",
    ],
    "div_remainder": [
        "What is the remainder when {a} is divided by {b}?",
        "Compute {a} modulo {b}.",
        "Find {a} mod {b}.",
        "After making groups of {b} from {a} items, how many are left?",
    ],
    "lcm": [
        "What is the least common multiple of {a} and {b}?",
        "Find the LCM of {a} and {b}.",
        "Compute the smallest common multiple of {a} and {b}.",
        "What denominator is shared by {a} and {b} if using the smallest common multiple?",
    ],
    "gcd": [
        "What is the greatest common divisor of {a} and {b}?",
        "Find the GCD of {a} and {b}.",
        "Compute the highest common factor of {a} and {b}.",
        "What is the largest integer factor shared by {a} and {b}?",
    ],
}

NEGATIVE_TEMPLATES = [
    ("quoted_expression_negative", "The note quotes '{a} times {b}' but asks for no calculation."),
    ("do_not_compute_negative", "Do not compute anything; just remember numbers {a} and {b}."),
    ("wrong_op_negative", "What is {a} plus {b}?"),
    ("natural_numeric_negative", "Room {a} contains shelf {b} and a blue label."),
]


def build_rows(args: argparse.Namespace, tokenizer: Any | None) -> list[Example]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for op in args.ops:
        templates = TARGET_TEMPLATES[op]
        for family_idx, template in enumerate(templates):
            for _ in range(args.n_per_family):
                if op == "lcm":
                    a = int(rng.integers(2, args.lcm_operand_hi + 1))
                    b = int(rng.integers(2, args.lcm_operand_hi + 1))
                elif op == "div_remainder":
                    a = int(rng.integers(20, args.operand_hi + 1))
                    b = int(rng.integers(2, min(args.operand_hi, a) + 1))
                else:
                    a = int(rng.integers(2, args.operand_hi + 1))
                    b = int(rng.integers(2, args.operand_hi + 1))
                rows.append(
                    {
                        "family": f"{op}_target_{family_idx}",
                        "op": op,
                        "is_target": 1,
                        "a": a,
                        "b": b,
                        "answer": target_answer(op, a, b),
                        "prompt": template.format(a=a, b=b) + ANSWER_SUFFIX,
                        "source": "goalB3_frozen_synthetic_templates",
                    }
                )
    for family, template in NEGATIVE_TEMPLATES:
        for _ in range(args.n_negative_per_family):
            a = int(rng.integers(2, args.operand_hi + 1))
            b = int(rng.integers(2, args.operand_hi + 1))
            rows.append(
                {
                    "family": family,
                    "op": "negative",
                    "is_target": 0,
                    "a": a,
                    "b": b,
                    "answer": None,
                    "prompt": template.format(a=a, b=b) + ANSWER_SUFFIX,
                    "source": "goalB3_negative_control",
                }
            )
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * args.train_frac)
    n_calib = int(n * args.calib_frac)
    examples: list[Example] = []
    for idx, row in enumerate(rows):
        split = "train" if idx < n_train else ("calib" if idx < n_train + n_calib else "locked_test")
        token_ids = encode_prompt(row["prompt"], args.backend, tokenizer)
        ex_id = hashlib.sha256(f"{args.seed}:{idx}:{row['prompt']}".encode()).hexdigest()[:16]
        examples.append(
            Example(
                example_id=ex_id,
                split=split,
                family=row["family"],
                op=row["op"],
                is_target=int(row["is_target"]),
                a=int(row["a"]),
                b=int(row["b"]),
                answer=row["answer"],
                prompt=row["prompt"],
                token_ids=token_ids,
                source=row["source"],
            )
        )
    return examples


class QwenAnswerSiteCapture:
    def __init__(self, model_id: str = MODEL_ID):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        if self.device.type != "cpu":
            try:
                self.model.to(self.device)
            except RuntimeError as exc:
                print(f"[goalB3-qwen] device fallback to CPU: {exc}", flush=True)
                self.device = torch.device("cpu")
        self.model.eval()
        self.layers = self.model.model.layers

    def capture(self, token_ids: list[int]) -> dict[str, np.ndarray]:
        ids_t = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        captured: dict[int, list[torch.Tensor | None]] = {L: [None] for L in FEATURE_LAYERS}

        def make_hook(layer: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                if isinstance(hs, torch.Tensor):
                    captured[layer][0] = hs[0, -1, :].clone().detach().to("cpu", torch.float32)
            return hook

        handles = []
        for layer in FEATURE_LAYERS:
            if layer < len(self.layers):
                handles.append(self.layers[layer].register_forward_hook(make_hook(layer)))
        try:
            with torch.no_grad():
                _ = self.model(input_ids=ids_t, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return {
            f"answer_site_L{layer}": value[0].numpy()
            for layer, value in captured.items()
            if value[0] is not None
        }


def synthetic_activations(ex: Example, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(ex.example_id[:8], 16) ^ seed)
    dim = 64
    base = rng.normal(0, 0.05, size=dim).astype(np.float32)
    if ex.is_target:
        op_id = OP_TO_ID[ex.op]
        base[op_id] += 4.0
        base[8] = ex.a / 1000.0
        base[9] = ex.b / 1000.0
    else:
        base[6] += 2.0
    return {f"answer_site_L{layer}": base.copy() for layer in FEATURE_LAYERS}


def capture_examples(
    examples: list[Example],
    backend: str,
    seed: int,
    qwen: QwenAnswerSiteCapture | None,
) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for ex in examples:
        if backend == "synthetic":
            out[ex.example_id] = synthetic_activations(ex, seed)
        else:
            assert qwen is not None
            out[ex.example_id] = qwen.capture(ex.token_ids)
    return out


def fit_op_readout(examples: list[Example], activations: dict[str, dict[str, np.ndarray]], args: argparse.Namespace) -> OpReadout:
    train = [e for e in examples if e.split == "train"]
    calib = [e for e in examples if e.split == "calib"]
    X = np.stack([activations[e.example_id][f"answer_site_L{OP_LAYER}"] for e in train])
    y = np.array([OP_TO_ID[e.op] if e.is_target else len(OPS) for e in train])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(Xs, y)
    threshold = args.op_threshold_min
    if calib:
        Xc = scaler.transform(
            np.stack([activations[e.example_id][f"answer_site_L{OP_LAYER}"] for e in calib])
        )
        proba = clf.predict_proba(Xc)
        target_scores = []
        neg_scores = []
        for ex, p in zip(calib, proba, strict=True):
            score = float(max(p[: len(OPS)]))
            if ex.is_target:
                target_scores.append(score)
            else:
                neg_scores.append(score)
        if neg_scores:
            threshold = max(threshold, max(neg_scores) + args.op_neg_margin)
        if target_scores:
            threshold = min(threshold, max(target_scores))
    return OpReadout(scaler=scaler, clf=clf, threshold=float(threshold))


def decode_operands_synthetic(runtime: RuntimeInputs) -> tuple[int, int, float]:
    h = runtime.activations[f"answer_site_L{OPERAND_LAYER}"]
    return int(round(float(h[8]) * 1000)), int(round(float(h[9]) * 1000)), 1.0


def decode_operands_qwen(runtime: RuntimeInputs, probe_bank: Any, args: argparse.Namespace) -> tuple[int, int, float]:
    h_np = runtime.activations[f"answer_site_L{OPERAND_LAYER}"]
    h = torch.tensor(h_np, dtype=torch.float32)
    pa = probe_bank.fourier.get(("a", OPERAND_LAYER))
    pb = probe_bank.fourier.get(("b", OPERAND_LAYER))
    if pa is None or pb is None:
        raise RuntimeError(f"Qwen probe bank lacks Fourier probes for L{OPERAND_LAYER}")
    a = int(pa.decode_codebook(h, lo=args.operand_lo, hi=args.operand_hi))
    b = int(pb.decode_codebook(h, lo=args.operand_lo, hi=args.operand_hi))
    return a, b, 0.0


def run_runtime(
    ex: Example,
    activations: dict[str, np.ndarray],
    op_readout: OpReadout,
    guard: ProvenanceGuard,
    args: argparse.Namespace,
    probe_bank: Any | None,
) -> dict[str, Any]:
    runtime = RuntimeInputs(
        example_id=ex.example_id,
        prompt_ids=tuple(ex.token_ids),
        activations=activations,
    )
    guard.assert_runtime_inputs(runtime)
    guard.reject_forbidden(
        prompt_text=None,
        regex_matches=None,
        tokenizer_decoded_spans=None,
        cli_op=None,
        harness_operands=None,
        gold_answer=None,
        gold_label=None,
    )
    x = op_readout.scaler.transform(
        np.expand_dims(runtime.activations[f"answer_site_L{OP_LAYER}"], axis=0)
    )
    p = op_readout.clf.predict_proba(x)[0]
    best_target_id = int(np.argmax(p[: len(OPS)]))
    op_score = float(p[best_target_id])
    if op_score < op_readout.threshold:
        return {
            "fired": False,
            "abstain_reason": "op_below_threshold",
            "op_score": op_score,
            "op_threshold": op_readout.threshold,
            "op_source": "activation",
            "operand_source": "activation",
            "answer_source": None,
        }
    op = ID_TO_OP[best_target_id]
    if args.backend == "synthetic":
        a, b, pair_conf = decode_operands_synthetic(runtime)
    else:
        a, b, pair_conf = decode_operands_qwen(runtime, probe_bank, args)
    if b == 0 and op == "div_remainder":
        return {
            "fired": False,
            "abstain_reason": "decoded_divisor_zero",
            "op_score": op_score,
            "op_threshold": op_readout.threshold,
            "op_source": "activation",
            "operand_source": "activation",
            "answer_source": None,
        }
    answer = target_answer(op, a, b)
    return {
        "fired": True,
        "decoded_op": op,
        "decoded_a": a,
        "decoded_b": b,
        "decoded_answer": answer,
        "op_score": op_score,
        "op_threshold": op_readout.threshold,
        "pair_confidence": pair_conf,
        "op_source": "activation",
        "operand_source": "activation",
        "answer_source": "python_from_decoded_tuple",
    }


def score_record(ex: Example, pipe: dict[str, Any]) -> dict[str, Any]:
    fired = bool(pipe.get("fired"))
    decoded_op = pipe.get("decoded_op")
    decoded_a = pipe.get("decoded_a")
    decoded_b = pipe.get("decoded_b")
    decoded_pair_exact = fired and decoded_a == ex.a and decoded_b == ex.b
    decoded_op_exact = fired and decoded_op == ex.op
    routed_correct = fired and ex.is_target and pipe.get("decoded_answer") == ex.answer
    return {
        "example_id": ex.example_id,
        "split": ex.split,
        "family": ex.family,
        "op": ex.op,
        "is_target": bool(ex.is_target),
        "fired": fired,
        "decoded_op": decoded_op,
        "decoded_a": decoded_a,
        "decoded_b": decoded_b,
        "decoded_op_exact": bool(decoded_op_exact),
        "decoded_pair_exact": bool(decoded_pair_exact),
        "readout_routing_correct": bool(routed_correct),
        "gold_answer_for_grading_only": ex.answer,
        "provenance": {
            "op_source": pipe.get("op_source"),
            "operand_source": pipe.get("operand_source"),
            "answer_source": pipe.get("answer_source"),
        },
        "diagnostics": {
            "op_score": pipe.get("op_score"),
            "op_threshold": pipe.get("op_threshold"),
            "abstain_reason": pipe.get("abstain_reason"),
            "pair_confidence": pipe.get("pair_confidence"),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def phase_prepare(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = None
    if args.backend == "qwen":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    rows = build_rows(args, tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_path = out_dir / f"{args.output_stem}_splits.jsonl"
    write_jsonl(splits_path, [asdict(e) for e in rows])
    manifest = {
        "experiment": "goalB3_qwen_strict_transfer",
        "backend": args.backend,
        "model_id": args.model_id,
        "ops": args.ops,
        "seed": args.seed,
        "n_examples": len(rows),
        "n_locked": sum(e.split == "locked_test" for e in rows),
        "locked_hash": stable_hash([e for e in rows if e.split == "locked_test"]),
        "runtime_contract": {
            "prompt": "opaque_token_ids",
            "op_source": "activation",
            "operand_source": "answer_site_activation_probe",
            "answer_source": "python_from_decoded_tuple",
        },
        "splits_path": str(splits_path),
    }
    manifest_path = out_dir / f"{args.output_stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def read_examples(path: Path) -> list[Example]:
    return [Example(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def run_full(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / f"{args.output_stem}_manifest.json"
    splits_path = out_dir / f"{args.output_stem}_splits.jsonl"
    if not splits_path.exists() or args.force_prepare:
        manifest = phase_prepare(args)
    else:
        manifest = json.loads(manifest_path.read_text())
    examples = read_examples(splits_path)

    qwen = None
    probe_bank = None
    if args.backend == "qwen":
        qwen = QwenAnswerSiteCapture(args.model_id)
        probe_bank = load_probe_bank(Path(args.probes_in))
    activations = capture_examples(examples, args.backend, args.seed, qwen)
    op_readout = fit_op_readout(examples, activations, args)
    guard = ProvenanceGuard(runtime_mode=True, allowed_op="multi")
    records = []
    for ex in examples:
        if ex.split != "locked_test":
            continue
        pipe = run_runtime(ex, activations[ex.example_id], op_readout, guard, args, probe_bank)
        records.append(score_record(ex, pipe))
    records_path = out_dir / f"{args.output_stem}_records.jsonl"
    write_jsonl(records_path, records)

    targets = [r for r in records if r["is_target"]]
    negatives = [r for r in records if not r["is_target"]]
    fired_targets = [r for r in targets if r["fired"]]
    exact_values = [float(r["readout_routing_correct"]) for r in targets]
    by_op = {}
    for op in args.ops:
        op_rows = [r for r in targets if r["op"] == op]
        op_fired = [r for r in op_rows if r["fired"]]
        by_op[op] = {
            "n": len(op_rows),
            "fire_rate": mean_bool(op_rows, "fired"),
            "op_exact_on_fired": mean_bool(op_fired, "decoded_op_exact"),
            "pair_exact_on_fired": mean_bool(op_fired, "decoded_pair_exact"),
            "routed_exact": mean_bool(op_rows, "readout_routing_correct"),
        }
    target_exact = mean_bool(targets, "readout_routing_correct")
    false_fire = mean_bool(negatives, "fired")
    pair_exact = mean_bool(fired_targets, "decoded_pair_exact")
    if args.backend == "synthetic":
        verdict = "SMOKE_NO_CLAIM" if args.smoke else "NO_CLAIM_SYNTHETIC_BACKEND"
    elif target_exact >= args.min_exact_gate and false_fire <= args.max_false_fire_gate and pair_exact >= args.min_pair_exact_gate:
        verdict = "STRICT_CROSS_MODEL_PASS"
    elif false_fire > args.max_false_fire_gate:
        verdict = "STRICT_CROSS_MODEL_UNSAFE_FALSE_FIRE"
    else:
        verdict = "STRICT_CROSS_MODEL_READOUT_FAIL"
    summary = {
        "experiment": "goalB3_qwen_strict_transfer",
        "backend": args.backend,
        "model_id": args.model_id,
        "smoke": args.smoke,
        "ops": args.ops,
        "seed": args.seed,
        "locked_hash": manifest["locked_hash"],
        "n_locked": len(records),
        "n_target_locked": len(targets),
        "n_negative_locked": len(negatives),
        "target_fire_rate": mean_bool(targets, "fired"),
        "target_exact": target_exact,
        "target_exact_bootstrap_ci": bootstrap_ci(exact_values, args.seed),
        "pair_exact_on_fired_target": pair_exact,
        "hard_negative_false_fire": false_fire,
        "by_op": by_op,
        "records_path": str(records_path),
        "verdict": verdict,
        "limitations": [
            "This is a strict non-Llama transfer attempt using answer-site activation operands.",
            "No prompt text, regex, token spans, harness operands, or gold answers are available to runtime.",
            "A READOUT_FAIL verdict falsifies this strict transfer path, not all possible Qwen mechanisms.",
        ],
    }
    out_json = out_dir / f"{args.output_stem}.json"
    out_md = out_dir / f"{args.output_stem}.md"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_md(out_md, summary)
    return summary


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Qwen Strict Transfer",
        "",
        f"- backend: `{summary['backend']}`",
        f"- model: `{summary['model_id']}`",
        f"- verdict: **{summary['verdict']}**",
        f"- locked hash: `{summary['locked_hash']}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| locked examples | {summary['n_locked']} |",
        f"| target locked | {summary['n_target_locked']} |",
        f"| negative locked | {summary['n_negative_locked']} |",
        f"| target fire rate | {summary['target_fire_rate']:.3f} |",
        f"| target exact | {summary['target_exact']:.3f} |",
        f"| pair exact on fired target | {summary['pair_exact_on_fired_target']:.3f} |",
        f"| hard-negative false-fire | {summary['hard_negative_false_fire']:.3f} |",
        "",
        "## Per Op",
        "",
        "| op | n | fire | op exact fired | pair exact fired | routed exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for op, row in summary["by_op"].items():
        lines.append(
            f"| {op} | {row['n']} | {row['fire_rate']:.3f} | "
            f"{row['op_exact_on_fired']:.3f} | {row['pair_exact_on_fired']:.3f} | "
            f"{row['routed_exact']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Runtime records fire only with `op_source=activation`, "
            "`operand_source=activation`, and "
            "`answer_source=python_from_decoded_tuple`.",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["synthetic", "qwen"], default="synthetic")
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--ops", nargs="+", choices=list(OPS), default=["mul", "div_remainder", "lcm", "gcd"])
    p.add_argument("--seed", type=int, default=703)
    p.add_argument("--n_per_family", type=int, default=20)
    p.add_argument("--n_negative_per_family", type=int, default=80)
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--calib_frac", type=float, default=0.25)
    p.add_argument("--operand_lo", type=int, default=0)
    p.add_argument("--operand_hi", type=int, default=999)
    p.add_argument("--lcm_operand_hi", type=int, default=80)
    p.add_argument("--op_threshold_min", type=float, default=0.65)
    p.add_argument("--op_neg_margin", type=float, default=0.05)
    p.add_argument("--min_exact_gate", type=float, default=0.20)
    p.add_argument("--min_pair_exact_gate", type=float, default=0.80)
    p.add_argument("--max_false_fire_gate", type=float, default=0.01)
    p.add_argument("--probes_in", default=str(DOCS / "p1_3_qwen_internal_value_probes.pt"))
    p.add_argument("--out_dir", default=str(DOCS))
    p.add_argument("--output_stem", default="goalB3_qwen_strict_transfer")
    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_per_family = min(args.n_per_family, 3)
        args.n_negative_per_family = min(args.n_negative_per_family, 8)
        args.output_stem += "_smoke"
    t0 = time.perf_counter()
    summary = run_full(args)
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "target_exact": summary["target_exact"],
                "hard_negative_false_fire": summary["hard_negative_false_fire"],
                "wall_s": round(time.perf_counter() - t0, 2),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
