#!/usr/bin/env python3
"""Goal B3 scaled causal interchange on the repaired operand route.

For matched fired target examples, patch donor selected L22 operand chunks into
recipient selected operand chunk positions. Then test:

- decoder follows donor operands;
- routed Python answer follows donor tuple;
- random-position patch control does not follow donor.

Runtime and patched decoding use opaque prompt IDs plus captured activations.
Prompt metadata is used only to reconstruct held-out examples and grade
donor-follow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
sys.path.insert(0, str(REPO / "scripts"))

import goalB2_lcm_benchmark_pipeline as b2  # noqa: E402
from goalB2_neurips_adversarial_audit import OP_CONFIG, build_examples, load_runtime_components  # noqa: E402
from goalB2_chunk_patch_validation import first_two_groups, group_positions  # noqa: E402
from goalB3_gate_robustness_repair import fit_safe_gate  # noqa: E402
from goalB3_op_gate_repair import fit_op_readout  # noqa: E402
from goalB3_operand_repair import fit_selector, parse_pair_thresholds  # noqa: E402


def log(msg: str) -> None:
    print(f"[goalB3 causal] {msg}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_groups(diag: dict[str, Any]) -> list[Any]:
    groups = diag.get("selected_groups")
    if groups is None:
        groups = diag.get("chunk_pair_selector", {}).get("selected_groups")
    if groups is None:
        groups = diag.get("chunk_groups")
    if groups is None:
        groups = diag.get("groups")
    return groups if isinstance(groups, list) else []


def candidate_ids_from_records(paths: list[str], op: str) -> set[str]:
    ids: set[str] = set()
    for raw_path in paths:
        for rec in load_jsonl(Path(raw_path)):
            diag = rec.get("readout_diagnostics")
            if (
                (rec.get("target_op") == op or rec.get("op") == op)
                and rec.get("is_target")
                and rec.get("fired")
                and rec.get("decoded_pair_exact")
                and isinstance(diag, dict)
                and len(selected_groups(diag)) >= 2
            ):
                ids.add(str(rec["example_id"]))
    return ids


def build_eval_examples(op: str, args: argparse.Namespace, tok: Any) -> list[b2.Example]:
    if args.split_source == "deepmind_interpolate":
        ns = SimpleNamespace(
            seed=args.eval_seed,
            target_op=op,
            n_per_family=args.deepmind_n_per_family,
            n_natural=args.deepmind_n_natural,
            train_frac=0.40,
            calib_frac=0.20,
            dm_scan_limit=args.dm_scan_limit,
            dm_dir=args.dm_dir,
            include_common_denominator=args.include_common_denominator,
            require_multitoken_answers=False,
            operand_lo=0,
            operand_hi=9999,
        )
        return [ex for ex in b2.build_deepmind_examples(ns, "llama") if ex.split == "locked_test"]
    return build_examples(op, args.eval_seed, args.eval_n_per_family, tok)


def dummy_base_readouts(pair_threshold: float) -> b2.Readouts:
    return b2.Readouts(
        op_w=np.zeros(b2.ACT_DIM, dtype=np.float32),
        op_b=0.0,
        op_threshold=0.5,
        operand_W=np.zeros((2, b2.ACT_DIM), dtype=np.float32),
        operand_b=np.zeros(2, dtype=np.float32),
        operand_rmse=1.0,
        pair_conf_threshold=pair_threshold,
    )


def load_base_components_or_dummy(
    base: Path, op: str, artifact_seed: int, pair_threshold: float
) -> tuple[b2.Readouts, b2.ChunkPairSelector | None]:
    try:
        base_readouts, _old_safe_gate, base_selector = load_runtime_components(base, op, artifact_seed)
        return base_readouts, base_selector
    except FileNotFoundError as exc:
        log(f"{op}: base artifact missing ({exc}); fitting from dummy readouts")
        return dummy_base_readouts(pair_threshold), None


def donor_match(op: str, decoded: Any, donor_a: int, donor_b: int) -> bool:
    if decoded is None:
        return False
    if op == "div_remainder":
        return int(decoded.a) == donor_a and int(decoded.b) == donor_b
    return {int(decoded.a), int(decoded.b)} == {donor_a, donor_b}


def patch_groups(
    rec_rt: b2.RuntimeInputs,
    donor_rt: b2.RuntimeInputs,
    rec_groups: list[dict[str, Any]],
    donor_groups: list[dict[str, Any]],
) -> tuple[b2.RuntimeInputs | None, list[dict[str, int]]]:
    patched_acts = dict(rec_rt.activations)
    H = np.array(patched_acts["all_positions_L22"], copy=True)
    patch_pairs = []
    for rec_group, donor_group in zip(rec_groups, donor_groups, strict=True):
        rec_positions = group_positions(rec_group)
        donor_positions = group_positions(donor_group)
        if len(rec_positions) != len(donor_positions):
            return None, []
        for rec_pos, donor_pos in zip(rec_positions, donor_positions, strict=True):
            H[rec_pos] = donor_rt.activations["all_positions_L22"][donor_pos]
            patch_pairs.append({"recipient_pos": int(rec_pos), "donor_pos": int(donor_pos)})
    patched_acts["all_positions_L22"] = H
    return b2.RuntimeInputs(rec_rt.example_id, rec_rt.prompt_ids, patched_acts), patch_pairs


def patch_random_positions(
    rec_rt: b2.RuntimeInputs,
    donor_rt: b2.RuntimeInputs,
    rec_groups: list[dict[str, Any]],
    donor_groups: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[b2.RuntimeInputs | None, list[dict[str, int]]]:
    donor_selected = {p for group in donor_groups for p in group_positions(group)}
    donor_H = donor_rt.activations["all_positions_L22"]
    candidates = [p for p in range(1, donor_H.shape[0]) if p not in donor_selected]
    patched_acts = dict(rec_rt.activations)
    H = np.array(patched_acts["all_positions_L22"], copy=True)
    patch_pairs = []
    for rec_group in rec_groups:
        for rec_pos in group_positions(rec_group):
            if not candidates:
                return None, []
            donor_pos = int(rng.choice(candidates))
            H[rec_pos] = donor_H[donor_pos]
            patch_pairs.append({"recipient_pos": int(rec_pos), "donor_pos": donor_pos})
    patched_acts["all_positions_L22"] = H
    return b2.RuntimeInputs(rec_rt.example_id, rec_rt.prompt_ids, patched_acts), patch_pairs


def decode_patched(
    runtime: b2.RuntimeInputs,
    *,
    op: str,
    readouts: b2.Readouts,
    selector: b2.ChunkPairSelector,
    chunk_probe: b2.J16ChunkProbe,
) -> tuple[Any, dict[str, Any], int | None]:
    guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=op)
    decoded, diag = b2.decode_from_activations(
        runtime,
        readouts,
        guard,
        backend="llama",
        target_op=op,
        operand_lo=0,
        operand_hi=9999,
        operand_decode_mode="attention_j16_l22_chunk",
        chunk_probe=chunk_probe,
        chunk_pair_selector=selector,
    )
    answer = None if decoded is None else guard.calculator(decoded)
    return decoded, diag, answer


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for op in sorted({r["target_op"] for r in rows}):
        rs = [r for r in rows if r["target_op"] == op]
        out[op] = {
            "n_pairs": len(rs),
            "decoder_follow_donor_rate": float(np.mean([r["decoder_followed_donor"] for r in rs])) if rs else float("nan"),
            "routed_answer_follow_donor_rate": float(np.mean([r["routed_answer_followed_donor"] for r in rs])) if rs else float("nan"),
            "random_decoder_follow_donor_rate": float(np.mean([r["random_decoder_followed_donor"] for r in rs])) if rs else float("nan"),
            "random_routed_answer_follow_donor_rate": float(np.mean([r["random_routed_answer_followed_donor"] for r in rs])) if rs else float("nan"),
        }
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Goal B3 Causal Interchange",
        "",
        f"- verdict: **{payload['verdict']}**",
        "- scope: per-run donor-follow/control rate verdict only; final powered claims require cross-seed pair-count aggregation",
        f"- fit seed: `{payload['fit_seed']}`",
        f"- eval seed: `{payload['eval_seed']}`",
        "",
        "| op | pairs | decoder donor-follow | routed donor-follow | random decoder follow | random routed follow |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for op, row in payload["ops"].items():
        lines.append(
            f"| `{op}` | {row['n_pairs']} | {row['decoder_follow_donor_rate']:.3f} | "
            f"{row['routed_answer_follow_donor_rate']:.3f} | "
            f"{row['random_decoder_follow_donor_rate']:.3f} | "
            f"{row['random_routed_answer_follow_donor_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "This patches only selected L22 operand chunk activations. Op/safe gate "
            "features remain recipient activations; Python receives only decoded "
            "activation-derived operands after patching.",
            "",
            "Do not interpret this per-run verdict as a powered causal gate pass unless",
            "a cross-seed summary also satisfies the frozen pair-count gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def verdict(summary: dict[str, Any], min_follow: float, max_random_follow: float) -> str:
    if not summary:
        return "CAUSAL_INTERCHANGE_NO_PAIRS"
    if all(
        row["decoder_follow_donor_rate"] >= min_follow
        and row["routed_answer_follow_donor_rate"] >= min_follow
        and row["random_routed_answer_follow_donor_rate"] <= max_random_follow
        for row in summary.values()
    ):
        return "CAUSAL_INTERCHANGE_PASS"
    return "CAUSAL_INTERCHANGE_PARTIAL_OR_FAIL"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", default="/tmp/rune_goalB2_neurips_suite_full")
    p.add_argument("--ops", nargs="+", choices=sorted(OP_CONFIG), default=["mul", "div_remainder", "lcm"])
    p.add_argument("--artifact_seed", type=int, default=632)
    p.add_argument("--fit_seed", type=int, default=701)
    p.add_argument("--eval_seed", type=int, default=702)
    p.add_argument("--fit_n_per_family", type=int, default=30)
    p.add_argument("--eval_n_per_family", type=int, default=20)
    p.add_argument("--split_source", choices=["synthetic_adversarial", "deepmind_interpolate"], default="synthetic_adversarial")
    p.add_argument("--candidate_records", nargs="*", default=[])
    p.add_argument("--deepmind_n_per_family", type=int, default=500)
    p.add_argument("--deepmind_n_natural", type=int, default=200)
    p.add_argument("--dm_scan_limit", type=int, default=200000)
    p.add_argument("--dm_dir", default="")
    p.add_argument("--include_common_denominator", action="store_true")
    p.add_argument("--max_eval_examples_per_op", type=int, default=0)
    p.add_argument("--n_pairs_per_op", type=int, default=20)
    p.add_argument("--seed", type=int, default=711)
    p.add_argument("--pair_threshold", action="append", default=["mul=0.05", "div_remainder=0.20", "lcm=0.20", "gcd=0.20"])
    p.add_argument("--chunk_probe", default=str(DOCS / "j16_multitoken_operand_probe.pt"))
    p.add_argument("--min_follow", type=float, default=1.0)
    p.add_argument("--max_random_follow", type=float, default=0.10)
    p.add_argument("--out_json", default=str(DOCS / "goalB3_causal_interchange.json"))
    p.add_argument("--out_md", default=str(DOCS / "goalB3_causal_interchange.md"))
    p.add_argument("--out_records", default="")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    base = Path(args.work_dir)
    pair_thresholds = parse_pair_thresholds(args.pair_threshold)
    model, tok, device = b2.load_llama()
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    rng = np.random.default_rng(args.seed)
    rows = []
    for op in args.ops:
        log(f"{op}: loading frozen base components")
        pair_threshold = pair_thresholds.get(op, float(OP_CONFIG[op]["pair_conf"]))
        base_readouts, base_selector = load_base_components_or_dummy(base, op, args.artifact_seed, pair_threshold)
        log(f"{op}: building {args.fit_n_per_family}/family fit examples")
        fit_examples = build_examples(op, args.fit_seed, args.fit_n_per_family, tok)
        log(f"{op}: fitting repaired op readout")
        readouts, _op_summary = fit_op_readout(
            base_readouts,
            fit_examples,
            op=op,
            seed=args.fit_seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=0.65,
            neg_margin=0.05,
            calib_frac=0.25,
        )
        readouts.pair_conf_threshold = pair_threshold
        log(f"{op}: fitting repaired safe gate")
        safe_gate, _safe_summary = fit_safe_gate(
            fit_examples,
            op=op,
            seed=args.fit_seed,
            backend="llama",
            model=model,
            device=device,
            threshold_min=0.65,
            neg_margin=0.05,
            calib_frac=0.25,
        )
        log(f"{op}: fitting operand chunk selector")
        selector, _selector_summary = fit_selector(
            fit_examples,
            op=op,
            seed=args.fit_seed,
            model=model,
            device=device,
            chunk_probe=chunk_probe,
            include_embeddings=True,
            chunk_top_k=12,
            chunk_window=1,
            chunk_pos_threshold=0.5,
            chunk_value_margin_threshold=0.0,
        )
        selector = selector if selector is not None else base_selector
        log(f"{op}: building eval examples from {args.split_source}")
        eval_examples_list = build_eval_examples(op, args, tok)
        if args.max_eval_examples_per_op > 0:
            eval_examples_list = eval_examples_list[: args.max_eval_examples_per_op]
        eval_examples = {ex.example_id: ex for ex in eval_examples_list}
        guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=op)
        runtime_cache: dict[str, b2.RuntimeInputs] = {}
        op_records = []
        log(f"{op}: scoring {len(eval_examples_list)} opaque eval examples")
        for idx, ex in enumerate(eval_examples_list, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(eval_examples_list):
                log(f"{op}: scored {idx - 1}/{len(eval_examples_list)} examples")
            runtime = b2.runtime_from_example(ex, args.eval_seed, "llama", model, device)
            runtime_cache[ex.example_id] = runtime
            pipe = b2.run_opaque_pipeline(
                runtime,
                readouts,
                guard,
                backend="llama",
                target_op=op,
                operand_lo=0,
                operand_hi=9999,
                operand_decode_mode="attention_j16_l22_chunk",
                chunk_probe=chunk_probe,
                safe_gate=safe_gate,
                chunk_pair_selector=selector,
            )
            rec = b2.score_eval_record(ex, pipe, None)
            rec["target_op"] = op
            op_records.append(rec)
        log(f"{op}: scored {len(eval_examples_list)}/{len(eval_examples_list)} examples")
        op_records = [
            r
            for r in op_records
            if r.get("is_target")
            and r.get("fired")
            and r.get("decoded_pair_exact")
            and len(first_two_groups(r.get("readout_diagnostics") or {})) >= 2
        ]
        if args.candidate_records:
            allowed = candidate_ids_from_records(args.candidate_records, op)
            op_records = [r for r in op_records if str(r["example_id"]) in allowed]
        log(f"{op}: found {len(op_records)} fired exact target candidates")
        rng.shuffle(op_records)
        pairs = list(zip(op_records[::2], op_records[1::2], strict=False))[: args.n_pairs_per_op]
        log(f"{op}: patching {len(pairs)} donor-recipient pairs")
        for rec_rec, donor_rec in pairs:
            rec_ex = eval_examples[rec_rec["example_id"]]
            donor_ex = eval_examples[donor_rec["example_id"]]
            rec_rt = runtime_cache[rec_ex.example_id]
            donor_rt = runtime_cache[donor_ex.example_id]
            rec_groups = first_two_groups(rec_rec["readout_diagnostics"])
            donor_groups = first_two_groups(donor_rec["readout_diagnostics"])
            patched_rt, patch_pairs = patch_groups(rec_rt, donor_rt, rec_groups, donor_groups)
            random_rt, random_pairs = patch_random_positions(rec_rt, donor_rt, rec_groups, donor_groups, rng)
            if patched_rt is None or random_rt is None:
                continue
            decoded, _diag, answer = decode_patched(
                patched_rt,
                op=op,
                readouts=readouts,
                selector=selector,
                chunk_probe=chunk_probe,
            )
            rand_decoded, _rand_diag, rand_answer = decode_patched(
                random_rt,
                op=op,
                readouts=readouts,
                selector=selector,
                chunk_probe=chunk_probe,
            )
            donor_a = int(donor_rec["decoded_a"])
            donor_b = int(donor_rec["decoded_b"])
            donor_answer = b2.compute_target(op, donor_a, donor_b)
            rows.append(
                {
                    "target_op": op,
                    "recipient": rec_rec["example_id"],
                    "donor": donor_rec["example_id"],
                    "patch_pairs": patch_pairs,
                    "random_patch_pairs": random_pairs,
                    "donor_decoded": {"a": donor_a, "b": donor_b},
                    "patched_decoded": None if decoded is None else {"a": decoded.a, "b": decoded.b},
                    "random_decoded": None if rand_decoded is None else {"a": rand_decoded.a, "b": rand_decoded.b},
                    "donor_answer": donor_answer,
                    "routed_answer_after_patch": answer,
                    "random_routed_answer_after_patch": rand_answer,
                    "decoder_followed_donor": donor_match(op, decoded, donor_a, donor_b),
                    "routed_answer_followed_donor": answer == donor_answer,
                    "random_decoder_followed_donor": donor_match(op, rand_decoded, donor_a, donor_b),
                    "random_routed_answer_followed_donor": rand_answer == donor_answer,
                }
            )
        log(f"{op}: finished patching")
    summary = summarize(rows)
    payload = {
        "fit_seed": args.fit_seed,
        "eval_seed": args.eval_seed,
        "n_pairs_per_op_requested": args.n_pairs_per_op,
        "ops": summary,
        "records": rows,
    }
    payload["verdict"] = verdict(summary, args.min_follow, args.max_random_follow)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(Path(args.out_md), payload)
    if args.out_records:
        with Path(args.out_records).open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"verdict": payload["verdict"], "ops": payload["ops"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
