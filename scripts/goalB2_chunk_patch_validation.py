#!/usr/bin/env python3
"""Causal validation for Goal B2 L22 chunk decoder.

This patches selected L22 operand chunk activations from donor prompts into
recipient prompts and tests whether the activation-only decoder and routed
calculator follow the donor operand state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import goalB2_lcm_benchmark_pipeline as b2
import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_chunk_position(diag: dict) -> int | None:
    groups = diag.get("chunk_groups") or []
    if not groups or not groups[0].get("chunks"):
        return None
    return int(groups[0]["chunks"][0]["pos"])


def first_two_groups(diag: dict) -> list[dict]:
    groups = diag.get("chunk_groups") or []
    return [g for g in groups[:2] if g.get("chunks")]


def group_positions(group: dict) -> list[int]:
    return [int(c["pos"]) for c in group.get("chunks", [])]


def group_values(diag: dict) -> list[int]:
    vals = []
    for group in first_two_groups(diag):
        text = "".join(str(int(c["value"])) for c in group.get("chunks", []))
        if text:
            vals.append(int(text))
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="docs/goalB2_mul_chunk_frozen_splits.jsonl")
    ap.add_argument("--records", default="docs/goalB2_mul_chunk_frozen_records.jsonl")
    ap.add_argument("--readouts", default="docs/goalB2_mul_chunk_frozen_readouts.npz")
    ap.add_argument("--chunk_probe", default="docs/j16_multitoken_operand_probe.pt")
    ap.add_argument("--out_json", default="docs/goalB2_chunk_patch_validation.json")
    ap.add_argument("--out_md", default="docs/goalB2_chunk_patch_validation.md")
    ap.add_argument("--target_op", choices=["mul", "div_remainder", "lcm"], default="mul")
    ap.add_argument("--operand_lo", type=int, default=0)
    ap.add_argument("--operand_hi", type=int, default=9999)
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=642)
    args = ap.parse_args()

    split_by_id = {r["example_id"]: r for r in load_jsonl(Path(args.splits))}
    records = [r for r in load_jsonl(Path(args.records)) if r["is_target"] and r["fired"]]
    records = [
        r
        for r in records
        if first_chunk_position(r.get("readout_diagnostics") or {}) is not None
    ]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(records)
    pairs = list(zip(records[::2], records[1::2], strict=False))[: args.n_pairs]

    model, _tok, device = b2.load_llama()
    readouts = b2.Readouts.load(Path(args.readouts))
    readouts.pair_conf_threshold = 0.0
    chunk_probe = b2.J16ChunkProbe.load(Path(args.chunk_probe))
    guard = b2.ProvenanceGuard(runtime_mode=True, allowed_op=args.target_op)

    rows = []
    for recipient_rec, donor_rec in pairs:
        rec_ex = b2.Example(**split_by_id[recipient_rec["example_id"]])
        donor_ex = b2.Example(**split_by_id[donor_rec["example_id"]])
        rec_rt = b2.runtime_from_example(rec_ex, args.seed, "llama", model, device)
        donor_rt = b2.runtime_from_example(donor_ex, args.seed, "llama", model, device)
        rec_groups = first_two_groups(recipient_rec["readout_diagnostics"])
        donor_groups = first_two_groups(donor_rec["readout_diagnostics"])
        if len(rec_groups) < 2 or len(donor_groups) < 2:
            continue
        patched_acts = dict(rec_rt.activations)
        H = np.array(patched_acts["all_positions_L22"], copy=True)
        patch_pairs = []
        shape_mismatch = False
        for rec_group, donor_group in zip(rec_groups, donor_groups, strict=True):
            rec_positions = group_positions(rec_group)
            donor_positions = group_positions(donor_group)
            if len(rec_positions) != len(donor_positions):
                shape_mismatch = True
                break
            for rec_pos, donor_pos in zip(rec_positions, donor_positions, strict=True):
                H[rec_pos] = donor_rt.activations["all_positions_L22"][donor_pos]
                patch_pairs.append({"recipient_pos": rec_pos, "donor_pos": donor_pos})
        if shape_mismatch:
            continue
        patched_acts["all_positions_L22"] = H
        patched_rt = b2.RuntimeInputs(rec_rt.example_id, rec_rt.prompt_ids, patched_acts)
        decoded, diag = b2.decode_from_activations(
            patched_rt,
            readouts,
            guard,
            backend="llama",
            target_op=args.target_op,
            operand_lo=args.operand_lo,
            operand_hi=args.operand_hi,
            operand_decode_mode="attention_j16_l22_chunk",
            chunk_probe=chunk_probe,
        )
        donor_values = group_values(donor_rec["readout_diagnostics"])
        patched_values = group_values(diag)
        donor_a = int(donor_rec["decoded_a"])
        donor_b = int(donor_rec["decoded_b"])
        if args.target_op == "div_remainder":
            decoder_followed_donor = decoded is not None and decoded.a == donor_a and decoded.b == donor_b
        else:
            decoder_followed_donor = (
                decoded is not None and {decoded.a, decoded.b} == {donor_a, donor_b}
            )
        donor_answer = b2.compute_target(args.target_op, donor_a, donor_b)
        routed_answer = None if decoded is None else guard.calculator(decoded)
        rows.append(
            {
                "recipient": recipient_rec["example_id"],
                "donor": donor_rec["example_id"],
                "patch_pairs": patch_pairs,
                "donor_group_values": donor_values,
                "patched_group_values": patched_values,
                "donor_decoded": {"a": donor_a, "b": donor_b},
                "donor_answer": donor_answer,
                "routed_answer_after_patch": routed_answer,
                "decoder_followed_donor": decoder_followed_donor,
                "routed_answer_followed_donor": routed_answer == donor_answer,
                "patched_decoded": None if decoded is None else {"a": decoded.a, "b": decoded.b},
            }
        )

    rate = float(np.mean([r["decoder_followed_donor"] for r in rows])) if rows else float("nan")
    routed_rate = (
        float(np.mean([r["routed_answer_followed_donor"] for r in rows]))
        if rows
        else float("nan")
    )
    payload = {
        "target_op": args.target_op,
        "n_pairs": len(rows),
        "decoder_follow_donor_rate": rate,
        "routed_answer_follow_donor_rate": routed_rate,
        "records": rows,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")
    Path(args.out_md).write_text(
        "\n".join(
            [
                "# Goal B2 Chunk Patch Validation",
                "",
                f"- target op: `{args.target_op}`",
                f"- pairs: {len(rows)}",
                f"- decoder followed donor operand pair: {rate:.3f}",
                f"- routed answer followed donor tuple: {routed_rate:.3f}",
                "",
                "This validates decoder and routed-answer sensitivity to patched L22 "
                "operand chunk activations; it does not validate the op gate.",
            ]
        )
    )
    print(
        json.dumps(
            {
                "n_pairs": len(rows),
                "decoder_follow_donor_rate": rate,
                "routed_answer_follow_donor_rate": routed_rate,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
