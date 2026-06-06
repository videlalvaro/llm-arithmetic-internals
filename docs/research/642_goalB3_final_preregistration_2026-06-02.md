# Goal B3 Final Frozen Preregistration

Date: 2026-06-02

This document freezes the next claim-bearing Goal B3 rerun. It is prospective
for future final reruns. The existing artifacts listed in the package audit are
strong retrospective evidence, but they were produced before this final
preregistration and must be labeled that way.

## Claim

Activation-derived tool use can improve exact arithmetic benchmark accuracy
under an opaque no-parser runtime:

```text
opaque prompt token IDs + captured activations
  -> activation-derived op
  -> activation-derived operands
  -> Python calculator only after decoded (op, a, b)
  -> exact-answer routing / scoring
```

Runtime must not receive prompt text, regex matches, tokenizer-decoded operand
spans, CLI operation, harness operands, or gold answers.

Post-run claim-control note: see
`docs/research/648_goalB3_claim_control_2026-06-03.md`. The current supported
claim is activation-derived tool arguments on Llama. It does not support native
arithmetic repair, residual JIT replacement, Qwen transfer, or powered final
DeepMind causal validation.

## Frozen Runtime Contract

- Primary model: `unsloth/Meta-Llama-3.1-8B`
- Strict transfer attempt: `Qwen/Qwen2.5-7B`
- Op source: `activation`
- Operand source: `activation`
- Answer source: `python_from_decoded_tuple`
- Operand route: `attention_j16_l22_chunk`
- Operand bounds: `[0, 9999]`
- Chunk probe: `docs/j16_multitoken_operand_probe.pt`

Frozen thresholds:

| parameter | value |
|---|---:|
| op threshold minimum | 0.65 |
| op negative margin | 0.05 |
| safe threshold minimum | 0.65 |
| safe negative margin | 0.05 |
| `mul` pair threshold | 0.05 |
| `div_remainder` pair threshold | 0.20 |
| `lcm` pair threshold | 0.20 |
| `gcd` pair threshold | 0.20 |
| chunk top-k | 12 |
| chunk window | 1 |
| chunk position threshold | 0.50 |
| chunk value margin threshold | 0.00 |
| max generation tokens | 12 |

## Benchmark Tiers

### Broad Frozen Arithmetic/Adversarial

- Runner: `scripts/goalB3_repaired_benchmark_suite.py`
- Split source: `broad_frozen_arithmetic_adversarial`
- Ops: `mul`, `div_remainder`, `lcm`, `gcd`
- Seeds: `801`, `811`, `821`
- `n_per_family`: 80
- `n_adversarial_per_family`: 250
- `fit_b3_aug_n_per_family`: 20

### DeepMind Interpolate Recognized Source

- Runner: `scripts/goalB3_repaired_benchmark_suite.py`
- Split source: `deepmind_interpolate`
- Ops: `gcd`, `div_remainder`, `lcm`
- Seeds: `911`, `921`, `931`
- `n_per_family`: 500
- `n_natural`: 200
- `dm_scan_limit`: 200000
- `include_common_denominator`: true
- Minimum locked examples: 1000
- Minimum target examples per op: 50

DeepMind `mul` is excluded from the powered recognized-source claim because
the source audit found too few supported two-integer locked targets under the
frozen route. It remains part of the broad frozen benchmark.

## Causal Interchange

- Runner: `scripts/goalB3_causal_interchange.py`
- Ops: `mul`, `div_remainder`, `lcm`, `gcd`
- Fit/eval seed pairs: `701/702`, `711/712`, `721/722`
- Requested patch pairs per op per seed: 20
- Intervention: patch selected donor L22 operand chunks into recipient selected
  operand chunk positions.
- Controls: random non-selected donor chunk patches.

Pass gates:

- total causal pairs per op >= 50;
- decoder donor-follow >= 1.0;
- routed answer donor-follow >= 1.0;
- random routed donor-follow <= 0.10.

## Adversarial/Paraphrase Families

Target robustness families:

- `pre_distractor`
- `between_distractor`
- `post_distractor`

Hard-negative safety families:

- `quoted_expression_negative`
- `do_not_compute_negative`
- `wrong_op_negative`
- `natural_numeric_negative`

Hard-negative false-fire must remain <= 0.01.

## Acceptance Gates

A final positive requires:

- at least 1000 locked examples in the benchmark tier;
- at least 3 operations;
- per-op exact-score lift >= +0.20 on locked target examples;
- hard-negative false-fire <= 0.01;
- pair-exact operand decode on fired targets >= 0.80;
- 3 frozen seeds with mean and minimum metrics;
- scaled causal donor-patch validation at the gates above;
- strict non-Llama transfer attempt or documented falsifier;
- emitted provenance records with activation op source, activation operand
  source, and Python-from-decoded-tuple answer source.

## Verdict Rules

- `NEURIPS_PACKAGE_PASS`: all benchmark, provenance, adversarial, causal, and
  strict-transfer/falsifier gates pass prospectively under this preregistration.
- `BENCHMARK_FAIL`: exact lift, false-fire, target coverage, or pair-exact gates
  fail.
- `CAUSAL_FAIL`: donor patch does not make decoder and routed answer follow the
  donor, or random controls exceed the frozen gate.
- `CAUSAL_UNDERPOWERED`: observed donor-follow/control rates pass, but the
  frozen seed-count or `>=50` total-pairs-per-op gate is not met.
- `PROVENANCE_FAIL`: runtime records or source tests show parser/token/harness
  leakage.
- `TRANSFER_FAIL_WITH_LLAMA_PASS`: Llama package passes but strict non-Llama
  transfer fails; acceptable only if documented as a falsifier rather than a
  positive transfer result.

## Manifest

The machine-readable frozen manifest is:

- `docs/goalB3_final_frozen_manifest.json`

The verifier for this preregistration must check that runner defaults,
package-audit artifacts, and current evidence remain consistent with the
manifest before a final rerun is claimed.
