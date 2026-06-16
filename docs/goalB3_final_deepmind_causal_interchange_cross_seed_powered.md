# Goal B3 Causal Interchange Cross-Seed Summary

- verdict: **CAUSAL_UNDERPOWERED**
- runs: 3

| op | verdict | runs | total pairs | min pairs | min decoder donor-follow | min routed donor-follow | max random routed-follow |
|---|---|---:|---:|---:|---:|---:|---:|
| `div_remainder` | `CAUSAL_GATE_PASS` | 3 | 75 | 25 | 1.000 | 1.000 | 0.000 |
| `gcd` | `CAUSAL_UNDERPOWERED` | 3 | 18 | 5 | 1.000 | 1.000 | 0.000 |
| `lcm` | `CAUSAL_UNDERPOWERED` | 3 | 20 | 6 | 1.000 | 1.000 | 0.000 |

This aggregation does not rerun models. It checks that selected L22 operand
chunk patches make decoder and routed answer follow the donor, while random
non-selected chunk patches stay below the frozen random-follow gate.

Verdicts distinguish rate failures from underpowered evidence: an op is
`CAUSAL_UNDERPOWERED` when observed donor-follow/control rates pass but
the frozen seed-count or pair-count gate is not met.
