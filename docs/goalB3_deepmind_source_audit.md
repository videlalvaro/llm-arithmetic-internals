# Goal B3 DeepMind Source Audit

Eval-only coverage audit. This does not run the activation route and does not make a Goal B claim.

- DeepMind root: `$DEEPMIND_MATH_ROOT`
- scan limit per file: 200000
- operand range: [0, 9999]

## Coverage

| split | file | op | scanned | accepted | locked 40% estimate | top rejection |
|---|---|---|---:|---:|---:|---|
| `interpolate` | `arithmetic__mul.txt` | `mul` | 10000 | 69 | 27 | `non_integer_gold`=7038 |
| `interpolate` | `numbers__div_remainder.txt` | `div_remainder` | 10000 | 414 | 165 | `operand_range`=9586 |
| `interpolate` | `numbers__gcd.txt` | `gcd` | 10000 | 194 | 77 | `operand_range`=9806 |
| `interpolate` | `numbers__lcm.txt` | `lcm` | 10000 | 422 | 168 | `operand_range`=9574 |
| `extrapolate` | `arithmetic__mul_big.txt` | `mul` | 10000 | 1 | 0 | `non_integer_gold`=7147 |
| `extrapolate` | `arithmetic__mul_div_multiple_longer.txt` | `mul` | 10000 | 11 | 4 | `operand_range`=4519 |

## Interpretation

- 3 source files appear to have enough supported examples for the current target-count gate before fitting/eval.
- If `arithmetic__mul.txt` remains sparse, a claim-bearing DeepMind 3-op pass must either expand the supported operand/value regime before preregistration or use a different recognized source with enough target coverage.
