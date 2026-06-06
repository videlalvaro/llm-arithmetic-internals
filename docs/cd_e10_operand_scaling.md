# cd_e10_operand_scaling — Llama-3.1-8B subtraction limit

**Verdict**: 50% exact-match threshold lies between 13 and 14 digits

Coarse grid + binary search for the 50% exact-match threshold. 30 subtractions per evaluated digit-band, free-generation (greedy).

## all evaluated bands

| digits | n | exact | off≤1 | prefix correct | prefix % | mean Lev |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 30 | 96.67% | 0.00% | 5.8 | 97.2% | 0.10 |
| 10 | 30 | 63.33% | 16.67% | 8.1 | 80.7% | 0.60 |
| 13 | 30 | 53.33% | 26.67% | 9.6 | 73.8% | 1.07 |
| 14 | 30 | 43.33% | 30.00% | 9.4 | 67.1% | 1.07 |
| 16 | 30 | 33.33% | 33.33% | 10.3 | 64.6% | 1.40 |
| 24 | 30 | 6.67% | 3.33% | 4.2 | 17.6% | 8.77 |
| 40 | 30 | 0.00% | 0.00% | 3.2 | 7.9% | 14.80 |
| 64 | 30 | 0.00% | 0.00% | 0.1 | 0.1% | 40.47 |

- **exact** — emitted digit-string == gold
- **off≤1** — Levenshtein digit-distance == 1
- **prefix correct** — average longest MSD-first correct prefix
- **prefix %** — that prefix length as a fraction of gold length
- **mean Lev** — mean Levenshtein distance (emitted vs gold)

## bisection trace

| step | d (mid) | bracket | exact-rate | new bracket |
|---|---:|---|---:|---|
| 1 | 13 | [10, 16] | 53.33% | [13, 16] |
| 2 | 14 | [13, 16] | 43.33% | [13, 14] |

