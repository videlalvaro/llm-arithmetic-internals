# eg_e2e_crowding_vs_chunks — does crowding scale monotonically? (op=sub)

## Trend: c3↔c4 min principal angle across patterns

| pattern | n_chunks | n | c3↔c4 |
|---|---:|---:|---:|
| d12_3333 | 4 | 1442 | 77.4° |
| d14_33332 | 5 | 1472 | 80.6° |
| d16_333331 | 6 | 917 | 73.1° |
| d18_333333 | 6 | 987 | 75.1° |
| d20_3333332 | 7 | 988 | 76.3° |

**Verdict**: NON-MONOTONE — c3↔c4 trajectory: 77.4° → 80.6° → 73.1° → 75.1° → 76.3°. The crowding signal is not strictly monotone.

## Per-pattern detail

### d12_3333 (pattern=[3, 3, 3, 3], n=1442, correct=875)

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9169 |
| c2 | 1000 | +0.7858 |
| c3 | 1000 | +0.8528 |
| c4 | 1000 | +0.8999 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 79.3° |
| c2_vs_c3 | 78.4° |
| c3_vs_c4 | 77.4° |

### d14_33332 (pattern=[3, 3, 3, 3, 2], n=1472, correct=469)

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9460 |
| c2 | 1000 | +0.7412 |
| c3 | 1000 | +0.7377 |
| c4 | 1000 | +0.7756 |
| c5 | 100 | +0.9747 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 81.2° |
| c2_vs_c3 | 80.6° |
| c3_vs_c4 | 80.6° |
| c4_vs_c5 | 82.5° |

### d16_333331 (pattern=[3, 3, 3, 3, 3, 1], n=917, correct=205)

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9334 |
| c2 | 1000 | +0.7749 |
| c3 | 1000 | +0.7971 |
| c4 | 1000 | +0.8309 |
| c5 | 1000 | +0.8844 |
| c6 | 10 | +0.8937 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 76.2° |
| c2_vs_c3 | 75.4° |
| c3_vs_c4 | 73.1° |
| c4_vs_c5 | 72.5° |
| c5_vs_c6 | 84.7° |

### d18_333333 (pattern=[3, 3, 3, 3, 3, 3], n=987, correct=241)

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.8181 |
| c2 | 1000 | +0.7606 |
| c3 | 1000 | +0.8335 |
| c4 | 1000 | +0.8474 |
| c5 | 1000 | +0.8998 |
| c6 | 1000 | +0.7454 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 79.8° |
| c2_vs_c3 | 75.5° |
| c3_vs_c4 | 75.1° |
| c4_vs_c5 | 70.7° |
| c5_vs_c6 | 75.6° |

### d20_3333332 (pattern=[3, 3, 3, 3, 3, 3, 2], n=988, correct=135)

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9286 |
| c2 | 1000 | +0.7904 |
| c3 | 1000 | +0.7754 |
| c4 | 1000 | +0.8470 |
| c5 | 1000 | +0.8759 |
| c6 | 1000 | +0.7592 |
| c7 | 100 | +0.9633 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 75.7° |
| c2_vs_c3 | 76.9° |
| c3_vs_c4 | 76.3° |
| c4_vs_c5 | 73.6° |
| c5_vs_c6 | 75.9° |
| c6_vs_c7 | 81.6° |

