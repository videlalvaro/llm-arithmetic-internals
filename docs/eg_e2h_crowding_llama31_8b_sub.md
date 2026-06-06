# eg_e2h_crowding — adjacent-chunk subspace angles (tag=llama31_8b, op=sub, layer=31, d_model=4096)

## Per-pattern summary

| pattern | n_chunks | n | min adj angle | mean adj angle | random baseline | gap (random − observed) |
|---|---:|---:|---:|---:|---:|---:|
| d8_332 | 3 | 940 | 73.1° | 75.5° | 84.6° | 11.5° |
| d12_3333 | 4 | 875 | 66.0° | 68.2° | 84.1° | 18.1° |
| d10_3331 | 4 | 757 | 65.2° | 73.5° | 84.3° | 19.2° |
| d14_33332 | 5 | 469 | 58.8° | 63.5° | 84.2° | 25.4° |

## Per-pattern detail

### d8_332

Per-chunk R²:

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9666 |
| c2 | 1000 | +0.8973 |
| c3 | 100 | +0.9839 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 73.1° |
| c2_vs_c3 | 77.9° |

### d12_3333

Per-chunk R²:

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9582 |
| c2 | 1000 | +0.9266 |
| c3 | 1000 | +0.9259 |
| c4 | 1000 | +0.9173 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 71.3° |
| c2_vs_c3 | 66.0° |
| c3_vs_c4 | 67.4° |

### d10_3331

Per-chunk R²:

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9644 |
| c2 | 1000 | +0.9061 |
| c3 | 1000 | +0.9236 |
| c4 | 10 | +0.9940 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 70.6° |
| c2_vs_c3 | 65.2° |
| c3_vs_c4 | 84.7° |

### d14_33332

Per-chunk R²:

| chunk | period | R² |
|---|---:|---:|
| c1 | 1000 | +0.9529 |
| c2 | 1000 | +0.8885 |
| c3 | 1000 | +0.8814 |
| c4 | 1000 | +0.8621 |
| c5 | 100 | +0.9674 |

Adjacent-pair min principal angles:

| pair | angle |
|---|---:|
| c1_vs_c2 | 65.3° |
| c2_vs_c3 | 60.2° |
| c3_vs_c4 | 58.8° |
| c4_vs_c5 | 69.9° |

