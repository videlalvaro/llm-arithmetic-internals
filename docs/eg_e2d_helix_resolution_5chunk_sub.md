# eg_e2d_helix_resolution_5chunk — does CDMA-crowding extrapolate? (op=sub)

Tests three predictions made by the eg_e2c reading (chunks crowd in residual; longer answers → worse crowding).

## Verdict — 2/3 predictions pass

CDMA-CROWDING-CONFIRMED — longer answers DO crowd adjacent chunks more, supporting the capacity-saturation reading of the d>14 breakdown.

## (A) Per-chunk R² — d=12 [3,3,3,3] vs d=14 [3,3,3,3,2]

| chunk | d=12 R² | d=14 R² | Δ |
|---|---:|---:|---:|
| c1 | +0.9582 | +0.9529 | -0.0053 |
| c2 | +0.9266 | +0.8885 | -0.0381 |
| c3 | +0.9259 | +0.8814 | -0.0445 |
| c4 | +0.9173 | +0.8621 | -0.0552 |
| c5 | — | +0.9674 | (period-100, not directly comparable) |

## (D) Adjacent-chunk min principal angles

| pair | d=12 | d=14 |
|---|---:|---:|
| c1_vs_c2 | 71.3° | 65.3° |
| c2_vs_c3 | 66.0° | 60.2° |
| c3_vs_c4 | 67.4° | 58.8° |
| c4_vs_c5 | — | 69.9° |

## Predictions

- **P1** — c4↔c5 in d=14 should be ≤ c2↔c3 in d=12 (the worst adjacent angle there, 66°). Observed: **69.9°**. → **FAIL**
- **P2** — same pair c3↔c4 should crowd MORE in d=14 than d=12. d=12: 67.4°, d=14: 58.8° (Δ=-8.6°). → **PASS**
- **P3** — per-chunk R² (c1-c4) should drop in d=14. Mean Δ R² = -0.0358. → **PASS**

