# eg_e2c_helix_resolution — helix-as-float-precision (op=sub)

Test the 'helix has bounded precision per chunk, errors compound like float arithmetic' analogy. 875 CORRECT 4-chunk answers (d=12, canonical [3,3,3,3]).

## (A) Per-chunk R² profile

| chunk-k | R² (from pre_k @ L31) |
|---|---:|
| c1 | +0.9582 |
| c2 | +0.9266 |
| c3 | +0.9259 |
| c4 | +0.9173 |

**Reading**: FLAT — per-chunk R² is uniform across positions, consistent with the 'bounded mantissa per chunk' prediction.

## (B) Per-frequency R² × chunk-position

| chunk | freq=1 | freq=2 | freq=5 | freq=10 | freq=20 |
|---|---:|---:|---:|---:|---:|
| c1 | +0.978 | +0.967 | +0.915 | +0.969 | +0.962 |
| c2 | +0.956 | +0.929 | +0.849 | +0.959 | +0.940 |
| c3 | +0.952 | +0.922 | +0.853 | +0.959 | +0.944 |
| c4 | +0.947 | +0.913 | +0.806 | +0.967 | +0.954 |

**Reading**: UNIFORM-FREQ-DECAY — all freqs decay together; not a clean LSB-first signature.

## (C) Angular reconstruction RMS error

| chunk | freq | rms (rad) | rms (chunk-units) |
|---|---:|---:|---:|
| c1 | 1 | 0.0905 | 14.40 |
| c1 | 2 | 0.1372 | 21.84 |
| c1 | 5 | 0.2269 | 36.11 |
| c1 | 10 | 0.1169 | 18.60 |
| c1 | 20 | 0.1401 | 22.30 |
| c2 | 1 | 0.1506 | 23.97 |
| c2 | 2 | 0.2040 | 32.47 |
| c2 | 5 | 0.3272 | 52.07 |
| c2 | 10 | 0.1452 | 23.12 |
| c2 | 20 | 0.1701 | 27.08 |
| c3 | 1 | 0.1626 | 25.88 |
| c3 | 2 | 0.2235 | 35.57 |
| c3 | 5 | 0.3314 | 52.74 |
| c3 | 10 | 0.1420 | 22.60 |
| c3 | 20 | 0.1738 | 27.67 |
| c4 | 1 | 0.1738 | 27.67 |
| c4 | 2 | 0.2402 | 38.23 |
| c4 | 5 | 0.4284 | 68.18 |
| c4 | 10 | 0.1328 | 21.13 |
| c4 | 20 | 0.1581 | 25.16 |

## (D) Inter-chunk subspace principal angles

| pair | min angle | mean angle |
|---|---:|---:|
| c1_vs_c2 | 71.3° | 76.9° |
| c1_vs_c3 | 72.4° | 77.5° |
| c1_vs_c4 | 74.9° | 79.5° |
| c2_vs_c3 | 66.0° | 72.3° |
| c2_vs_c4 | 71.0° | 75.9° |
| c3_vs_c4 | 67.4° | 74.4° |

**Reading**: SUBSPACE-CROWDING — min principal angle 66.0°, well below 90° — chunks share readout directions (capacity-saturation signature, H1).

