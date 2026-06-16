#!/usr/bin/env python3
"""Generate article-friendly helix figures from existing Rune artifacts.

This script is intentionally lightweight: it does not run a model or read
activation memmaps. It creates:

1. A conceptual "integer helix" figure.
2. An empirical helix-resolution figure from docs/eg_e2c_helix_resolution_*.json.
3. An empirical operand Fourier-readout heatmap from docs/f3_operand_helix_decoder.json.

Outputs PNG and SVG files under docs/article_figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = DOCS / "article_figures"


def save(fig, stem: str, dpi: int = 180) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {svg}")


def conceptual_helix() -> None:
    import matplotlib.pyplot as plt

    vals = np.arange(0, 80)
    dense = np.linspace(0, 79, 800)
    period = 10
    theta_dense = 2 * np.pi * dense / period
    xd = np.cos(theta_dense)
    yd = np.sin(theta_dense)
    zd = dense / period

    theta = 2 * np.pi * vals / period
    x = np.cos(theta)
    y = np.sin(theta)
    z = vals / period

    fig = plt.figure(figsize=(8.6, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xd, yd, zd, color="#1f77b4", linewidth=2.2)
    ax.scatter(x, y, z, c=vals, cmap="viridis", s=22, depthshade=True)

    for v in [0, 10, 20, 30, 40, 50, 60, 70]:
        ax.text(x[v] * 1.08, y[v] * 1.08, z[v], str(v), fontsize=9)

    ax.set_title("A matrix-only way to count: integers as phase plus height", pad=16)
    ax.set_xlabel("cos(2*pi*n/10)")
    ax.set_ylabel("sin(2*pi*n/10)")
    ax.set_zlabel("coarse value")
    ax.view_init(elev=23, azim=-55)
    ax.grid(alpha=0.25)

    # Keep panes quiet so the figure reads well in print.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.05)

    save(fig, "conceptual_integer_helix")
    plt.close(fig)


def helix_resolution(op: str) -> None:
    import matplotlib.pyplot as plt

    path = DOCS / f"eg_e2c_helix_resolution_{op}.json"
    data = json.loads(path.read_text())
    chunks = [1, 2, 3, 4]
    freqs = [1, 2, 5, 10, 20]

    per_chunk = data["per_chunk_R2"]
    r2s = [float(per_chunk[str(k)] if str(k) in per_chunk else per_chunk[k]) for k in chunks]

    per_freq = data["per_freq_R2"]
    heat = np.array([
        [float(per_freq[str(k)][str(f)] if str(f) in per_freq[str(k)] else per_freq[str(k)][f])
         for f in freqs]
        for k in chunks
    ])

    angles = data["inter_chunk_principal_angles"]
    pairs = list(angles.keys())
    mins = [float(angles[p]["min_principal_angle_deg"]) for p in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)

    ax = axes[0]
    ax.plot(chunks, r2s, marker="o", color="#0f6b8f", linewidth=2.4)
    ax.set_title("Chunk value is phase-decodable")
    ax.set_xlabel("answer chunk")
    ax.set_ylabel("held-out R2")
    ax.set_xticks(chunks)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)

    ax = axes[1]
    im = ax.imshow(heat, cmap="mako_r" if "mako_r" in plt.colormaps() else "viridis",
                   aspect="auto", vmin=0, vmax=1)
    ax.set_title("Fourier readout by frequency")
    ax.set_xlabel("frequency")
    ax.set_ylabel("answer chunk")
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(freqs)
    ax.set_yticks(range(len(chunks)))
    ax.set_yticklabels([f"c{k}" for k in chunks])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if heat[i, j] < 0.45 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[2]
    ax.bar(range(len(pairs)), mins, color="#5b8e3e")
    ax.axhline(90, color="#777777", linestyle=":", linewidth=1.4)
    ax.axhline(70, color="#d08b00", linestyle=":", linewidth=1.4)
    ax.set_title("Chunk readout subspaces separate")
    ax.set_ylabel("min principal angle, degrees")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("c", "").replace("_vs_", "-") for p in pairs], rotation=40)
    ax.set_ylim(0, 95)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Empirical helix geometry in Rune ({op}, L31)", fontsize=14)
    save(fig, f"empirical_helix_resolution_{op}")
    plt.close(fig)


def operand_fourier_heatmap() -> None:
    import matplotlib.pyplot as plt

    path = DOCS / "f3_operand_helix_decoder.json"
    data = json.loads(path.read_text())
    per_probe = data["per_probe"]

    layers = sorted({int(v["L"]) for v in per_probe.values() if v.get("kind") == "fourier"})
    periods = sorted({int(v.get("period", v.get("T"))) for v in per_probe.values() if v.get("kind") == "fourier"})
    heat = np.full((len(layers), len(periods)), np.nan)
    for i, layer in enumerate(layers):
        for j, period in enumerate(periods):
            rec = per_probe.get(f"fourier_L{layer}_T{period}")
            if rec is not None:
                heat[i, j] = float(rec["r2"])

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    im = ax.imshow(heat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Operand values are readable as Fourier phases")
    ax.set_xlabel("period T")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat[i, j]):
                label = "<0" if heat[i, j] < 0 else f"{heat[i, j]:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if heat[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="held-out Fourier R2")
    subtitle = (
        f"best Fourier: {data.get('best_fourier_key')} "
        f"(R2={float(data.get('best_fourier_r2', float('nan'))):.3f})"
    )
    ax.text(0.5, -0.18, subtitle, ha="center", va="center", transform=ax.transAxes)
    save(fig, "operand_fourier_readout_heatmap")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="sub", choices=["add", "sub"])
    ap.add_argument("--skip_concept", action="store_true")
    args = ap.parse_args()

    if not args.skip_concept:
        conceptual_helix()
    helix_resolution(args.op)
    operand_fourier_heatmap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
