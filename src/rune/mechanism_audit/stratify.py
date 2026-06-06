"""Unified stratified-table writer for MUST-FIX MF5 + MF7.

`StratifiedTable` requires every row to declare all dimensions (op × band ×
phrasing × answer_token_count × native_pair × arm by default). It refuses to
write any row missing a declared dimension. Pooled summaries are available only
via the explicit `pooled_summary(reason=...)` escape hatch that logs the
reason.

Provides 95% bootstrap CI per cell and a `verdict` helper that encapsulates the
PASS / KILL_TOO_SMALL / KILL_CONTROL_MATCHED logic from
`scripts/hpj_e155_stratified_interchange.py:153-191` (lines mostly mirrored).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


log = logging.getLogger(__name__)


DEFAULT_DIMENSIONS = ("op", "band", "phrasing", "answer_token_count", "native_pair", "arm")


@dataclass
class Verdict:
    label: str          # "PASS" | "KILL_TOO_SMALL" | "KILL_CONTROL_MATCHED" | "AMBIGUOUS"
    metric: str
    mean: float | None
    n: int
    ci_lo: float | None
    ci_hi: float | None
    notes: str = ""


class PooledWriteError(RuntimeError):
    pass


class StratifiedTable:
    """Append-only stratified accumulator.

    Usage:
        t = StratifiedTable(dimensions=("op", "band", "arm"))
        t.add({"op":"add", "band":"d5", "arm":"L29_patch"}, {"donor_follow": 1.0})
        t.add({"op":"add", "band":"d5", "arm":"L29_patch"}, {"donor_follow": 0.0})
        summary = t.summarize("donor_follow")              # per-cell mean + bootstrap CI
        t.write_markdown(Path("out.md"), metrics=["donor_follow"])
    """

    def __init__(self, dimensions: Sequence[str] = DEFAULT_DIMENSIONS, seed: int = 17):
        self.dimensions: tuple[str, ...] = tuple(dimensions)
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions must be unique")
        # cell_key (tuple of dimension values, in declared order) -> {metric_name: list[float]}
        self._cells: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # cell_key -> dict of dimension key→value (for write)
        self._cell_index: dict[tuple, dict] = {}
        self._rng = np.random.default_rng(seed)

    # ---- ingest ----

    def _check_cell_key(self, cell_key: dict) -> tuple:
        missing = [d for d in self.dimensions if d not in cell_key]
        if missing:
            raise PooledWriteError(
                f"row missing required dimension(s) {missing}; got {sorted(cell_key)}"
            )
        return tuple(cell_key[d] for d in self.dimensions)

    def add(self, cell_key: dict, metrics: dict[str, float | bool | None]) -> None:
        """Add one row's metric values to the cell indexed by cell_key.

        Booleans and None are normalised to {0.0, 1.0, NaN}.
        """
        key = self._check_cell_key(cell_key)
        if key not in self._cell_index:
            self._cell_index[key] = {d: cell_key[d] for d in self.dimensions}
        for m_name, m_val in metrics.items():
            if m_val is None:
                v = float("nan")
            elif isinstance(m_val, bool):
                v = 1.0 if m_val else 0.0
            else:
                v = float(m_val)
            self._cells[key][m_name].append(v)

    # ---- summary ----

    def _bootstrap_ci(self, vals: np.ndarray, n_resamples: int = 1000,
                     alpha: float = 0.05) -> tuple[float | None, float | None]:
        if vals.size == 0:
            return None, None
        if vals.size == 1:
            return float(vals[0]), float(vals[0])
        n = vals.size
        idx = self._rng.integers(0, n, size=(n_resamples, n))
        means = vals[idx].mean(axis=1)
        lo = float(np.quantile(means, alpha / 2))
        hi = float(np.quantile(means, 1 - alpha / 2))
        return lo, hi

    def summarize(self, metric: str, ci: bool = True,
                  n_resamples: int = 1000) -> dict[tuple, dict]:
        """Return per-cell {mean, n, ci_lo, ci_hi} for the named metric.

        Cells with no recorded values for the metric are skipped.
        """
        out: dict[tuple, dict] = {}
        for key, metric_dict in self._cells.items():
            vals = np.asarray(metric_dict.get(metric, []), dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            ci_lo: float | None
            ci_hi: float | None
            if ci:
                ci_lo, ci_hi = self._bootstrap_ci(vals, n_resamples=n_resamples)
            else:
                ci_lo = ci_hi = None
            out[key] = {
                "mean": mean,
                "n": int(vals.size),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "dimensions": self._cell_index[key],
            }
        return out

    def verdict(self, metric: str, *,
                true_arm: str,
                control_arms: Sequence[str],
                arm_dim: str = "arm",
                pass_threshold: float = 0.5,
                control_margin: float = 0.3,
                n_min: int = 8) -> dict[tuple, Verdict]:
        """Per-cell verdict (PASS / KILL_*) mirroring hpj_e155.summarize logic.

        Returns a dict keyed by the *non-arm* cell tuple (i.e. cells with
        identical (op, band, phrasing, ...) values but different arms collapse
        into one verdict).
        """
        if arm_dim not in self.dimensions:
            raise ValueError(f"arm_dim {arm_dim!r} not in dimensions {self.dimensions}")
        arm_idx = self.dimensions.index(arm_dim)

        per_cell = self.summarize(metric, ci=True, n_resamples=500)

        # group cells by everything except arm
        groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
        for key, summary in per_cell.items():
            non_arm = tuple(v for i, v in enumerate(key) if i != arm_idx)
            arm_val = str(key[arm_idx])
            groups[non_arm][arm_val] = summary

        verdicts: dict[tuple, Verdict] = {}
        for non_arm_key, arm_summaries in groups.items():
            true_s = arm_summaries.get(true_arm)
            if true_s is None:
                continue
            ti_mean = true_s["mean"]
            ti_n = true_s["n"]
            ctrl_means = [arm_summaries[a]["mean"] for a in control_arms
                          if a in arm_summaries]
            if ti_n < n_min:
                v = Verdict(
                    label="KILL_TOO_SMALL", metric=metric, mean=ti_mean,
                    n=ti_n, ci_lo=true_s["ci_lo"], ci_hi=true_s["ci_hi"],
                    notes=f"n={ti_n} < {n_min}",
                )
            elif ti_mean < pass_threshold:
                v = Verdict(
                    label="KILL_BELOW_THRESHOLD", metric=metric, mean=ti_mean,
                    n=ti_n, ci_lo=true_s["ci_lo"], ci_hi=true_s["ci_hi"],
                    notes=f"true_arm mean {ti_mean:.3f} < threshold {pass_threshold}",
                )
            elif ctrl_means and not all((ti_mean - cm) >= control_margin
                                         for cm in ctrl_means):
                v = Verdict(
                    label="KILL_CONTROL_MATCHED", metric=metric, mean=ti_mean,
                    n=ti_n, ci_lo=true_s["ci_lo"], ci_hi=true_s["ci_hi"],
                    notes=f"controls within {control_margin} of true: {ctrl_means}",
                )
            else:
                v = Verdict(
                    label="PASS", metric=metric, mean=ti_mean,
                    n=ti_n, ci_lo=true_s["ci_lo"], ci_hi=true_s["ci_hi"],
                    notes=f"vs controls: {ctrl_means}",
                )
            verdicts[non_arm_key] = v
        return verdicts

    def pooled_summary(self, metric: str, *, reason: str,
                       group_by: Sequence[str] = ()) -> dict:
        """Explicit escape hatch for pooled summaries. Logs the reason at WARN.

        Per MUST-FIX 5: pooled headlines must be opt-in with a reason, never
        the default.
        """
        log.warning("pooled_summary called for metric=%s reason=%s group_by=%s",
                    metric, reason, list(group_by))
        all_vals: list[float] = []
        per_group: dict[tuple, list[float]] = defaultdict(list)
        for key, metric_dict in self._cells.items():
            vals = [v for v in metric_dict.get(metric, []) if not np.isnan(v)]
            all_vals.extend(vals)
            if group_by:
                gk = tuple(self._cell_index[key].get(g) for g in group_by)
                per_group[gk].extend(vals)
        out: dict[str, Any] = {
            "reason": reason,
            "pooled_n": len(all_vals),
            "pooled_mean": float(np.mean(all_vals)) if all_vals else None,
        }
        if group_by:
            out["by_group"] = {
                "/".join(str(x) for x in gk): {
                    "n": len(vs),
                    "mean": float(np.mean(vs)) if vs else None,
                }
                for gk, vs in per_group.items()
            }
        return out

    # ---- output ----

    def write_json(self, path: Path | str, metrics: Sequence[str] | None = None) -> None:
        path = Path(path)
        if metrics is None:
            metrics = sorted({m for d in self._cells.values() for m in d.keys()})
        cells_out: list[dict] = []
        for key, metric_dict in self._cells.items():
            row: dict[str, Any] = {"dimensions": self._cell_index[key]}
            for m in metrics:
                vals = np.asarray(metric_dict.get(m, []), dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row[m] = None
                    row[f"{m}_n"] = 0
                else:
                    lo, hi = self._bootstrap_ci(vals)
                    row[m] = float(vals.mean())
                    row[f"{m}_n"] = int(vals.size)
                    row[f"{m}_ci_lo"] = lo
                    row[f"{m}_ci_hi"] = hi
            cells_out.append(row)
        path.write_text(json.dumps({
            "dimensions": list(self.dimensions),
            "metrics": list(metrics),
            "cells": cells_out,
        }, indent=2) + "\n")

    def write_markdown(self, path: Path | str, metrics: Sequence[str] | None = None,
                       title: str | None = None) -> None:
        path = Path(path)
        if metrics is None:
            metrics = sorted({m for d in self._cells.values() for m in d.keys()})
        lines: list[str] = []
        if title:
            lines += [f"# {title}", ""]
        lines += [f"**Dimensions:** `{' × '.join(self.dimensions)}`", ""]

        header_cols = list(self.dimensions) + [m for mn in metrics for m in (mn, f"{mn}_n")]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

        sorted_keys = sorted(self._cells.keys(), key=lambda k: tuple(str(x) for x in k))
        for key in sorted_keys:
            dim_vals = self._cell_index[key]
            metric_dict = self._cells[key]
            row_cells = [str(dim_vals[d]) for d in self.dimensions]
            for m in metrics:
                vals = np.asarray(metric_dict.get(m, []), dtype=float)
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row_cells += ["—", "0"]
                else:
                    lo, hi = self._bootstrap_ci(vals, n_resamples=500)
                    if lo is None or hi is None:
                        row_cells.append(f"{vals.mean():.3f}")
                    else:
                        row_cells.append(f"{vals.mean():.3f} [{lo:.2f},{hi:.2f}]")
                    row_cells.append(str(vals.size))
            lines.append("| " + " | ".join(row_cells) + " |")
        path.write_text("\n".join(lines) + "\n")
