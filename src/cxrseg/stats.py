"""Statistical comparisons across contrast scenarios (docs/PROTOCOL.md section 9)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRIC_COLS = ("dice", "iou", "hd95", "coverage", "area_error_pct")


def average_over_seeds(df: pd.DataFrame, failure_dice: float = 0.90) -> pd.DataFrame:
    """Average per-image metrics across seeds (e.g. 11, 22, 33) before statistical tests.

    Recomputes `failed` as `dice < failure_dice` on the seed-averaged per-image Dice.
    """
    group_cols = [c for c in ("fold", "scenario", "id", "source", "group") if c in df.columns]
    agg_cols = [c for c in METRIC_COLS if c in df.columns]
    out = df.groupby(group_cols, as_index=False, sort=True)[agg_cols].mean()
    if "dice" in out.columns:
        out["failed"] = out["dice"] < float(failure_dice)
    return out


def bootstrap_mean_diff_ci(
    diff: Sequence[float] | np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of paired differences."""
    d = np.asarray(diff, dtype=np.float64)
    if d.size == 0:
        raise ValueError("diff must not be empty")
    rng = np.random.default_rng(seed)
    n = len(d)
    # Chunk bootstrap draws to keep memory bounded even for large n_boot * n
    means = np.empty(n_boot, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start : start + size] = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def paired_comparison(
    df: pd.DataFrame,
    baseline: str = "S0",
    scenario: str = "S1",
    source: str = "darwin",
    *,
    metric: str = "dice",
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
    run_wilcoxon: bool | None = None,
) -> dict:
    """Paired comparison of `scenario` vs `baseline` on `source` (foreign images when `group` is present).

    Runs paired Wilcoxon signed-rank test on Dice (or when `run_wilcoxon=True`)
    and computes mean difference (`scenario - baseline`) with percentile bootstrap CI.
    """
    sub = df[df["source"] == source]
    if "group" in sub.columns and (sub["group"] == "foreign").any():
        sub = sub[sub["group"] == "foreign"]

    base_df = sub[sub["scenario"] == baseline][["id", metric]].rename(columns={metric: "base"})
    scen_df = sub[sub["scenario"] == scenario][["id", metric]].rename(columns={metric: "scen"})
    merged = pd.merge(base_df, scen_df, on="id", how="inner").sort_values("id")
    if merged.empty:
        raise ValueError(f"No paired observations for source={source!r}, baseline={baseline!r}, scenario={scenario!r}")

    diff = merged["scen"].to_numpy(dtype=np.float64) - merged["base"].to_numpy(dtype=np.float64)
    mean_diff = float(diff.mean())
    ci_low, ci_high = bootstrap_mean_diff_ci(diff, n_boot=n_boot, seed=seed, alpha=alpha)

    if run_wilcoxon is None:
        run_wilcoxon = metric == "dice"

    p_value: float | None
    if run_wilcoxon:
        if np.allclose(diff, 0.0):
            p_value = 1.0
        else:
            try:
                p_value = float(wilcoxon(diff).pvalue)
            except ValueError:
                p_value = 1.0
    else:
        p_value = None

    return {
        "source": source,
        "baseline": baseline,
        "scenario": scenario,
        "metric": metric,
        "n_images": int(len(diff)),
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * float(pvalues[idx])
        val = max(val, prev)
        val = min(val, 1.0)
        adjusted[idx] = val
        prev = val
    return adjusted


def is_meaningful(
    p_holm: float,
    mean_diff: float,
    alpha: float = 0.05,
    min_effect: float = 0.01,
) -> bool:
    """True iff Holm-adjusted p < alpha AND |mean_diff| >= min_effect (PROTOCOL section 9)."""
    return bool(float(p_holm) < float(alpha) and abs(float(mean_diff)) >= float(min_effect))
