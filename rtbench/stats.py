from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_std_ci95(x: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(x, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) <= 1:
        return mean, std, mean, mean
    sem = stats.sem(arr)
    ci_low, ci_high = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=sem)
    return mean, std, float(ci_low), float(ci_high)


def wilcoxon_greater(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    # If all differences are exactly zero, scipy raises.
    if np.allclose(vals, 0.0):
        return 1.0
    return float(stats.wilcoxon(vals, alternative="greater", zero_method="wilcox").pvalue)


def bh_fdr(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(x) for x in out]


def summarize_vs_paper(
    per_seed_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    fdr_q: float,
) -> pd.DataFrame:
    rows = []
    p_mae = []
    p_r2 = []
    datasets = sorted(per_seed_df["dataset"].unique())
    for ds in datasets:
        cur = per_seed_df.loc[per_seed_df["dataset"] == ds]
        base = baseline_df.loc[baseline_df["dataset"] == ds].iloc[0]
        paper_mae = float(base["paper_mae"])
        paper_r2 = float(base["paper_r2"])
        mae_vals = cur["mae"].to_numpy()
        r2_vals = cur["r2"].to_numpy()
        p1 = wilcoxon_greater(paper_mae - mae_vals)
        p2 = wilcoxon_greater(r2_vals - paper_r2)
        p_mae.append(p1)
        p_r2.append(p2)
        rows.append(
            {
                "dataset": ds,
                "paper_mae": paper_mae,
                "paper_r2": paper_r2,
                "our_mae_mean": float(np.mean(mae_vals)),
                "our_r2_mean": float(np.mean(r2_vals)),
                "delta_mae": float(paper_mae - np.mean(mae_vals)),
                "delta_r2": float(np.mean(r2_vals) - paper_r2),
                "p_mae": p1,
                "p_r2": p2,
            }
        )
    out = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    out["p_adj_mae"] = bh_fdr(p_mae)
    out["p_adj_r2"] = bh_fdr(p_r2)
    out["win_both"] = (
        (out["our_mae_mean"] < out["paper_mae"])
        & (out["our_r2_mean"] > out["paper_r2"])
        & (out["p_adj_mae"] < fdr_q)
        & (out["p_adj_r2"] < fdr_q)
    )
    return out
