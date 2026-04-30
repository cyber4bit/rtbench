from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import mean_std_ci95


def _markdown_table(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        items = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                items.append(format(v, float_fmt))
            else:
                items.append(str(v))
        lines.append("| " + " | ".join(items) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_avg_mae: float,
    baseline_avg_r2: float,
    success_win_required: int,
    feature_importance_failed: dict[str, dict[str, float]],
) -> None:
    rows = []
    for ds in sorted(per_seed["dataset"].unique()):
        cur = per_seed.loc[per_seed["dataset"] == ds]
        mae_mean, mae_std, mae_l, mae_h = mean_std_ci95(cur["mae"].to_numpy())
        r2_mean, r2_std, r2_l, r2_h = mean_std_ci95(cur["r2"].to_numpy())
        rows.append(
            {
                "dataset": ds,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mae_ci95_low": mae_l,
                "mae_ci95_high": mae_h,
                "r2_mean": r2_mean,
                "r2_std": r2_std,
                "r2_ci95_low": r2_l,
                "r2_ci95_high": r2_h,
            }
        )
    ds_stats = pd.DataFrame(rows)
    avg_mae = float(ds_stats["mae_mean"].mean())
    avg_r2 = float(ds_stats["r2_mean"].mean())
    wins = int(summary["win_both"].sum())
    n_datasets = int(summary["dataset"].nunique()) if ("dataset" in summary.columns) else int(ds_stats.shape[0])
    success = (avg_mae < baseline_avg_mae) and (avg_r2 > baseline_avg_r2) and (wins >= success_win_required)

    lines = []
    lines.append("# RepoRT 14+14 Benchmark Report")
    lines.append("")
    lines.append("## Final Verdict")
    lines.append(f"- Success criteria met: **{success}**")
    lines.append(f"- Avg MAE (ours vs paper): **{avg_mae:.4f} vs {baseline_avg_mae:.4f}**")
    lines.append(f"- Avg R2 (ours vs paper): **{avg_r2:.4f} vs {baseline_avg_r2:.4f}**")
    lines.append(
        f"- Win-both datasets (FDR-filtered): **{wins} / {n_datasets}** (required >= {success_win_required})"
    )
    lines.append("")
    lines.append("## Dataset-Level Metrics (mean/std/95%CI)")
    lines.append(_markdown_table(ds_stats, float_fmt=".5f"))
    lines.append("")
    lines.append("## Paper Comparison + Significance")
    lines.append(_markdown_table(summary, float_fmt=".6f"))
    lines.append("")
    lines.append("## Failure Error Decomposition (Feature Groups)")
    failed = summary.loc[~summary["win_both"], "dataset"].tolist()
    if not failed:
        lines.append("- No failed datasets under the win-both criterion.")
    else:
        for ds in failed:
            imp = feature_importance_failed.get(ds, {})
            if not imp:
                lines.append(f"- {ds}: feature importances unavailable (top model without tree importances).")
                continue
            pairs = ", ".join([f"{k}={v:.3f}" for k, v in sorted(imp.items())])
            lines.append(f"- {ds}: {pairs}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
