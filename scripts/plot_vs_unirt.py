from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score


METHOD_ORDER = ("Ours", "Uni-RT", "MDL-TL", "DeepGCN-RT", "GNN-TL")
METRICS = ("mae", "medae", "mre", "r2")


def _normalise_method(value: Any) -> str:
    text = str(value).strip()
    return {
        "GNN-RT": "GNN-TL",
        "GNN RT": "GNN-TL",
        "DeepGCN RT": "DeepGCN-RT",
        "Uni RT": "Uni-RT",
        "MDL TL": "MDL-TL",
    }.get(text, text)


def _load_baseline(path: str | Path, *, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"dataset", "method", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Baseline CSV is missing columns: {missing}")
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df["method"] = df["method"].map(_normalise_method)
    if "mode" in df.columns:
        df = df.loc[df["mode"].astype(str).str.lower() == mode.lower()].copy()
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df.dropna(subset=["mae", "mre", "r2"]).reset_index(drop=True)


def _load_per_seed(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"dataset", "seed", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"per_seed CSV is missing columns: {missing}")
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df.dropna(subset=list(METRICS)).reset_index(drop=True)


def _predictions_from_run_dir(run_dir: Path) -> pd.DataFrame:
    rows = sorted((run_dir / "predictions").glob("*/seed_*.csv"))
    frames = []
    for path in rows:
        df = pd.read_csv(path, encoding="utf-8")
        if not {"y_true_sec", "y_pred_sec"}.issubset(df.columns):
            continue
        dataset = path.parent.name.zfill(4)
        match = re.search(r"seed_(\d+)", path.stem)
        seed = int(match.group(1)) if match else -1
        cur = df[["y_true_sec", "y_pred_sec"]].copy()
        cur["dataset"] = dataset
        cur["seed"] = seed
        cur["method"] = "Ours"
        frames.append(cur)
    if not frames:
        raise ValueError(f"No prediction CSVs found under {run_dir / 'predictions'}")
    return pd.concat(frames, ignore_index=True)


def _read_prediction_csv(path: Path, *, method: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    rename_map = {}
    for true_col in ("y_true_sec", "y_true", "true", "rt_true", "experimental_rt"):
        if true_col in df.columns:
            rename_map[true_col] = "y_true_sec"
            break
    for pred_col in ("y_pred_sec", "y_pred", "pred", "rt_pred", "predicted_rt"):
        if pred_col in df.columns:
            rename_map[pred_col] = "y_pred_sec"
            break
    df = df.rename(columns=rename_map)
    if not {"y_true_sec", "y_pred_sec"}.issubset(df.columns):
        raise ValueError(f"Prediction CSV lacks true/pred columns: {path}")
    out = df[["y_true_sec", "y_pred_sec"]].copy()
    out["method"] = method
    out["dataset"] = str(df["dataset"].iloc[0]).zfill(4) if "dataset" in df.columns and len(df) else ""
    out["seed"] = int(df["seed"].iloc[0]) if "seed" in df.columns and len(df) else -1
    return out


def _predictions_from_extra_specs(specs: list[str]) -> pd.DataFrame:
    frames = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --extra-predictions spec {spec!r}; expected METHOD=glob")
        method, pattern = spec.split("=", 1)
        method = method.strip()
        paths = [Path(p) for p in sorted(glob.glob(pattern))]
        if not paths:
            raise ValueError(f"No prediction CSVs matched: {pattern}")
        for path in paths:
            frames.append(_read_prediction_csv(path, method=method))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3 or np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return np.ones_like(x, dtype=np.float64)
    try:
        return stats.gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))
    except Exception:
        return np.ones_like(x, dtype=np.float64)


def plot_predicted_vs_true(predictions: pd.DataFrame, out_path: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in set(predictions["method"])]
    if not methods:
        methods = sorted(predictions["method"].unique())
    n = len(methods)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows), squeeze=False)
    for ax, method in zip(axes.ravel(), methods):
        cur = predictions.loc[predictions["method"] == method].dropna(subset=["y_true_sec", "y_pred_sec"])
        x = cur["y_true_sec"].to_numpy(dtype=np.float64)
        y = cur["y_pred_sec"].to_numpy(dtype=np.float64)
        z = _density(x, y)
        order = np.argsort(z)
        x, y, z = x[order], y[order], z[order]
        ax.scatter(x, y, c=z, s=9, cmap="viridis", alpha=0.80, linewidths=0)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        ax.plot([lo, hi], [lo, hi], color="#202020", linewidth=1.0)
        r2 = r2_score(x, y) if len(x) >= 2 else float("nan")
        ax.text(0.04, 0.96, f"R2 = {r2:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=10)
        ax.set_title(method)
        ax.set_xlabel("Experimental RT (s)")
        ax.set_ylabel("Predicted RT (s)")
    for ax in axes.ravel()[len(methods) :]:
        ax.axis("off")
    fig.suptitle("Predicted vs Experimental RT", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _ours_dataset_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    mean_df = per_seed.groupby("dataset", as_index=False)[list(METRICS)].mean()
    std_df = per_seed.groupby("dataset", as_index=False)[list(METRICS)].std(ddof=1).fillna(0.0)
    mean_df["method"] = "Ours"
    for metric in METRICS:
        mean_df[f"{metric}_std"] = std_df[metric]
    return mean_df


def plot_mre_by_dataset(per_seed: pd.DataFrame, baseline: pd.DataFrame, out_path: Path) -> None:
    ours = _ours_dataset_stats(per_seed)
    datasets = sorted(set(ours["dataset"]).intersection(set(baseline["dataset"])))
    if not datasets:
        datasets = sorted(ours["dataset"].unique())
    baseline = baseline.loc[baseline["dataset"].isin(datasets)].copy()
    ours = ours.loc[ours["dataset"].isin(datasets)].copy()
    combined = pd.concat([ours, baseline], ignore_index=True, sort=False)
    methods = [m for m in METHOD_ORDER if m in set(combined["method"])]
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.82 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.9), 5.2))
    for i, method in enumerate(methods):
        cur = combined.loc[combined["method"] == method].set_index("dataset")
        vals = np.array([float(cur.loc[d, "mre"]) if d in cur.index else np.nan for d in datasets], dtype=np.float64)
        yerr = np.array(
            [float(cur.loc[d, "mre_std"]) if d in cur.index and "mre_std" in cur.columns else 0.0 for d in datasets],
            dtype=np.float64,
        )
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=method, yerr=yerr, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("MRE")
    ax.set_xlabel("Dataset ID")
    ax.set_title("Dataset-Level MRE Comparison")
    ax.legend(ncol=min(len(methods), 5), frameon=False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_summary_boxplots(per_seed: pd.DataFrame, baseline: pd.DataFrame, out_path: Path) -> None:
    ours = _ours_dataset_stats(per_seed)
    datasets = sorted(set(ours["dataset"]).intersection(set(baseline["dataset"])))
    baseline = baseline.loc[baseline["dataset"].isin(datasets)].copy()
    ours = ours.loc[ours["dataset"].isin(datasets)].copy()
    combined = pd.concat([ours[["dataset", "method", *METRICS]], baseline[["dataset", "method", *METRICS]]], ignore_index=True)
    methods = [m for m in METHOD_ORDER if m in set(combined["method"])]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), squeeze=False)
    for ax, metric in zip(axes.ravel(), METRICS):
        data = [combined.loc[combined["method"] == method, metric].dropna().to_numpy(dtype=np.float64) for method in methods]
        ax.boxplot(data, tick_labels=methods, showmeans=True, patch_artist=True)
        ax.set_title(metric.upper() if metric != "medae" else "MedAE")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.7)
    fig.suptitle("Cross-Dataset Performance Distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rtbench results against Uni-RT supplementary baselines.")
    parser.add_argument("--run-dir", required=True, help="rtbench output directory")
    parser.add_argument("--baseline", default="data/baseline/unirt_sota_28.csv", help="Uni-RT long-format baseline CSV")
    parser.add_argument("--mode", default="RPLC", help="Chromatographic mode to filter in baseline CSV")
    parser.add_argument("--out-dir", default="", help="Output figure directory; defaults to <run-dir>/figures")
    parser.add_argument(
        "--extra-predictions",
        action="append",
        default=[],
        help="Optional METHOD=glob CSV predictions to add to predicted-vs-true panels",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = _load_per_seed(run_dir / "metrics" / "per_seed.csv")
    baseline = _load_baseline(args.baseline, mode=args.mode)
    predictions = _predictions_from_run_dir(run_dir)
    extra_predictions = _predictions_from_extra_specs(args.extra_predictions)
    if not extra_predictions.empty:
        predictions = pd.concat([predictions, extra_predictions], ignore_index=True, sort=False)

    plot_predicted_vs_true(predictions, out_dir / "predicted_vs_true_rt.png")
    plot_mre_by_dataset(per_seed, baseline, out_dir / "mre_by_dataset.png")
    plot_summary_boxplots(per_seed, baseline, out_dir / "summary_boxplots.png")


if __name__ == "__main__":
    main()
