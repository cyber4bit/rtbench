from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .config import parse_seed_range, resolve_config


METRICS = ("mae", "medae", "mre", "r2")
METHOD_ORDER = ("Uni-RT", "MDL-TL", "DeepGCN-RT", "GNN-TL")
DEFAULT_METHOD_N_MODELS = {
    "Uni-RT": 1,
    "MDL-TL": 17,
    "DeepGCN-RT": 14,
    "GNN-TL": 14,
}
DEFAULT_EXPECTED_SEEDS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
PUBLISHED_UNIRT_SUMMARY = {
    "RPLC": {
        "Uni-RT": {"r2": (0.91, 0.08), "mae": (25.55, 16.32), "medae": (14.31, 11.17), "mre": (0.12, 0.07)},
        "MDL-TL": {"r2": (0.90, 0.09), "mae": (27.14, 20.34), "medae": (16.67, 13.90), "mre": (0.14, 0.08)},
        "DeepGCN-RT": {"r2": (0.90, 0.08), "mae": (26.05, 13.52), "medae": (14.48, 7.70), "mre": (0.16, 0.05)},
        "GNN-TL": {"r2": (0.65, 0.20), "mae": (53.61, 22.73), "medae": (27.94, 14.38), "mre": (0.33, 0.11)},
    },
    "HILIC": {
        "Uni-RT": {"r2": (0.83, 0.15), "mae": (48.09, 20.61), "medae": (27.61, 15.31), "mre": (0.19, 0.07)},
        "MDL-TL": {"r2": (0.67, 0.24), "mae": (49.97, 14.88), "medae": (28.82, 10.96), "mre": (0.20, 0.15)},
        "DeepGCN-RT": {"r2": (0.51, 0.33), "mae": (63.67, 25.47), "medae": (42.24, 18.83), "mre": (0.24, 0.11)},
        "GNN-TL": {"r2": (0.02, 0.70), "mae": (117.90, 54.34), "medae": (82.53, 40.89), "mre": (0.33, 0.12)},
    },
}


def _metric_label(metric: str) -> str:
    return {"mae": "MAE", "medae": "MedAE", "mre": "MRE", "r2": "R2"}[metric]


def _metric_fmt(metric: str) -> str:
    return ".4f" if metric in {"mre", "r2"} else ".2f"


def _fmt_num(value: Any, metric: str = "mae") -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return format(x, _metric_fmt(metric))


def _fmt_avg_std(avg: Any, std: Any, metric: str) -> str:
    a = _fmt_num(avg, metric)
    s = _fmt_num(std, metric)
    if not a or not s:
        return ""
    return f"{a} +/- {s}"


def _as_float(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return float(x) if np.isfinite(x) else float("nan")


def _sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _normalise_method(value: Any) -> str:
    text = str(value).strip()
    aliases = {
        "GNN-RT": "GNN-TL",
        "GNN RT": "GNN-TL",
        "DeepGCN RT": "DeepGCN-RT",
        "DeepGCN": "DeepGCN-RT",
        "MDL TL": "MDL-TL",
        "Uni RT": "Uni-RT",
    }
    return aliases.get(text, text)


def load_per_seed_metrics(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"dataset", "seed", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"per-seed metrics file is missing columns: {missing}")
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df["seed"] = df["seed"].astype(int)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df.dropna(subset=list(METRICS)).copy()


def load_unirt_baseline(path: str | Path, *, mode: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "dataset" not in df.columns:
        raise ValueError("Uni-RT baseline CSV must contain a dataset column")

    if {"method", "mae", "medae", "mre", "r2"}.issubset(df.columns):
        out = df.copy()
    elif {"paper_mae", "paper_r2"}.issubset(df.columns):
        out = df.rename(columns={"paper_mae": "mae", "paper_r2": "r2"}).copy()
        out["method"] = "Uni-RT"
        if "paper_medae" in out.columns:
            out["medae"] = out["paper_medae"]
        if "paper_mre" in out.columns:
            out["mre"] = out["paper_mre"]
    else:
        raise ValueError(
            "Uni-RT baseline CSV must be long-format dataset/method/mae/medae/mre/r2 "
            "or compact paper_mae/paper_r2 format"
        )

    out["dataset"] = out["dataset"].astype(str).str.strip()
    out = out.loc[out["dataset"].str.fullmatch(r"\d+")].copy()
    out["dataset"] = out["dataset"].str.zfill(4)
    out["method"] = out["method"].map(_normalise_method)
    if mode and "mode" in out.columns:
        out = out.loc[out["mode"].astype(str).str.strip().str.lower() == str(mode).strip().lower()].copy()
    for metric in METRICS:
        if metric not in out.columns:
            out[metric] = np.nan
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    if "n_model" in out.columns:
        out["n_model"] = pd.to_numeric(out["n_model"], errors="coerce")
    return out.dropna(subset=["mae", "r2"]).reset_index(drop=True)


def per_dataset_stats(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, cur in per_seed.groupby("dataset", sort=True):
        row: dict[str, Any] = {"dataset": dataset, "n_runs": int(cur["seed"].nunique())}
        for metric in METRICS:
            values = cur[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.allclose(diff, 0.0):
        return 1.0
    if np.allclose(diff, diff[0]):
        return 0.0
    res = stats.ttest_rel(a, b, alternative="two-sided", nan_policy="omit")
    return float(res.pvalue) if np.isfinite(res.pvalue) else float("nan")


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if np.allclose(diff, 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _one_sample_ttest(values: np.ndarray, baseline: float) -> float:
    if len(values) < 2:
        return float("nan")
    vals = np.asarray(values, dtype=np.float64)
    if np.allclose(vals - float(baseline), 0.0):
        return 1.0
    if np.allclose(vals, vals[0]):
        return 0.0
    res = stats.ttest_1samp(vals, popmean=float(baseline), alternative="two-sided", nan_policy="omit")
    return float(res.pvalue) if np.isfinite(res.pvalue) else float("nan")


def _method_n_model(baseline: pd.DataFrame, method: str) -> int:
    if "n_model" in baseline.columns:
        vals = baseline.loc[baseline["method"] == method, "n_model"].dropna().to_numpy()
        if len(vals):
            return int(vals[0])
    return int(DEFAULT_METHOD_N_MODELS.get(method, 0))


def build_cross_dataset_summary(
    dataset_stats: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    ours_label: str,
    n_model: int,
    mode: str = "RPLC",
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    run_datasets = set(dataset_stats["dataset"].astype(str))
    baseline = baseline.loc[baseline["dataset"].isin(run_datasets)].copy()
    use_published = len(run_datasets) == 14 and all(
        set(baseline.loc[baseline["method"] == method, "dataset"].astype(str)) == run_datasets for method in METHOD_ORDER
    )

    ours_display_label = f"{ours_label} (avg +/- Std)" if "avg" not in ours_label.lower() else ours_label

    for metric in ("r2", "mae", "medae", "mre"):
        row: dict[str, str] = {"Metrics": f"{_metric_label(metric)} (average +/- Std)"}
        ours_values = dataset_stats[f"{metric}_mean"].to_numpy(dtype=np.float64)
        row[ours_display_label] = _fmt_avg_std(np.mean(ours_values), _sample_std(ours_values), metric)
        for method in METHOD_ORDER:
            published = PUBLISHED_UNIRT_SUMMARY.get(str(mode).upper(), {}).get(method, {}).get(metric)
            if use_published and published is not None:
                row[method] = _fmt_avg_std(published[0], published[1], metric)
            else:
                cur = baseline.loc[baseline["method"] == method, metric].dropna().to_numpy(dtype=np.float64)
                row[method] = _fmt_avg_std(np.mean(cur), _sample_std(cur), metric) if len(cur) else ""
        rows.append(row)

    n_row: dict[str, str] = {"Metrics": "nModel", ours_display_label: str(int(n_model))}
    for method in METHOD_ORDER:
        n_row[method] = str(_method_n_model(baseline, method))
    rows.append(n_row)
    return pd.DataFrame(rows)


def build_dataset_comparison(per_seed: pd.DataFrame, dataset_stats: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    unirt = baseline.loc[baseline["method"] == "Uni-RT"].set_index("dataset")
    rows: list[dict[str, Any]] = []
    for _, ds_row in dataset_stats.iterrows():
        dataset = str(ds_row["dataset"])
        if dataset not in unirt.index:
            continue
        base = unirt.loc[dataset]
        cur = per_seed.loc[per_seed["dataset"] == dataset]
        p_mae = _one_sample_ttest(cur["mae"].to_numpy(dtype=np.float64), float(base["mae"]))
        p_r2 = _one_sample_ttest(cur["r2"].to_numpy(dtype=np.float64), float(base["r2"]))
        rows.append(
            {
                "Dataset": dataset,
                "Ours MAE (10-run avg)": _fmt_num(ds_row["mae_mean"], "mae"),
                "Uni-RT MAE": _fmt_num(base["mae"], "mae"),
                "Delta MAE (Ours-UniRT)": _fmt_num(float(ds_row["mae_mean"]) - float(base["mae"]), "mae"),
                "Ours R2 (10-run avg)": _fmt_num(ds_row["r2_mean"], "r2"),
                "Uni-RT R2": _fmt_num(base["r2"], "r2"),
                "Delta R2 (Ours-UniRT)": _fmt_num(float(ds_row["r2_mean"]) - float(base["r2"]), "r2"),
                "p-value (MAE)": _fmt_num(p_mae, "r2"),
                "p-value (R2)": _fmt_num(p_r2, "r2"),
            }
        )
    return pd.DataFrame(rows)


def build_significance_summary(dataset_stats: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    unirt = baseline.loc[baseline["method"] == "Uni-RT", ["dataset", *METRICS]].copy()
    merged = dataset_stats.merge(unirt, on="dataset", suffixes=("_ours", "_unirt"))
    rows: list[dict[str, str]] = []
    for metric in ("mae", "medae", "mre", "r2"):
        ours = merged[f"{metric}_mean"].to_numpy(dtype=np.float64)
        base = merged[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Metric": _metric_label(metric),
                "paired t-test p": _fmt_num(_paired_ttest(ours, base), "r2"),
                "Wilcoxon p": _fmt_num(_paired_wilcoxon(ours, base), "r2"),
                "n paired datasets": str(int(len(merged))),
            }
        )
    return pd.DataFrame(rows)


def _published_metric(mode: str, metric: str) -> tuple[float, float] | None:
    return PUBLISHED_UNIRT_SUMMARY.get(str(mode).upper(), {}).get("Uni-RT", {}).get(metric)


def _overall_assessment_lines(
    dataset_stats: pd.DataFrame,
    significance: pd.DataFrame,
    *,
    mode: str,
) -> list[str]:
    if dataset_stats.empty:
        return ["- No dataset-level results were available for overall assessment."]
    sig = significance.set_index("Metric") if not significance.empty and "Metric" in significance.columns else pd.DataFrame()

    def pvals(metric_label: str) -> tuple[float, float]:
        if sig.empty or metric_label not in sig.index:
            return float("nan"), float("nan")
        row = sig.loc[metric_label]
        return _as_float(row.get("paired t-test p")), _as_float(row.get("Wilcoxon p"))

    lines: list[str] = []
    for metric, better_word, comparator in (
        ("mae", "lower", lambda ours, base: ours < base),
        ("r2", "higher", lambda ours, base: ours > base),
    ):
        published = _published_metric(mode, metric)
        if published is None:
            continue
        ours_value = float(dataset_stats[f"{metric}_mean"].mean())
        base_value = float(published[0])
        t_p, w_p = pvals(_metric_label(metric))
        is_better = bool(comparator(ours_value, base_value))
        p_text = (
            f"paired t-test p={_fmt_num(t_p, 'r2')}, Wilcoxon p={_fmt_num(w_p, 'r2')}"
            if np.isfinite(t_p) or np.isfinite(w_p)
            else "cross-dataset p-value unavailable"
        )
        significant = bool((np.isfinite(t_p) and t_p < 0.05) or (np.isfinite(w_p) and w_p < 0.05))
        lines.append(
            f"- {_metric_label(metric)}: ours {_fmt_num(ours_value, metric)} vs Uni-RT {_fmt_num(base_value, metric)} "
            f"({better_word} is better: {'yes' if is_better else 'no'}; {p_text}; "
            f"{'significant at 0.05' if significant else 'not significant at 0.05'})."
        )

    published_mae = _published_metric(mode, "mae")
    published_r2 = _published_metric(mode, "r2")
    if published_mae and published_r2:
        ours_mae = float(dataset_stats["mae_mean"].mean())
        ours_r2 = float(dataset_stats["r2_mean"].mean())
        both_better = ours_mae < float(published_mae[0]) and ours_r2 > float(published_r2[0])
        lines.append(
            "- Overall conclusion: "
            + (
                "both average MAE and average R2 beat Uni-RT; check statistical significance before claiming superiority."
                if both_better
                else "do not claim overall superiority over Uni-RT because at least one core average metric is worse."
            )
        )
    lines.append(
        "- Fairness caveat: our result uses 14 per-dataset models, whereas Uni-RT reports one unified model for the chromatographic mode."
    )
    return lines


def build_per_dataset_markdown_table(dataset_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in dataset_stats.iterrows():
        rows.append(
            {
                "Dataset": str(row["dataset"]),
                "n_runs": str(int(row["n_runs"])),
                "MAE_mean +/- std": _fmt_avg_std(row["mae_mean"], row["mae_std"], "mae"),
                "MedAE_mean +/- std": _fmt_avg_std(row["medae_mean"], row["medae_std"], "medae"),
                "MRE_mean +/- std": _fmt_avg_std(row["mre_mean"], row["mre_std"], "mre"),
                "R2_mean +/- std": _fmt_avg_std(row["r2_mean"], row["r2_std"], "r2"),
            }
        )
    return pd.DataFrame(rows)


def _load_optional_diagnostics(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype={"Dataset": str, "dataset": str}, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    if "dataset" in df.columns and "Dataset" not in df.columns:
        df = df.rename(columns={"dataset": "Dataset"})
    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].astype(str).str.zfill(4)
    return df


def _diagnostic_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def _build_diagnosis_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty or "Dataset" not in diagnostics.columns:
        return pd.DataFrame()
    rows: list[dict[str, str]] = []
    for _, row in diagnostics.iterrows():
        flags = []
        if _diagnostic_bool(row.get("small_dataset_lt100", False)):
            flags.append("n<100")
        if _diagnostic_bool(row.get("low_source_overlap", False)):
            flags.append("low source overlap")
        if _diagnostic_bool(row.get("high_structure_diversity", False)):
            flags.append("high structure diversity")
        if _diagnostic_bool(row.get("chromatographic_mismatch", False)):
            flags.append("chromatographic mismatch")
        if _diagnostic_bool(row.get("high_outlier_rate", False)):
            flags.append("high outlier rate")
        if not flags:
            flags.append("no quantified flag")
        rows.append(
            {
                "Dataset": str(row["Dataset"]).zfill(4),
                "Status": str(row.get("status", row.get("win_loss", ""))),
                "n_rows": _fmt_num(row.get("n_rows"), "mae"),
                "source_overlap": _fmt_num(row.get("source_overlap_rate"), "r2"),
                "fp_diversity": _fmt_num(row.get("fingerprint_diversity"), "r2"),
                "nearest_cp_dist": _fmt_num(row.get("nearest_cp_zdist"), "r2"),
                "outlier_rate": _fmt_num(row.get("outlier_rate"), "r2"),
                "Diagnostic flags": ", ".join(flags),
            }
        )
    return pd.DataFrame(rows)


def _diagnosis_lines(diagnostics: pd.DataFrame) -> list[str]:
    if diagnostics.empty or "Dataset" not in diagnostics.columns:
        return ["No diagnostics CSV was available for dataset-level failure analysis."]
    table = _build_diagnosis_table(diagnostics)
    lines = [_markdown_table(table)]
    losing = diagnostics.copy()
    if "win_both" in losing.columns:
        losing = losing.loc[~losing["win_both"].map(_diagnostic_bool)]
    if losing.empty:
        lines.extend(["", "- All datasets win on both MAE and R2; no losing-set diagnosis is required."])
        return lines

    reason_counts = {
        "n<100": int(losing.get("small_dataset_lt100", pd.Series(dtype=bool)).map(_diagnostic_bool).sum()),
        "low source overlap": int(losing.get("low_source_overlap", pd.Series(dtype=bool)).map(_diagnostic_bool).sum()),
        "high structure diversity": int(losing.get("high_structure_diversity", pd.Series(dtype=bool)).map(_diagnostic_bool).sum()),
        "chromatographic mismatch": int(losing.get("chromatographic_mismatch", pd.Series(dtype=bool)).map(_diagnostic_bool).sum()),
        "high outlier rate": int(losing.get("high_outlier_rate", pd.Series(dtype=bool)).map(_diagnostic_bool).sum()),
    }
    lines.append("")
    lines.append(
        "- Losing-dataset quantified flags: "
        + ", ".join(f"{name}={count}" for name, count in reason_counts.items())
        + "."
    )
    lines.append(
        "- Interpret these as diagnostic signals, not causal proof; structure diversity and chromatographic mismatch are derived from RepoRT fingerprints/metadata, while Uni-RT per-seed predictions are unavailable."
    )
    return lines


def _fairness_lines(
    per_seed: pd.DataFrame,
    dataset_stats: pd.DataFrame,
    *,
    expected_seeds: list[int],
    split_cfg: dict[str, Any] | None,
    stats_test: str,
    fdr_correction: str,
    n_model: int,
) -> list[str]:
    observed = sorted(int(x) for x in per_seed["seed"].unique())
    seed_ok = observed == sorted(int(x) for x in expected_seeds)
    split_cfg = split_cfg or {}
    train = float(split_cfg.get("train", float("nan")))
    val = float(split_cfg.get("val", float("nan")))
    test = float(split_cfg.get("test", float("nan")))
    split_ok = np.allclose([train, val, test], [0.8, 0.1, 0.1], atol=1e-8)
    full_run_count = int((dataset_stats["n_runs"] == len(expected_seeds)).sum()) if not dataset_stats.empty else 0
    return [
        f"- Multiple-run averaging: {'PASS' if seed_ok else 'CHECK'}; observed seeds={observed}, expected={expected_seeds}.",
        f"- Complete 10-run datasets: {full_run_count}/{int(len(dataset_stats))}.",
        f"- Split ratio: {'PASS' if split_ok else 'CHECK'}; train/val/test={train:.2f}/{val:.2f}/{test:.2f}.",
        "- Metric definitions: MAE, MedAE, MRE=mean(abs(y-yhat)/max(abs(y), eps)), R2 are computed on held-out test rows.",
        "- MRE unit: reported as the same decimal ratio used in Uni-RT tables, so 0.12 corresponds to 12%.",
        f"- Statistical testing: dataset-level paired t-test and Wilcoxon; configured test={stats_test}, correction={fdr_correction}.",
        f"- Model count: ours nModel={int(n_model)}; compare against Uni-RT nModel=1 per chromatographic mode.",
    ]


def _win_loss_lines(dataset_stats: pd.DataFrame, baseline: pd.DataFrame, *, mode: str) -> list[str]:
    unirt = baseline.loc[baseline["method"] == "Uni-RT", ["dataset", "mae", "r2"]].copy()
    merged = dataset_stats.merge(unirt, on="dataset", suffixes=("_ours", "_unirt"))
    if merged.empty:
        return ["- No overlapping Uni-RT dataset baselines were available for win/loss accounting."]
    mae_wins = int((merged["mae_mean"] < merged["mae"]).sum())
    r2_wins = int((merged["r2_mean"] > merged["r2"]).sum())
    ours_mae = float(merged["mae_mean"].mean())
    ours_r2 = float(merged["r2_mean"].mean())
    published_mae = _published_metric(mode, "mae")
    published_r2 = _published_metric(mode, "r2")
    unirt_mae = float(published_mae[0]) if published_mae else float(merged["mae"].mean())
    unirt_r2 = float(published_r2[0]) if published_r2 else float(merged["r2"].mean())
    return [
        f"- MAE wins: {mae_wins}/{len(merged)} datasets where our 10-run mean MAE is lower than Uni-RT.",
        f"- R2 wins: {r2_wins}/{len(merged)} datasets where our 10-run mean R2 is higher than Uni-RT.",
        f"- Overall MAE: ours {_fmt_num(ours_mae, 'mae')} vs Uni-RT {_fmt_num(unirt_mae, 'mae')} "
        f"({'better' if ours_mae < unirt_mae else 'not better'}).",
        f"- Overall R2: ours {_fmt_num(ours_r2, 'r2')} vs Uni-RT {_fmt_num(unirt_r2, 'r2')} "
        f"({'better' if ours_r2 > unirt_r2 else 'not better'}).",
    ]


def _visualization_lines(out_path: Path) -> list[str]:
    fig_dir = out_path.parent / "figures"
    figures = [
        ("Predicted-vs-true scatter", fig_dir / "predicted_vs_true_rt.png"),
        ("MRE bar chart", fig_dir / "mre_by_dataset.png"),
        ("Summary boxplot", fig_dir / "summary_boxplots.png"),
    ]
    lines = []
    for label, path in figures:
        if path.exists():
            lines.append(f"- {label}: `{path.as_posix()}`")
        else:
            lines.append(f"- {label}: missing (`{path.as_posix()}`)")
    return lines


def write_unirt_report(
    *,
    out_path: str | Path,
    per_seed: pd.DataFrame | str | Path,
    baseline_csv: str | Path,
    mode: str = "RPLC",
    ours_label: str = "Ours",
    n_model: int,
    expected_seeds: list[int] | None = None,
    split_cfg: dict[str, Any] | None = None,
    stats_test: str = "wilcoxon_signed_rank",
    fdr_correction: str = "bh_fdr",
    output_dir: str | Path | None = None,
    diagnostics_csv: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    per_seed_df = load_per_seed_metrics(per_seed) if isinstance(per_seed, (str, Path)) else per_seed.copy()
    per_seed_df["dataset"] = per_seed_df["dataset"].astype(str).str.zfill(4)
    baseline_df = load_unirt_baseline(baseline_csv, mode=mode)
    dataset_stats = per_dataset_stats(per_seed_df)
    run_dataset_set = set(dataset_stats["dataset"].astype(str))
    baseline_df = baseline_df.loc[baseline_df["dataset"].isin(run_dataset_set)].copy()

    per_dataset_table = build_per_dataset_markdown_table(dataset_stats)
    cross_summary = build_cross_dataset_summary(
        dataset_stats=dataset_stats,
        baseline=baseline_df,
        ours_label=ours_label,
        n_model=n_model,
        mode=mode,
    )
    dataset_comparison = build_dataset_comparison(per_seed_df, dataset_stats, baseline_df)
    significance = build_significance_summary(dataset_stats, baseline_df)
    diagnostics_path = diagnostics_csv
    if diagnostics_path is None and output_dir is not None:
        diagnostics_path = Path(output_dir) / "diagnostics_vs_unirt.csv"
    diagnostics = _load_optional_diagnostics(diagnostics_path)

    expected = [int(x) for x in (expected_seeds or DEFAULT_EXPECTED_SEEDS)]
    lines = [
        f"# Uni-RT-Aligned {mode.upper()} Benchmark Report",
        "",
        "## Evaluation Protocol",
        *(
            _fairness_lines(
                per_seed_df,
                dataset_stats,
                expected_seeds=expected,
                split_cfg=split_cfg,
                stats_test=stats_test,
                fdr_correction=fdr_correction,
                n_model=n_model,
            )
        ),
        "",
        "## Per-Dataset Performance Table",
        _markdown_table(per_dataset_table),
        "",
        "## Cross-Dataset Summary",
        _markdown_table(cross_summary),
        "",
        "## Overall Assessment",
        *_overall_assessment_lines(dataset_stats, significance, mode=mode),
        "",
        "## Dataset-Level Comparison",
        _markdown_table(dataset_comparison)
        if not dataset_comparison.empty
        else "No overlapping Uni-RT per-dataset baselines were available.",
        "",
        "## Dataset-Level Diagnosis",
        *_diagnosis_lines(diagnostics),
        "",
        "## Statistical Tests",
        _markdown_table(significance) if not significance.empty else "No overlapping datasets for statistical tests.",
        "",
        "## Win/Loss Summary",
        *_win_loss_lines(dataset_stats, baseline_df, mode=mode),
        "",
        "## Visualizations",
        *_visualization_lines(Path(out_path)),
        "",
        "## Notes",
        "- Cross-dataset average and Std are computed over dataset-level 10-run means, matching the Uni-RT Table 1/2 convention.",
        "- For a complete 14-dataset RPLC/HILIC run, baseline summary columns use the published Uni-RT Table 1/2 values.",
        "- Dataset-level p-values use one-sample tests against published Uni-RT dataset means because per-seed Uni-RT predictions are not available in the supplementary tables.",
        "- Negative Delta MAE and positive Delta R2 indicate improvement over Uni-RT.",
        "- Diagnostic flags are computed from the evaluated RepoRT data and are used only to explain failure modes; they are not used to select seeds or datasets.",
        "",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset_stats.to_csv(out_dir / "per_dataset_10run_stats.csv", index=False, encoding="utf-8")
        cross_summary.to_csv(out_dir / "cross_dataset_summary_vs_unirt.csv", index=False, encoding="utf-8")
        dataset_comparison.to_csv(out_dir / "dataset_level_comparison_vs_unirt.csv", index=False, encoding="utf-8")
        significance.to_csv(out_dir / "significance_vs_unirt.csv", index=False, encoding="utf-8")

    return {
        "dataset_stats": dataset_stats,
        "cross_summary": cross_summary,
        "dataset_comparison": dataset_comparison,
        "significance": significance,
        "diagnostics": diagnostics,
    }


def _parse_seed_list(text: str) -> list[int]:
    return parse_seed_range(text) if text else list(DEFAULT_EXPECTED_SEEDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Uni-RT-aligned Markdown report from rtbench per-seed metrics.")
    parser.add_argument("--run-dir", default="", help="Run directory containing metrics/per_seed.csv")
    parser.add_argument("--per-seed", default="", help="Path to metrics/per_seed.csv")
    parser.add_argument("--baseline", default="", help="Uni-RT long-format baseline CSV")
    parser.add_argument("--out", default="", help="Output Markdown path")
    parser.add_argument("--config", default="", help="Optional rtbench config used to fill split, seeds, and report defaults")
    parser.add_argument("--mode", default="RPLC", help="Chromatographic mode to filter in the Uni-RT baseline CSV")
    parser.add_argument("--ours-label", default="Ours")
    parser.add_argument("--n-model", type=int, default=0)
    parser.add_argument("--expected-seeds", default="", help="Expected seeds, e.g. 0,2,4,6,8,10,12,14,16,18")
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = resolve_config(args.config).config

    run_dir = Path(args.run_dir) if args.run_dir else None
    per_seed_path = Path(args.per_seed) if args.per_seed else (run_dir / "metrics" / "per_seed.csv" if run_dir else None)
    if per_seed_path is None:
        raise SystemExit("Provide --per-seed or --run-dir")

    baseline = args.baseline or (str(cfg.metrics.get("unirt_baseline_csv", cfg.data["baseline_csv"])) if cfg else "")
    if not baseline:
        raise SystemExit("Provide --baseline or --config with metrics.unirt_baseline_csv/data.baseline_csv")

    out_path = Path(args.out) if args.out else (run_dir / "report_vs_unirt.md" if run_dir else Path("report_vs_unirt.md"))
    split_cfg = cfg.split if cfg else None
    expected_seeds = _parse_seed_list(args.expected_seeds or (str(cfg.seeds["default"]) if cfg else ""))
    mode = str(cfg.metrics.get("unirt_mode", args.mode)) if cfg else args.mode
    ours_label = str(cfg.metrics.get("ours_label", args.ours_label)) if cfg else args.ours_label
    n_model = int(args.n_model or (cfg.metrics.get("ours_n_model", 0) if cfg else 0) or 0)
    if n_model <= 0:
        per_seed_tmp = load_per_seed_metrics(per_seed_path)
        n_model = int(per_seed_tmp["dataset"].nunique())

    write_unirt_report(
        out_path=out_path,
        per_seed=per_seed_path,
        baseline_csv=baseline,
        mode=mode,
        ours_label=ours_label,
        n_model=n_model,
        expected_seeds=expected_seeds,
        split_cfg=split_cfg,
        stats_test=str(cfg.stats.get("test", "wilcoxon_signed_rank")) if cfg else "wilcoxon_signed_rank",
        fdr_correction=str(cfg.stats.get("correction", "bh_fdr")) if cfg else "bh_fdr",
        output_dir=(run_dir / "metrics") if run_dir else out_path.parent,
    )


if __name__ == "__main__":
    main()
