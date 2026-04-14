from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .archive import _normalize_relpath
from .registry import load_registry


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column].replace("", pd.NA), errors="coerce")


def _resolve_registry_row(df: pd.DataFrame, experiment_id: str) -> pd.Series:
    key = _normalize_relpath(experiment_id)
    direct = df[(df["run_dir"] == key) | (df["experiment_name"] == key)]
    if len(direct) == 1:
        return direct.iloc[0]
    if len(direct) > 1:
        raise ValueError(f"Ambiguous experiment id: {experiment_id}")

    output_root = df[df["output_root"] == key]
    if len(output_root) == 1:
        return output_root.iloc[0]
    if len(output_root) > 1:
        raise ValueError(f"Experiment id matches multiple runs under output root: {experiment_id}")

    raise KeyError(f"Experiment not found in registry: {experiment_id}")


def query_experiments(
    project_root: Path,
    *,
    metric: str = "avg_mae",
    sort: str = "asc",
    top: int = 10,
    status: str | None = None,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    df, _ = load_registry(project_root, registry_path=registry_path)
    if metric not in df.columns:
        raise ValueError(f"Unsupported metric column: {metric}")
    if sort.lower() not in {"asc", "desc"}:
        raise ValueError(f"Unsupported sort order: {sort}")

    view = df.copy()
    if status:
        view = view[view["status"].str.lower() == status.lower()].copy()
    else:
        view = view[view["status"].str.lower() != "failed"].copy()

    sort_values = _coerce_numeric_column(view, metric)
    if sort_values.isna().all():
        view = view[view[metric].astype(str).str.strip() != ""].copy()
        ascending = sort.lower() == "asc"
        view = view.sort_values([metric, "run_date", "run_dir"], ascending=[ascending, False, True])
    else:
        view["_sort_value"] = sort_values
        view = view[view["_sort_value"].notna()].copy()
        ascending = sort.lower() == "asc"
        view = view.sort_values(["_sort_value", "run_date", "run_dir"], ascending=[ascending, False, True])

    columns = [
        "run_dir",
        "status",
        "avg_mae",
        "avg_r2",
        "win_both",
        "dataset_count",
        "seed_count",
        "config_path",
        "archived",
    ]
    out = view.head(max(1, int(top)))
    if "_sort_value" in out.columns:
        out = out.drop(columns="_sort_value")
    return out[columns].reset_index(drop=True)


def _summary_df_for_run(project_root: Path, row: pd.Series) -> pd.DataFrame:
    summary_rel = str(row.get("summary_path", "")).strip()
    if summary_rel:
        summary_path = project_root / summary_rel
    else:
        summary_path = project_root / str(row["run_dir"]) / "metrics" / "summary_vs_paper.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary metrics not found for run: {row['run_dir']}")
    summary = pd.read_csv(summary_path, dtype={"dataset": str}, encoding="utf-8").fillna("")
    if "dataset" in summary.columns:
        summary["dataset"] = summary["dataset"].astype(str).str.zfill(4)
    return summary


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    return int(float(value))


def compare_experiments(
    project_root: Path,
    run_a: str,
    run_b: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    df, _ = load_registry(project_root, registry_path=registry_path)
    row_a = _resolve_registry_row(df, run_a)
    row_b = _resolve_registry_row(df, run_b)

    summary_a = _summary_df_for_run(project_root, row_a).rename(
        columns={
            "our_mae_mean": "our_mae_mean_a",
            "our_r2_mean": "our_r2_mean_a",
            "delta_mae": "delta_mae_a",
            "delta_r2": "delta_r2_a",
            "win_both": "win_both_a",
        }
    )
    summary_b = _summary_df_for_run(project_root, row_b).rename(
        columns={
            "our_mae_mean": "our_mae_mean_b",
            "our_r2_mean": "our_r2_mean_b",
            "delta_mae": "delta_mae_b",
            "delta_r2": "delta_r2_b",
            "win_both": "win_both_b",
        }
    )

    keep_cols_a = [
        c
        for c in ["dataset", "paper_mae", "paper_r2", "our_mae_mean_a", "our_r2_mean_a", "delta_mae_a", "delta_r2_a", "win_both_a"]
        if c in summary_a.columns
    ]
    keep_cols_b = [
        c
        for c in ["dataset", "paper_mae", "paper_r2", "our_mae_mean_b", "our_r2_mean_b", "delta_mae_b", "delta_r2_b", "win_both_b"]
        if c in summary_b.columns
    ]
    merged = summary_a[keep_cols_a].merge(summary_b[keep_cols_b], on=["dataset", "paper_mae", "paper_r2"], how="outer")

    mae_a = pd.to_numeric(merged.get("our_mae_mean_a"), errors="coerce")
    mae_b = pd.to_numeric(merged.get("our_mae_mean_b"), errors="coerce")
    r2_a = pd.to_numeric(merged.get("our_r2_mean_a"), errors="coerce")
    r2_b = pd.to_numeric(merged.get("our_r2_mean_b"), errors="coerce")
    merged["mae_delta_b_minus_a"] = mae_b - mae_a
    merged["r2_delta_b_minus_a"] = r2_b - r2_a
    overlap = mae_a.notna() & mae_b.notna()
    merged["run_b_better_mae"] = overlap & (merged["mae_delta_b_minus_a"] < 0.0)
    merged["run_b_better_r2"] = overlap & (merged["r2_delta_b_minus_a"] > 0.0)
    merged["run_b_better_both"] = merged["run_b_better_mae"] & merged["run_b_better_r2"]
    merged = merged.sort_values("dataset").reset_index(drop=True)

    dataset_rows = []
    for record in merged.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if hasattr(value, "item") and not isinstance(value, str):
                value = value.item()
            if pd.isna(value):
                clean[key] = None
            elif isinstance(value, (bool, str)):
                clean[key] = value
            elif key.endswith("_a") or key.endswith("_b") or key.endswith("_minus_a") or key in {"paper_mae", "paper_r2"}:
                clean[key] = float(value)
            else:
                clean[key] = value
        dataset_rows.append(clean)

    avg_mae_a = _float_or_none(row_a["avg_mae"])
    avg_mae_b = _float_or_none(row_b["avg_mae"])
    avg_r2_a = _float_or_none(row_a["avg_r2"])
    avg_r2_b = _float_or_none(row_b["avg_r2"])

    return {
        "run_a": {
            "run_dir": str(row_a["run_dir"]),
            "status": str(row_a["status"]),
            "avg_mae": avg_mae_a,
            "avg_r2": avg_r2_a,
            "win_both": _int_or_none(row_a["win_both"]),
        },
        "run_b": {
            "run_dir": str(row_b["run_dir"]),
            "status": str(row_b["status"]),
            "avg_mae": avg_mae_b,
            "avg_r2": avg_r2_b,
            "win_both": _int_or_none(row_b["win_both"]),
        },
        "summary": {
            "dataset_overlap_count": int(overlap.sum()),
            "run_a_only_count": int(mae_a.notna().sum() - overlap.sum()),
            "run_b_only_count": int(mae_b.notna().sum() - overlap.sum()),
            "run_b_better_mae_count": int(merged["run_b_better_mae"].sum()),
            "run_b_better_r2_count": int(merged["run_b_better_r2"].sum()),
            "run_b_better_both_count": int(merged["run_b_better_both"].sum()),
            "avg_mae_delta_b_minus_a": _float_or_none(avg_mae_b - avg_mae_a if avg_mae_a is not None and avg_mae_b is not None else None),
            "avg_r2_delta_b_minus_a": _float_or_none(avg_r2_b - avg_r2_a if avg_r2_a is not None and avg_r2_b is not None else None),
        },
        "datasets": dataset_rows,
    }


__all__ = ["compare_experiments", "query_experiments"]
