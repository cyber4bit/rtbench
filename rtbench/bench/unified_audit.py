from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .unified_cv import UnifiedCVFold, UnifiedCVFoldPrediction


@dataclass(frozen=True)
class UnifiedAuditConfig:
    """Configuration for strict unified CV prediction auditing."""

    out_of_train_margin_sec: float = 0.0
    out_of_train_margin_fraction: float = 0.0
    extreme_margin_sec: float = 600.0
    extreme_margin_train_range_multiplier: float = 5.0


@dataclass(frozen=True)
class UnifiedAuditGuardConfig:
    """Conservative guard thresholds. The guard is only applied when called."""

    max_prediction_nonfinite_count: int = 0
    max_extreme_out_of_train_range_count: int = 0
    max_extreme_out_of_train_range_fraction: float = 0.0


@dataclass(frozen=True)
class UnifiedAuditGuardResult:
    passed: bool
    failures: tuple[str, ...]
    total_prediction_nonfinite_count: int
    total_extreme_out_of_train_range_count: int
    max_extreme_out_of_train_range_fraction: float
    rows_checked: int
    config: Mapping[str, Any] = field(default_factory=dict)


def audit_unified_cv_predictions(
    *,
    folds: Sequence[UnifiedCVFold],
    predictions: Sequence[UnifiedCVFoldPrediction] | None = None,
    prediction_df: pd.DataFrame | None = None,
    config: UnifiedAuditConfig | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Audit strict unified CV predictions per fold and dataset.

    `folds` provide the train/validation/test label ranges. Predictions can be
    supplied either as `UnifiedCVFoldPrediction` objects or as a DataFrame from
    `prediction_frame`/`unified_cv_predictions.csv`-style output. DataFrame input
    must contain dataset, fold, local row index, and prediction columns.
    """

    cfg = _coerce_audit_config(config)
    fold_by_id = {int(fold.fold_id): fold for fold in folds}
    if not fold_by_id:
        raise ValueError("audit_unified_cv_predictions requires at least one fold")
    if predictions is None and prediction_df is None:
        raise ValueError("audit_unified_cv_predictions requires predictions or prediction_df")
    if predictions is not None and prediction_df is not None:
        raise ValueError("provide predictions or prediction_df, not both")

    pred_df = (
        _prediction_objects_to_frame(fold_by_id, predictions)
        if predictions is not None
        else _normalise_prediction_frame(prediction_df)
    )
    pred_df = _attach_truth_from_folds(pred_df, fold_by_id)

    rows: list[dict[str, Any]] = []
    grouped = pred_df.groupby(["fold", "dataset"], sort=True, dropna=False)
    for (fold_id_raw, dataset_raw), cur in grouped:
        fold_id = int(fold_id_raw)
        dataset_id = str(dataset_raw).zfill(4)
        if fold_id not in fold_by_id:
            raise KeyError(f"prediction frame references unknown fold {fold_id}")
        fold = fold_by_id[fold_id]
        split_ranges = _dataset_split_ranges(fold, dataset_id)
        pred = cur["y_pred_sec"].to_numpy(dtype=np.float64)
        y_true = cur["y_true_sec"].to_numpy(dtype=np.float64)
        pred_finite = np.isfinite(pred)
        y_finite = np.isfinite(y_true)
        abs_err = np.full(len(cur), np.inf, dtype=np.float64)
        both_finite = pred_finite & y_finite
        abs_err[both_finite] = np.abs(pred[both_finite] - y_true[both_finite])

        y_train_min = split_ranges["y_train_min"]
        y_train_max = split_ranges["y_train_max"]
        train_range = max(float(y_train_max - y_train_min), 0.0) if np.isfinite(y_train_min + y_train_max) else 0.0
        ordinary_margin = max(float(cfg.out_of_train_margin_sec), train_range * float(cfg.out_of_train_margin_fraction))
        extreme_margin = max(float(cfg.extreme_margin_sec), train_range * float(cfg.extreme_margin_train_range_multiplier))
        ordinary_out = pred_finite & ((pred < y_train_min - ordinary_margin) | (pred > y_train_max + ordinary_margin))
        extreme_out = pred_finite & ((pred < y_train_min - extreme_margin) | (pred > y_train_max + extreme_margin))

        worst = _worst_row(cur, abs_err)
        rows.append(
            {
                "fold": fold_id,
                "dataset": dataset_id,
                "n_rows": int(len(cur)),
                **split_ranges,
                "prediction_min": _finite_min(pred),
                "prediction_max": _finite_max(pred),
                "prediction_nonfinite_count": int(np.sum(~pred_finite)),
                "y_true_nonfinite_count": int(np.sum(~y_finite)),
                "max_abs_error": float(np.max(abs_err)) if len(abs_err) else 0.0,
                "out_of_train_margin_sec": float(ordinary_margin),
                "out_of_train_range_count": int(np.sum(ordinary_out)),
                "out_of_train_range_fraction": _fraction(np.sum(ordinary_out), len(cur)),
                "extreme_out_of_train_margin_sec": float(extreme_margin),
                "extreme_out_of_train_range_count": int(np.sum(extreme_out)),
                "extreme_out_of_train_range_fraction": _fraction(np.sum(extreme_out), len(cur)),
                **worst,
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "dataset"]).reset_index(drop=True)


def guard_unified_audit(
    audit_df: pd.DataFrame,
    config: UnifiedAuditGuardConfig | Mapping[str, Any] | None = None,
) -> UnifiedAuditGuardResult:
    """Classify an audit frame as pass/fail for extreme prediction failures."""

    cfg = _coerce_guard_config(config)
    if audit_df.empty:
        return UnifiedAuditGuardResult(
            passed=True,
            failures=tuple(),
            total_prediction_nonfinite_count=0,
            total_extreme_out_of_train_range_count=0,
            max_extreme_out_of_train_range_fraction=0.0,
            rows_checked=0,
            config=asdict(cfg),
        )

    nonfinite = int(audit_df["prediction_nonfinite_count"].sum())
    extreme_count = int(audit_df["extreme_out_of_train_range_count"].sum())
    extreme_fraction = float(audit_df["extreme_out_of_train_range_fraction"].max())
    failures: list[str] = []
    if nonfinite > int(cfg.max_prediction_nonfinite_count):
        failures.append(
            f"prediction_nonfinite_count={nonfinite} exceeds threshold {cfg.max_prediction_nonfinite_count}"
        )
    if extreme_count > int(cfg.max_extreme_out_of_train_range_count):
        failures.append(
            "extreme_out_of_train_range_count="
            f"{extreme_count} exceeds threshold {cfg.max_extreme_out_of_train_range_count}"
        )
    if extreme_fraction > float(cfg.max_extreme_out_of_train_range_fraction):
        failures.append(
            "extreme_out_of_train_range_fraction="
            f"{extreme_fraction:.6g} exceeds threshold {cfg.max_extreme_out_of_train_range_fraction:.6g}"
        )
    return UnifiedAuditGuardResult(
        passed=not failures,
        failures=tuple(failures),
        total_prediction_nonfinite_count=nonfinite,
        total_extreme_out_of_train_range_count=extreme_count,
        max_extreme_out_of_train_range_fraction=extreme_fraction,
        rows_checked=int(audit_df["n_rows"].sum()),
        config=asdict(cfg),
    )


def write_unified_audit(
    audit_df: pd.DataFrame,
    out_root: str | Path,
    guard_result: UnifiedAuditGuardResult | None = None,
) -> tuple[Path, Path | None]:
    """Write audit CSV and optional guard JSON under an output root."""

    root = Path(out_root)
    metrics_root = root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    audit_path = metrics_root / "unified_cv_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8")
    guard_path = None
    if guard_result is not None:
        guard_path = metrics_root / "unified_cv_guard.json"
        guard_path.write_text(json.dumps(asdict(guard_result), indent=2, sort_keys=True), encoding="utf-8")
    return audit_path, guard_path


def _coerce_audit_config(config: UnifiedAuditConfig | Mapping[str, Any] | None) -> UnifiedAuditConfig:
    if config is None:
        return UnifiedAuditConfig()
    if isinstance(config, UnifiedAuditConfig):
        return config
    return UnifiedAuditConfig(**dict(config))


def _coerce_guard_config(config: UnifiedAuditGuardConfig | Mapping[str, Any] | None) -> UnifiedAuditGuardConfig:
    if config is None:
        return UnifiedAuditGuardConfig()
    if isinstance(config, UnifiedAuditGuardConfig):
        return config
    return UnifiedAuditGuardConfig(**dict(config))


def _prediction_objects_to_frame(
    fold_by_id: Mapping[int, UnifiedCVFold],
    predictions: Sequence[UnifiedCVFoldPrediction] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pred in predictions or ():
        fold_id = int(pred.fold_id)
        if fold_id not in fold_by_id:
            raise KeyError(f"prediction references unknown fold {fold_id}")
        values = np.asarray(pred.test_pred_sec, dtype=np.float64).reshape(-1)
        if len(values) != len(pred.test_meta):
            raise ValueError(f"prediction length mismatch for fold {fold_id}")
        for row_meta, y_pred in zip(pred.test_meta, values, strict=True):
            rows.append(
                {
                    "fold": fold_id,
                    "dataset": row_meta.dataset_id,
                    "split": row_meta.split_name,
                    "original_row_id": row_meta.original_row_id,
                    "local_row_index": int(row_meta.local_row_index),
                    "y_pred_sec": float(y_pred),
                }
            )
    return pd.DataFrame(rows)


def _normalise_prediction_frame(prediction_df: pd.DataFrame | None) -> pd.DataFrame:
    if prediction_df is None:
        return pd.DataFrame()
    df = prediction_df.copy()
    if "dataset" not in df.columns and "dataset_id" in df.columns:
        df = df.rename(columns={"dataset_id": "dataset"})
    if "fold" not in df.columns and "seed" in df.columns:
        df = df.rename(columns={"seed": "fold"})
    required = {"dataset", "fold", "local_row_index", "y_pred_sec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")
    if "original_row_id" not in df.columns:
        df["original_row_id"] = df["local_row_index"].astype(str)
    if "split" not in df.columns:
        df["split"] = "test"
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df["fold"] = df["fold"].astype(int)
    df["local_row_index"] = df["local_row_index"].astype(int)
    df["y_pred_sec"] = pd.to_numeric(df["y_pred_sec"], errors="coerce")
    if "y_true_sec" in df.columns:
        df["y_true_sec"] = pd.to_numeric(df["y_true_sec"], errors="coerce")
    return df


def _attach_truth_from_folds(pred_df: pd.DataFrame, fold_by_id: Mapping[int, UnifiedCVFold]) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()
    df = _normalise_prediction_frame(pred_df)
    if "y_true_sec" in df.columns and df["y_true_sec"].notna().all():
        return df
    truth: dict[tuple[int, str, int], float] = {}
    for fold in fold_by_id.values():
        for row_meta, y_true in zip(fold.test_meta, np.asarray(fold.y_test_sec, dtype=np.float64), strict=True):
            truth[(int(fold.fold_id), row_meta.dataset_id, int(row_meta.local_row_index))] = float(y_true)
    values = []
    for _, row in df.iterrows():
        key = (int(row["fold"]), str(row["dataset"]).zfill(4), int(row["local_row_index"]))
        if key not in truth:
            raise KeyError(f"missing y_true for fold={key[0]} dataset={key[1]} local_row_index={key[2]}")
        values.append(truth[key])
    df["y_true_sec"] = values
    return df


def _dataset_split_ranges(fold: UnifiedCVFold, dataset_id: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for split_name, meta, values in (
        ("train", fold.train_meta, fold.y_train_sec),
        ("val", fold.val_meta, fold.y_val_sec),
        ("test", fold.test_meta, fold.y_test_sec),
    ):
        arr = np.asarray(
            [
                float(y)
                for row_meta, y in zip(meta, np.asarray(values, dtype=np.float64), strict=True)
                if row_meta.dataset_id == dataset_id
            ],
            dtype=np.float64,
        )
        finite = arr[np.isfinite(arr)]
        out[f"y_{split_name}_min"] = float(np.min(finite)) if finite.size else float("nan")
        out[f"y_{split_name}_max"] = float(np.max(finite)) if finite.size else float("nan")
        out[f"y_{split_name}_nonfinite_count"] = int(len(arr) - len(finite))
    return out


def _worst_row(cur: pd.DataFrame, abs_err: np.ndarray) -> dict[str, Any]:
    if cur.empty:
        return {
            "worst_original_row_id": "",
            "worst_local_row_index": -1,
            "worst_y_true_sec": float("nan"),
            "worst_y_pred_sec": float("nan"),
            "worst_abs_error": 0.0,
        }
    idx = int(np.argmax(abs_err))
    row = cur.iloc[idx]
    return {
        "worst_original_row_id": str(row.get("original_row_id", "")),
        "worst_local_row_index": int(row["local_row_index"]),
        "worst_y_true_sec": float(row["y_true_sec"]),
        "worst_y_pred_sec": float(row["y_pred_sec"]),
        "worst_abs_error": float(abs_err[idx]),
    }


def _finite_min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _fraction(count: int | np.integer, total: int) -> float:
    return float(count) / float(total) if int(total) > 0 else 0.0
