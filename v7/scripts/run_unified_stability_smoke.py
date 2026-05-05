from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_FILES = (
    Path("predictions/unified_cv_predictions.csv"),
    Path("metrics/unified_cv_predictions.csv"),
)
PER_SEED_PATH = Path("metrics/per_seed.csv")
METADATA_PATH = Path("unified_cv_run.json")
CROSS_SUMMARY_PATH = Path("metrics/cross_dataset_summary_vs_unirt.csv")
REPORT_VS_UNIRT_PATH = Path("report_vs_unirt.md")
DEFAULT_WORST_N = 25
DEFAULT_MAX_ABS_ERROR_SEC = 600.0
DEFAULT_MAX_RANGE_EXTENSION_RATIO = 2.0
DEFAULT_RANGE_PADDING_SEC = 60.0

TRAIN_AUDIT_CANDIDATES = (
    Path("metrics/unified_cv_fold_audit.csv"),
    Path("metrics/unified_cv_audit.csv"),
    Path("metrics/fold_audit.csv"),
    Path("metrics/split_audit.csv"),
)
TRAIN_MIN_COLUMNS = ("train_y_min_sec", "y_train_min_sec", "y_train_min", "train_min_y_sec", "train_y_min")
TRAIN_MAX_COLUMNS = ("train_y_max_sec", "y_train_max_sec", "y_train_max", "train_max_y_sec", "train_y_max")
ACCEPTED_TRAIN_RANGE_FIELDS = (
    "fold",
    "train_y_min_sec",
    "train_y_max_sec",
    "train_rows",
    "val_rows",
    "test_rows",
)
OPTIONAL_AUDIT_FIELDS = (
    "n_model",
    "unified_model_id",
    "val_y_min_sec",
    "val_y_max_sec",
    "test_y_min_sec",
    "test_y_max_sec",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose strict unified CV stability from an output directory produced by v7/scripts/run_unified_cv.py."
    )
    parser.add_argument("--output-root", required=True, help="Unified CV output root to diagnose.")
    parser.add_argument(
        "--report",
        default="",
        help="Markdown report path. Defaults to <output-root>/metrics/unified_stability_report.md.",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=DEFAULT_WORST_N,
        help="Number of worst dataset/fold rows to show in the markdown report.",
    )
    parser.add_argument(
        "--max-abs-error-sec",
        type=float,
        default=DEFAULT_MAX_ABS_ERROR_SEC,
        help="Hard gate: largest finite absolute prediction error allowed.",
    )
    parser.add_argument(
        "--max-range-extension-ratio",
        type=float,
        default=DEFAULT_MAX_RANGE_EXTENSION_RATIO,
        help="Hard gate: allowed prediction extension beyond reference range as a multiple of reference span.",
    )
    parser.add_argument(
        "--range-padding-sec",
        type=float,
        default=DEFAULT_RANGE_PADDING_SEC,
        help="Hard gate: absolute seconds padding added to prediction range checks.",
    )
    parser.add_argument(
        "--require-train-range",
        action="store_true",
        help="Make missing train range audit fields a hard failure instead of a warning.",
    )
    return parser


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    aliases = {
        "Dataset": "dataset",
        "Fold": "fold",
        "y_true": "y_true_sec",
        "target": "y_true_sec",
        "true": "y_true_sec",
        "prediction": "y_pred_sec",
        "pred": "y_pred_sec",
        "y_pred": "y_pred_sec",
        "row_id": "original_row_id",
        "id": "original_row_id",
    }
    rename = {col: aliases[col] for col in out.columns if col in aliases}
    if "seed" in out.columns and "fold" not in out.columns:
        rename["seed"] = "fold"
    out = out.rename(columns=rename)
    if "dataset" in out.columns:
        out["dataset"] = out["dataset"].astype(str).str.strip().str.zfill(4)
    if "fold" in out.columns:
        out["fold"] = pd.to_numeric(out["fold"], errors="coerce").astype("Int64")
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    return _normalise_columns(pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8"))


def _find_predictions(output_root: Path) -> tuple[pd.DataFrame, Path | None, list[str]]:
    errors: list[str] = []
    for rel_path in PREDICTION_FILES:
        path = output_root / rel_path
        if path.exists():
            return _read_csv(path), path, errors

    pred_dir = output_root / "predictions"
    if pred_dir.exists():
        frames = []
        for path in sorted(pred_dir.rglob("*.csv")):
            try:
                frame = _read_csv(path)
            except Exception as exc:
                errors.append(f"Could not read prediction CSV {path}: {exc}")
                continue
            if "y_pred_sec" in frame.columns:
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True, sort=False), pred_dir, errors

    errors.append(
        "Missing predictions CSV. Expected predictions/unified_cv_predictions.csv or a CSV under predictions/ with y_pred_sec."
    )
    return pd.DataFrame(), None, errors


def _find_per_seed(output_root: Path) -> tuple[pd.DataFrame, Path | None, list[str]]:
    path = output_root / PER_SEED_PATH
    if not path.exists():
        return pd.DataFrame(), None, [f"Missing per-seed metrics: {path}"]
    try:
        return _read_csv(path), path, []
    except Exception as exc:
        return pd.DataFrame(), path, [f"Could not read per-seed metrics {path}: {exc}"]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8").lstrip("\ufeff"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _nmodel_evidence(output_root: Path, per_seed: pd.DataFrame) -> tuple[pd.DataFrame, bool, list[int]]:
    rows: list[dict[str, Any]] = []

    meta_path = output_root / METADATA_PATH
    if meta_path.exists():
        meta = _load_json(meta_path)
        rows.append({"source": str(meta_path), "field": "n_model", "value": meta.get("n_model")})

    if "n_model" in per_seed.columns and not per_seed.empty:
        values = sorted({int(v) for v in pd.to_numeric(per_seed["n_model"], errors="coerce").dropna().astype(int)})
        for value in values:
            rows.append({"source": str(output_root / PER_SEED_PATH), "field": "n_model unique value", "value": value})

    cross_path = output_root / CROSS_SUMMARY_PATH
    if cross_path.exists():
        try:
            cross = pd.read_csv(cross_path, encoding="utf-8")
            metric_col = next((col for col in cross.columns if str(col).strip().lower() == "metrics"), None)
            if metric_col is not None:
                n_rows = cross.loc[cross[metric_col].astype(str).str.strip().str.lower() == "nmodel"]
                for _, row in n_rows.iterrows():
                    ours_col = _ours_metric_column(cross.columns, metric_col)
                    if ours_col is not None:
                        parsed = _int_or_none(row.get(ours_col))
                        if parsed is not None:
                            rows.append({"source": str(cross_path), "field": str(ours_col), "value": parsed})
        except Exception as exc:
            rows.append({"source": str(cross_path), "field": "read_error", "value": str(exc)})

    report_path = output_root / REPORT_VS_UNIRT_PATH
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"ours\s+nModel\s*=\s*(\d+)", text, flags=re.IGNORECASE):
            value = match.group(1)
            rows.append({"source": str(report_path), "field": "markdown nModel", "value": value})

    evidence = pd.DataFrame(rows, columns=["source", "field", "value"])
    parsed_values = [_int_or_none(value) for value in evidence["value"].tolist()] if not evidence.empty else []
    parsed = [value for value in parsed_values if value is not None]
    ok = bool(parsed) and all(value == 1 for value in parsed)
    return evidence, ok, parsed


def _ours_metric_column(columns: Any, metric_col: str) -> str | None:
    non_metric = [str(col) for col in columns if str(col) != str(metric_col)]
    for col in non_metric:
        text = col.lower()
        if "rtbench" in text or "ours" in text:
            return col
    return non_metric[0] if non_metric else None


def _load_train_ranges(output_root: Path) -> tuple[pd.DataFrame, Path | None, list[str]]:
    missing = [
        "No train range audit CSV was found.",
        "Accepted paths: " + ", ".join(path.as_posix() for path in TRAIN_AUDIT_CANDIDATES) + ".",
        "Required fields: " + ", ".join(ACCEPTED_TRAIN_RANGE_FIELDS) + ".",
        "Optional evidence fields: " + ", ".join(OPTIONAL_AUDIT_FIELDS) + ".",
    ]
    for rel_path in TRAIN_AUDIT_CANDIDATES:
        path = output_root / rel_path
        if not path.exists():
            continue
        try:
            audit = _read_csv(path)
        except Exception as exc:
            return pd.DataFrame(), path, [f"Could not read train range audit CSV {path}: {exc}"]
        min_col = next((col for col in TRAIN_MIN_COLUMNS if col in audit.columns), None)
        max_col = next((col for col in TRAIN_MAX_COLUMNS if col in audit.columns), None)
        if min_col is None or max_col is None or "fold" not in audit.columns:
            fields = ", ".join(str(col) for col in audit.columns)
            return pd.DataFrame(), path, [
                f"Train range audit CSV {path} is missing fold plus train min/max fields.",
                f"Observed fields: {fields}",
                "Accepted train min fields: " + ", ".join(TRAIN_MIN_COLUMNS),
                "Accepted train max fields: " + ", ".join(TRAIN_MAX_COLUMNS),
            ]
        audit = audit.rename(columns={min_col: "train_y_min_sec", max_col: "train_y_max_sec"})
        audit["train_y_min_sec"] = pd.to_numeric(audit["train_y_min_sec"], errors="coerce")
        audit["train_y_max_sec"] = pd.to_numeric(audit["train_y_max_sec"], errors="coerce")
        return audit, path, []
    return pd.DataFrame(), None, missing


def _prediction_diagnostics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str]]:
    errors: list[str] = []
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, ["No predictions were available."]
    required = {"dataset", "fold", "y_pred_sec"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        errors.append(f"Prediction data is missing required columns: {missing}")
        return pd.DataFrame(), pd.DataFrame(), 0, errors
    if "y_true_sec" not in predictions.columns:
        errors.append("Prediction data is missing y_true_sec; worst-error diagnostics cannot be computed.")

    pred = predictions.copy()
    pred["y_pred_sec"] = pd.to_numeric(pred["y_pred_sec"], errors="coerce")
    if "y_true_sec" in pred.columns:
        pred["y_true_sec"] = pd.to_numeric(pred["y_true_sec"], errors="coerce")
    pred["fold"] = pd.to_numeric(pred["fold"], errors="coerce")

    nonfinite_mask = ~np.isfinite(pred["y_pred_sec"].to_numpy(dtype=np.float64))
    nonfinite_count = int(nonfinite_mask.sum())

    finite = pred.loc[~nonfinite_mask].copy()
    if "y_true_sec" in finite.columns:
        finite["abs_error_sec"] = (finite["y_pred_sec"] - finite["y_true_sec"]).abs()
        finite["signed_error_sec"] = finite["y_pred_sec"] - finite["y_true_sec"]
    else:
        finite["abs_error_sec"] = np.nan
        finite["signed_error_sec"] = np.nan

    worst_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "fold"]
    for (dataset, fold), cur in finite.groupby(group_cols, dropna=False, sort=True):
        row_id = ""
        if cur["abs_error_sec"].notna().any():
            idx = cur["abs_error_sec"].idxmax()
            row_id = str(cur.loc[idx, "original_row_id"]) if "original_row_id" in cur.columns else ""
            worst_abs = float(cur.loc[idx, "abs_error_sec"])
            worst_signed = float(cur.loc[idx, "signed_error_sec"])
            mae = float(cur["abs_error_sec"].mean())
            true_min = float(cur["y_true_sec"].min())
            true_max = float(cur["y_true_sec"].max())
        else:
            worst_abs = float("nan")
            worst_signed = float("nan")
            mae = float("nan")
            true_min = float("nan")
            true_max = float("nan")
        pred_min = float(cur["y_pred_sec"].min()) if not cur.empty else float("nan")
        pred_max = float(cur["y_pred_sec"].max()) if not cur.empty else float("nan")
        worst_rows.append(
            {
                "dataset": str(dataset).zfill(4),
                "fold": int(fold) if pd.notna(fold) else "",
                "n_rows": int(len(cur)),
                "mae_sec": mae,
                "max_abs_error_sec": worst_abs,
                "signed_error_at_worst_sec": worst_signed,
                "worst_row_id": row_id,
            }
        )
        range_rows.append(
            {
                "dataset": str(dataset).zfill(4),
                "fold": int(fold) if pd.notna(fold) else "",
                "n_rows": int(len(cur)),
                "pred_min_sec": pred_min,
                "pred_max_sec": pred_max,
                "y_true_min_sec": true_min,
                "y_true_max_sec": true_max,
            }
        )

    worst = pd.DataFrame(worst_rows).sort_values("max_abs_error_sec", ascending=False, na_position="last")
    ranges = pd.DataFrame(range_rows).sort_values(["dataset", "fold"]).reset_index(drop=True)
    return worst.reset_index(drop=True), ranges, nonfinite_count, errors


def _apply_train_ranges(ranges: pd.DataFrame, train_ranges: pd.DataFrame) -> pd.DataFrame:
    if ranges.empty:
        return ranges
    out = ranges.copy()
    out["range_reference"] = "y_true"
    out["ref_min_sec"] = out["y_true_min_sec"]
    out["ref_max_sec"] = out["y_true_max_sec"]
    if train_ranges.empty:
        return out

    audit = train_ranges.copy()
    if "dataset" in audit.columns:
        merge_cols = ["dataset", "fold"]
        audit["dataset"] = audit["dataset"].astype(str).str.zfill(4)
    else:
        merge_cols = ["fold"]
    cols = [*merge_cols, "train_y_min_sec", "train_y_max_sec"]
    merged = out.merge(audit[cols], on=merge_cols, how="left")
    has_train = merged["train_y_min_sec"].notna() & merged["train_y_max_sec"].notna()
    merged.loc[has_train, "range_reference"] = "train"
    merged.loc[has_train, "ref_min_sec"] = merged.loc[has_train, "train_y_min_sec"]
    merged.loc[has_train, "ref_max_sec"] = merged.loc[has_train, "train_y_max_sec"]
    return merged


def _range_violations(ranges: pd.DataFrame, *, max_extension_ratio: float, padding_sec: float) -> pd.DataFrame:
    if ranges.empty:
        return ranges
    out = ranges.copy()
    span = (out["ref_max_sec"] - out["ref_min_sec"]).clip(lower=0.0)
    allowance = float(padding_sec) + float(max_extension_ratio) * span
    out["range_allowance_sec"] = allowance
    out["below_ref_sec"] = (out["ref_min_sec"] - out["pred_min_sec"]).clip(lower=0.0)
    out["above_ref_sec"] = (out["pred_max_sec"] - out["ref_max_sec"]).clip(lower=0.0)
    out["range_violation"] = (out["below_ref_sec"] > allowance) | (out["above_ref_sec"] > allowance)
    return out


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows) if max_rows is not None else df
    cols = list(shown.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in shown.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _gate_row(name: str, passed: bool, detail: str, hard: bool = True) -> dict[str, str]:
    return {
        "gate": name,
        "status": "PASS" if passed else ("FAIL" if hard else "WARN"),
        "hard": "yes" if hard else "no",
        "detail": detail,
    }


def diagnose(
    output_root: Path,
    *,
    report_path: Path,
    worst_n: int,
    max_abs_error_sec: float,
    max_range_extension_ratio: float,
    range_padding_sec: float,
    require_train_range: bool,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_seed, per_seed_path, per_seed_errors = _find_per_seed(output_root)
    predictions, predictions_path, prediction_load_errors = _find_predictions(output_root)
    nmodel, nmodel_ok, nmodel_values = _nmodel_evidence(output_root, per_seed)
    train_ranges, train_range_path, train_range_messages = _load_train_ranges(output_root)
    worst, ranges_raw, nonfinite_count, pred_errors = _prediction_diagnostics(predictions)
    ranges = _range_violations(
        _apply_train_ranges(ranges_raw, train_ranges),
        max_extension_ratio=max_range_extension_ratio,
        padding_sec=range_padding_sec,
    )

    worst_path = metrics_dir / "unified_stability_worst_errors.csv"
    range_path = metrics_dir / "unified_stability_ranges.csv"
    summary_path = metrics_dir / "unified_stability_summary.json"
    if not worst.empty:
        worst.to_csv(worst_path, index=False, encoding="utf-8")
    else:
        worst_path.write_text("", encoding="utf-8")
    if not ranges.empty:
        ranges.to_csv(range_path, index=False, encoding="utf-8")
    else:
        range_path.write_text("", encoding="utf-8")

    max_abs_error = float(worst["max_abs_error_sec"].max()) if not worst.empty and worst["max_abs_error_sec"].notna().any() else float("nan")
    range_violation_count = int(ranges["range_violation"].sum()) if "range_violation" in ranges.columns else 0
    train_range_available = train_range_path is not None and not train_ranges.empty
    hard_errors = per_seed_errors + prediction_load_errors + pred_errors
    gates = [
        _gate_row("per_seed.csv readable", per_seed_path is not None and not per_seed.empty, "; ".join(per_seed_errors) or str(per_seed_path)),
        _gate_row("predictions readable", predictions_path is not None and not predictions.empty, "; ".join(prediction_load_errors) or str(predictions_path)),
        _gate_row("nModel evidence is exactly 1", nmodel_ok, f"observed nModel evidence={nmodel_values or 'none'}"),
        _gate_row("all predictions finite", nonfinite_count == 0, f"nonfinite prediction count={nonfinite_count}"),
        _gate_row("truth errors computable", not pred_errors and not worst.empty, "; ".join(pred_errors) or f"dataset/fold groups={len(worst)}"),
        _gate_row(
            "max absolute error within smoke threshold",
            math.isfinite(max_abs_error) and max_abs_error <= float(max_abs_error_sec),
            f"max_abs_error_sec={max_abs_error:.4f} threshold={float(max_abs_error_sec):.4f}" if math.isfinite(max_abs_error) else "max_abs_error_sec unavailable",
        ),
        _gate_row(
            "prediction range within broad reference envelope",
            range_violation_count == 0 and not ranges.empty,
            f"violating dataset/fold ranges={range_violation_count}; padding_sec={range_padding_sec}; extension_ratio={max_range_extension_ratio}",
        ),
        _gate_row(
            "train range audit available",
            train_range_available,
            str(train_range_path) if train_range_available else " ".join(train_range_messages),
            hard=bool(require_train_range),
        ),
    ]
    gate_df = pd.DataFrame(gates)
    overall_pass = not (gate_df.loc[gate_df["hard"] == "yes", "status"] == "FAIL").any()

    summary = {
        "output_root": str(output_root),
        "overall_status": "PASS" if overall_pass else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_seed_path": None if per_seed_path is None else str(per_seed_path),
        "predictions_path": None if predictions_path is None else str(predictions_path),
        "nmodel_values": nmodel_values,
        "nonfinite_prediction_count": nonfinite_count,
        "max_abs_error_sec": None if not math.isfinite(max_abs_error) else max_abs_error,
        "range_violation_count": range_violation_count,
        "train_range_audit_path": None if train_range_path is None else str(train_range_path),
        "train_range_available": train_range_available,
        "hard_errors": hard_errors,
        "worst_errors_csv": str(worst_path),
        "ranges_csv": str(range_path),
        "summary_json": str(summary_path),
        "report": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Unified CV Stability Smoke Report",
        "",
        f"- Output root: `{output_root}`",
        f"- Overall gate: **{summary['overall_status']}**",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Gates",
        "",
        _markdown_table(gate_df),
        "",
        "## nModel Evidence",
        "",
        _markdown_table(nmodel),
        "",
        "## Worst Dataset/Fold Errors",
        "",
        _markdown_table(worst, max_rows=max(1, int(worst_n))),
        "",
        "## Prediction Range Diagnostics",
        "",
        _markdown_table(ranges.sort_values("range_violation", ascending=False) if "range_violation" in ranges.columns else ranges, max_rows=max(1, int(worst_n))),
        "",
        "## Train Range Audit Hook",
        "",
    ]
    if train_range_available:
        lines.append(f"Train range audit was read from `{train_range_path}`.")
    else:
        lines.extend(
            [
                "Current outputs do not expose the pooled train y-range needed for the strongest range stability gate.",
                "",
                "Missing fields:",
                *[f"- {item}" for item in train_range_messages],
                "",
                "Minimal hook requested from the audit worker:",
                "- Write `<output-root>/metrics/unified_cv_fold_audit.csv` with one row per fold.",
                "- Required columns: `fold`, `train_y_min_sec`, `train_y_max_sec`, `train_rows`, `val_rows`, `test_rows`.",
                "- Strongly recommended columns: `n_model`, `unified_model_id`, `val_y_min_sec`, `val_y_max_sec`, `test_y_min_sec`, `test_y_max_sec`.",
                "- If dataset-specific audit rows are emitted, include `dataset`; otherwise fold-level pooled rows are sufficient.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Summary JSON: `{summary_path}`",
            f"- Worst errors CSV: `{worst_path}`",
            f"- Prediction ranges CSV: `{range_path}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = Path(args.output_root)
    report_path = Path(args.report) if args.report else output_root / "metrics" / "unified_stability_report.md"
    summary = diagnose(
        output_root,
        report_path=report_path,
        worst_n=int(args.worst_n),
        max_abs_error_sec=float(args.max_abs_error_sec),
        max_range_extension_ratio=float(args.max_range_extension_ratio),
        range_padding_sec=float(args.range_padding_sec),
        require_train_range=bool(args.require_train_range),
    )
    print(f"overall_status: {summary['overall_status']}")
    print(f"report: {summary['report']}")
    print(f"summary_json: {summary['summary_json']}")


if __name__ == "__main__":
    main()
