# Unified CV Stability Smoke Plan

## Scope

This branch adds report-only support for strict unified S5 smoke/evaluation stability checks. It does not change model behavior, add per-dataset rules, tune final/test behavior, or alter `run_unified_cv.py`.

## Diagnostic Script

Script:

```powershell
python v7\scripts\run_unified_stability_smoke.py --output-root <run_unified_cv_output_root>
```

Default outputs:

- `<output-root>/metrics/unified_stability_report.md`
- `<output-root>/metrics/unified_stability_summary.json`
- `<output-root>/metrics/unified_stability_worst_errors.csv`
- `<output-root>/metrics/unified_stability_ranges.csv`

The script consumes the current unified CV output shape:

- `<output-root>/metrics/per_seed.csv`
- `<output-root>/predictions/unified_cv_predictions.csv`
- `<output-root>/unified_cv_run.json`
- Optional Uni-RT report evidence from `<output-root>/report_vs_unirt.md` and `<output-root>/metrics/cross_dataset_summary_vs_unirt.csv`

## Stability Gates

Hard gates by default:

- `per_seed.csv` is readable.
- prediction CSV is readable.
- nModel evidence is present and exactly `1`.
- all predictions are finite.
- prediction errors are computable against `y_true_sec`.
- maximum absolute error is within the smoke threshold, default `600` seconds.
- prediction range is within a broad reference envelope, default truth/train range plus `60` seconds and `2x` reference span.

Warning by default:

- pooled train y-range audit is unavailable.

Use `--require-train-range` to make missing train range a hard failure.

## Missing Audit Hook

Current `run_unified_cv.py` outputs do not persist pooled train y-ranges. That prevents the strongest stability check: comparing prediction ranges against the exact train range for each strict unified fold.

Minimal hook requested from the audit worker:

- Write `<output-root>/metrics/unified_cv_fold_audit.csv`.
- Emit one row per fold.
- Required columns: `fold`, `train_y_min_sec`, `train_y_max_sec`, `train_rows`, `val_rows`, `test_rows`.
- Recommended columns: `n_model`, `unified_model_id`, `val_y_min_sec`, `val_y_max_sec`, `test_y_min_sec`, `test_y_max_sec`.
- If dataset-specific rows are emitted, include `dataset`; otherwise fold-level pooled rows are sufficient.

Until that hook exists, the script falls back to held-out truth y-range for range diagnostics and clearly marks the reference as `y_true`.
