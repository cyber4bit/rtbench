# Strict Unified CV Stability Round Summary

Date: 2026-05-05

## Scope

This round continued from the strict `nModel=1` unified CV infrastructure after
the pooled HyperTL attempt failed S5/HILIC with severe extrapolation. The goal
was stability first: add prediction auditing, train-only robust features,
stable tabular heads, dataset-balanced validation selection, and smoke gates.

The strict claim constraints remain unchanged:

- one pooled model per sheet/mode/fold
- no lookup
- no HILIC pool
- no local fast candidate
- no dataset-id feature or per-dataset override
- no test/final fold tuning

## Integrated Code

- `rtbench/bench/unified_audit.py`
  - Per fold/dataset audit from strict unified folds and predictions.
  - Reports train/validation/test target ranges, prediction ranges, nonfinite
    counts, out-of-train-range counts, and worst-row metadata.
- `rtbench/bench/unified_features.py`
  - Added opt-in train-only robust preprocessing: quantile clipping,
    median/IQR scaling, and finite-value guards.
- `rtbench/bench/unified_tabular.py`
  - Added strict pooled tabular heads: HistGradientBoosting, ExtraTrees, and
    RandomForest builders.
  - Each callback fits exactly one pooled model per fold.
- `rtbench/bench/unified_selection.py`
  - Added opt-in dataset-balanced validation objective that ranks by mean
    dataset R2 under a mean dataset MAE constraint.
- `v7/scripts/run_unified_stability_smoke.py`
  - Report-only stability gate for strict unified CV output directories.
- `rtbench/bench/unified_cv.py`
  - Added `UNIFIED_CV_LEARNER=tabular` hook.
  - Writes `metrics/unified_cv_audit.csv` for strict unified runs.

## Worker Outcomes

| Worker | Outcome |
| --- | --- |
| `unified-audit-guard` | MERGE. Additive audit/guard module and tests. |
| `unified-robust-features` | MERGE. Opt-in robust feature preprocessing. |
| `unified-tabular-heads` | MERGE. Strict pooled tabular learner builders. |
| `unified-balanced-selection` | MERGE. Opt-in dataset-balanced selector objective. |
| `unified-s5-smoke-eval` | MERGE. Report-only stability smoke script. |

## S5/HILIC 3-Fold Smoke Results

| Candidate | Avg MAE | Avg R2 | Stability |
| --- | ---: | ---: | --- |
| ExtraTrees, robust, no interactions | 85.27 | 0.3928 | Stable, no nonfinite predictions |
| HistGradientBoosting, robust | 89.32 | 0.3400 | Stable, no nonfinite predictions |
| RandomForest, robust | 98.82 | 0.2045 | Stable, no nonfinite predictions |
| ExtraTrees, robust, interactions | 84.62 | 0.4651 | Stable, no nonfinite predictions |

The tabular path removed the previous million-scale HyperTL extrapolation, but
none of the 3-fold candidates approached the S5 Uni-RT R2 gate.

## S5/HILIC Full 10-Fold Result

Best 3-fold candidate was evaluated full 10-fold:

```powershell
python v7\scripts\run_unified_cv.py --sheet S5 --mode HILIC --config v7\reports\_generated_configs\V7_stability_S5_tabular_et_interact.yaml --folds 10 --shuffle-seed 20260505 --output-root outputs_v7_unified_cv10_S5_tabular_et_interact_stability --baseline data\baseline\unirt_sota_28.csv --no-download
```

Output directory:

- `outputs_v7_unified_cv10_S5_tabular_et_interact_stability`

Strict evidence:

- `unified_cv_run.json` reports `n_model=1`.
- Uni-RT summary row reports `nModel,1,1,17,14,14`.
- `metrics/per_seed.csv` contains 14 datasets x 10 folds.
- `metrics/unified_cv_audit.csv` is written.
- Stability smoke with train-range audit passed.

10-fold performance:

| Metric | Ours | Uni-RT | Result |
| --- | ---: | ---: | --- |
| MAE average | 53.92 | 48.09 | FAIL |
| R2 average | 0.3580 | 0.8300 | FAIL |
| MedAE average | 30.16 | 27.61 | FAIL |
| MRE average | 0.2307 | 0.1900 | FAIL |

Stability smoke:

- `overall_status: PASS`
- `nmodel_values: [1, 1, 1]`
- `nonfinite_prediction_count: 0`
- `range_violation_count: 0`
- `max_abs_error_sec: 681.17`
- train range audit available from `metrics/unified_cv_audit.csv`

Primary remaining failures:

- `0283` mean R2 `-3.5825` vs Uni-RT `0.6774`.
- `0372` mean R2 `0.3526` vs Uni-RT `0.8089`.
- `0282` mean R2 `0.1760` vs Uni-RT `0.5775`.
- `0027/0183/0184/0185` still lose MAE and R2 to Uni-RT.

## Verification

```powershell
python -m pytest tests\test_rtbench.py tests\test_prepare.py tests\test_runner.py tests\test_hyper.py tests\test_models.py tests\test_report_vs_unirt.py tests\test_hyper_validation_indices.py tests\test_unified_cv_core.py tests\test_unified_cv_cli_report.py tests\test_unified_features.py tests\test_unified_selection.py tests\test_unified_hypertl_learner.py tests\test_unified_audit.py tests\test_unified_tabular.py -q
```

Result: `103 passed, 2 warnings`.

## Conclusion

The stability-first tabular route is a useful infrastructure improvement and
eliminates the catastrophic HyperTL extrapolation, but S5/HILIC still fails the
strict Uni-RT `nModel=1` gate. Do not claim overall strict unified superiority
over Uni-RT.

Next useful work is not another blind full run. It should target the remaining
low-dynamic-range and cross-condition failures, especially `0283`, `0282`,
`0372`, and the `0027/0183/0184/0185` MAE/R2 losses, using only pooled
validation folds and no dataset-id override.
