# Strict Unified CV Model Round Summary

Date: 2026-05-05

## Scope

This round continued from the merged strict unified CV infrastructure and
replaced the Ridge-only fallback with additive strict `nModel=1` learner
building blocks. The claim path remains the strict pooled runner only:

- one pooled model per sheet/mode/fold
- no lookup
- no HILIC pool
- no local fast candidate
- no per-dataset override
- no test/final fold tuning
- pooled validation folds only for model or hyperparameter selection

The existing official `V7_validate` and `V7_final` PASS results remain
guardrails. They are not evidence for a strict Uni-RT `nModel=1` claim.

## Integrated Additive Code

- `rtbench/bench/unified_features.py`
  - Train-only pooled molecule/CP feature encoder.
  - Fits imputation/scaling/residualizer state on pooled train rows only.
- `rtbench/bench/unified_selection.py`
  - Validation-only pooled selector over explicit candidate grids.
  - Records selected params, validation metrics, refit policy, and `n_model=1`
    metadata.
- `rtbench/bench/unified_hypertl.py`
  - Strict pooled CP-conditioned HyperTL callback.
  - Calls `pretrain_hyper_tl` once per fold with explicit pooled
    `train_idx`/`val_idx`.
  - Predicts pooled validation/test rows without dataset-specific fitting.
- `rtbench/bench/unified_cv.py`
  - Keeps Ridge fallback by default.
  - Uses strict HyperTL only when the config explicitly requests HyperTL or the
    legacy HyperTL-only unified config is active.
- Tests:
  - `tests/test_unified_features.py`
  - `tests/test_unified_selection.py`
  - `tests/test_unified_hypertl_learner.py`

## Worker Outcomes

| Worker | Outcome |
| --- | --- |
| `unified-model-selection` | MERGE. Additive validation-only selector, no runner behavior change. |
| `unified-model-cp-features` | MERGE. Additive pooled train-only CP/condition feature builder. |
| `unified-model-hypertl` | MERGE as implementation mechanism. It gives a real strict CP-conditioned learner path, but not a performance claim. |
| `unified-model-s5-r2` | NO-MERGE model changes. Diagnostic-only report: strict Ridge and global clips still fail S5 R2. |
| `unified-model-eval-report` | NO-MERGE claim. Full Ridge fallback strict reports prove `nModel=1` reporting, but performance fails badly. |

## Strict S5/HILIC HyperTL Full CV

Command:

```powershell
python v7\scripts\run_unified_cv.py --sheet S5 --mode HILIC --config v7\configs\v7_unified.yaml --folds 10 --shuffle-seed 20260505 --output-root outputs_v7_unified_cv10_S5_hypertl --baseline data\baseline\unirt_sota_28.csv --no-download
```

Output paths:

- `outputs_v7_unified_cv10_S5_hypertl/metrics/per_seed.csv`
- `outputs_v7_unified_cv10_S5_hypertl/predictions/unified_cv_predictions.csv`
- `outputs_v7_unified_cv10_S5_hypertl/unified_cv_run.json`
- `outputs_v7_unified_cv10_S5_hypertl/report_vs_unirt.md`
- `outputs_v7_unified_cv10_S5_hypertl/metrics/cross_dataset_summary_vs_unirt.csv`
- `outputs_v7_unified_cv10_S5_hypertl/metrics/dataset_level_comparison_vs_unirt.csv`

`nModel=1` evidence:

- `unified_cv_run.json`: `"n_model": 1`, `"n_splits": 10`, 14 S5 datasets.
- `report_vs_unirt.md`: `Model count: ours nModel=1`.
- Cross-dataset report row: `nModel | 1 | 1 | 17 | 14 | 14`.
- `per_seed.csv`: 140 rows, 14 datasets x seeds `0..9`.

10-fold result:

| Sheet | Mode | Learner | Ours avg MAE | Uni-RT avg MAE | Ours avg R2 | Uni-RT avg R2 | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| S5 | HILIC | strict pooled HyperTL | 4185.70 | 48.09 | -1158035.0542 | 0.8300 | FAIL |

Worst S5 failures:

- `0183 seed 6`: MAE `576984.3844`, R2 `-1.621249e+08`.
- `0183` 10-fold mean: MAE `57760.60`, R2 `-16212491.5465`.
- `0283` 10-fold mean: MAE `41.56`, R2 `-4.9256`.
- `0372` 10-fold mean: MAE `69.45`, R2 `-0.1154`.

## Diagnostics

The Ridge diagnostic worker found that strict pooled linear models are not
viable for S5:

| Variant | Avg MAE | Avg R2 | Beat Both |
| --- | ---: | ---: | ---: |
| `ridge_cp_sec` | 164.6389 | -0.3328 | 0 |
| `ridge_full_sec_clip_q01_99` | 92.2795 | -1.5008 | 0 |
| `ridge_full_sec` | 2806.2536 | -501460.8898 | 0 |

The HyperTL full run improves some 037x rows relative to Ridge but still fails
the strict S5 gate by a large margin. The main failure remains pooled
extrapolation and poor low-dynamic-range R2 behavior, especially `0183` and
`0283`.

## Verification

Required integration test set plus new strict unified tests:

```powershell
python -m pytest tests\test_rtbench.py tests\test_prepare.py tests\test_runner.py tests\test_hyper.py tests\test_models.py tests\test_report_vs_unirt.py tests\test_hyper_validation_indices.py tests\test_unified_cv_core.py tests\test_unified_cv_cli_report.py tests\test_unified_features.py tests\test_unified_selection.py tests\test_unified_hypertl_learner.py -q
```

Result: `86 passed, 1 warning`.

## Conclusion

S5/HILIC still fails the strict Uni-RT `nModel=1` R2 gate. Do not claim that
RTBench-v7 exceeds Uni-RT under the strict unified one-model evaluation.

The merged code is useful infrastructure and a fair strict HyperTL learner
path, but it is not yet a Uni-RT-beating model.
