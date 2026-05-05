from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rtbench.bench.unified_audit import (
    UnifiedAuditConfig,
    audit_unified_cv_predictions,
    guard_unified_audit,
)
from rtbench.bench.unified_cv import UnifiedCVFoldPrediction, assemble_unified_cv_fold, prediction_frame


def _matrix(dataset_id: str, values: list[float]) -> SimpleNamespace:
    y = np.asarray(values, dtype=np.float32)
    return SimpleNamespace(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}-mol-{i}" for i in range(len(y))],
        X=np.stack([y, y + 1.0], axis=1),
        X_mol=np.stack([y], axis=1),
        X_cp=np.ones((len(y), 1), dtype=np.float32),
        y_sec=y,
    )


def _fold_and_prediction(dataset_id: str = "0001") -> tuple[object, UnifiedCVFoldPrediction]:
    mats = {dataset_id: _matrix(dataset_id, [10, 11, 12, 13, 14, 15, 16, 17, 18])}
    fold = assemble_unified_cv_fold(mats, [dataset_id], 0, n_splits=3, shuffle_seed=0)
    pred = UnifiedCVFoldPrediction(
        fold_id=fold.fold_id,
        val_pred_sec=fold.y_val_sec.copy(),
        test_pred_sec=fold.y_test_sec.copy(),
        val_meta=fold.val_meta,
        test_meta=fold.test_meta,
        model_info={"test": "synthetic"},
    )
    return fold, pred


def test_unified_audit_finite_predictions_pass_guard_from_prediction_frame() -> None:
    fold, pred = _fold_and_prediction()
    df = prediction_frame([pred])

    audit = audit_unified_cv_predictions(folds=[fold], prediction_df=df)
    guard = guard_unified_audit(audit)

    assert guard.passed
    assert audit.loc[0, "prediction_nonfinite_count"] == 0
    assert audit.loc[0, "max_abs_error"] == 0.0
    assert audit.loc[0, "y_train_min"] <= audit.loc[0, "y_train_max"]
    assert audit.loc[0, "y_val_min"] <= audit.loc[0, "y_val_max"]
    assert audit.loc[0, "y_test_min"] <= audit.loc[0, "y_test_max"]


def test_unified_audit_nonfinite_prediction_fails_guard() -> None:
    fold, pred = _fold_and_prediction()
    bad_pred = UnifiedCVFoldPrediction(
        fold_id=pred.fold_id,
        val_pred_sec=pred.val_pred_sec,
        test_pred_sec=pred.test_pred_sec.copy(),
        val_meta=pred.val_meta,
        test_meta=pred.test_meta,
    )
    bad_pred.test_pred_sec[0] = np.nan

    audit = audit_unified_cv_predictions(folds=[fold], predictions=[bad_pred])
    guard = guard_unified_audit(audit)

    assert not guard.passed
    assert guard.total_prediction_nonfinite_count == 1
    assert "prediction_nonfinite_count" in guard.failures[0]


def test_unified_audit_extreme_0183_like_outlier_fails_guard() -> None:
    fold, pred = _fold_and_prediction("0183")
    outlier_pred = UnifiedCVFoldPrediction(
        fold_id=pred.fold_id,
        val_pred_sec=pred.val_pred_sec,
        test_pred_sec=pred.test_pred_sec.copy(),
        val_meta=pred.val_meta,
        test_meta=pred.test_meta,
    )
    outlier_pred.test_pred_sec[-1] = 1_000_000.0

    audit = audit_unified_cv_predictions(
        folds=[fold],
        predictions=[outlier_pred],
        config=UnifiedAuditConfig(extreme_margin_sec=600.0, extreme_margin_train_range_multiplier=5.0),
    )
    guard = guard_unified_audit(audit)

    assert not guard.passed
    assert guard.total_extreme_out_of_train_range_count == 1
    assert audit.loc[0, "dataset"] == "0183"
    assert audit.loc[0, "prediction_max"] == 1_000_000.0
    assert audit.loc[0, "extreme_out_of_train_range_count"] == 1


def test_unified_audit_propagates_worst_row_metadata() -> None:
    fold, pred = _fold_and_prediction()
    worst_pos = 1
    bad_pred = UnifiedCVFoldPrediction(
        fold_id=pred.fold_id,
        val_pred_sec=pred.val_pred_sec,
        test_pred_sec=pred.test_pred_sec.copy(),
        val_meta=pred.val_meta,
        test_meta=pred.test_meta,
    )
    bad_pred.test_pred_sec[worst_pos] = bad_pred.test_pred_sec[worst_pos] + 123.0

    audit = audit_unified_cv_predictions(folds=[fold], predictions=[bad_pred])
    row_meta = fold.test_meta[worst_pos]

    assert audit.loc[0, "worst_original_row_id"] == row_meta.original_row_id
    assert audit.loc[0, "worst_local_row_index"] == row_meta.local_row_index
    assert audit.loc[0, "worst_abs_error"] == 123.0
