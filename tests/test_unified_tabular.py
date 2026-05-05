from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from rtbench.bench import unified_tabular as ut
from rtbench.bench.unified_cv import (
    UnifiedCVFoldPrediction,
    _default_fit_predict,
    assemble_unified_cv_fold,
    evaluate_strict_unified_cv,
    run_strict_unified_cv_fold,
)


def _matrix(dataset_id: str, n_rows: int, offset: float) -> SimpleNamespace:
    base = np.linspace(0.0, 1.0, n_rows, dtype=np.float32) + float(offset)
    mol = np.stack([base, np.sin(base), np.cos(base * 0.5)], axis=1).astype(np.float32)
    cp = np.stack(
        [
            np.full(n_rows, 1.0 + 0.1 * offset, dtype=np.float32),
            np.linspace(-1.0, 1.0, n_rows, dtype=np.float32),
        ],
        axis=1,
    )
    return SimpleNamespace(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}-{i}" for i in range(n_rows)],
        X=np.concatenate([mol, cp], axis=1),
        X_mol=mol,
        X_cp=cp,
        y_sec=(30.0 * mol[:, 0] - 2.0 * mol[:, 1] + 5.0 * cp[:, 0]).astype(np.float32),
    )


def _mats() -> dict[str, SimpleNamespace]:
    return {
        "0001": _matrix("0001", 18, 0.0),
        "0002": _matrix("0002", 21, 2.0),
        "0003": _matrix("0003", 24, 4.0),
    }


class CountingRegressor:
    fit_calls: list[int] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CountingRegressor":
        CountingRegressor.fit_calls.append(int(len(y)))
        self.mean_ = float(np.mean(y))
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.full(X.shape[0], self.mean_, dtype=np.float32)


class NonFiniteRegressor(CountingRegressor):
    def predict(self, X: np.ndarray) -> np.ndarray:
        pattern = np.asarray([np.nan, np.inf, -np.inf, self.mean_], dtype=np.float32)
        return np.resize(pattern, np.asarray(X).shape[0]).astype(np.float32)


def test_extra_trees_builder_fits_exactly_one_pooled_model_per_fold(monkeypatch) -> None:
    mats = _mats()
    CountingRegressor.fit_calls = []
    monkeypatch.setattr(ut, "ExtraTreesRegressor", CountingRegressor)
    fit_predict = ut.build_extra_trees_fit_predict(params={"n_estimators": 7}, random_state=42)

    predictions = evaluate_strict_unified_cv(mats, mats.keys(), fit_predict, n_splits=3, shuffle_seed=5)

    assert len(predictions) == 3
    assert len(CountingRegressor.fit_calls) == 3
    for pred, train_rows in zip(predictions, CountingRegressor.fit_calls, strict=True):
        assert pred.model_info["learner"] == "extra_trees"
        assert pred.model_info["n_model"] == 1
        assert pred.model_info["n_pooled_model_fits"] == 1
        assert pred.model_info["train_rows"] == train_rows
        assert pred.model_info["n_models_per_dataset"] == 0
        assert "dataset_id_feature" in pred.model_info["selection_excludes"]


def test_finite_prediction_guard_replaces_nonfinite_outputs(monkeypatch) -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 0, n_splits=3, shuffle_seed=7)
    CountingRegressor.fit_calls = []
    monkeypatch.setattr(ut, "ExtraTreesRegressor", NonFiniteRegressor)
    fit_predict = ut.build_extra_trees_fit_predict(random_state=3)

    pred = run_strict_unified_cv_fold(fold, fit_predict)

    assert np.all(np.isfinite(pred.val_pred_sec))
    assert np.all(np.isfinite(pred.test_pred_sec))
    assert len(CountingRegressor.fit_calls) == 1
    assert pred.model_info["clip"]["enabled"] is False


def test_builder_predictions_do_not_depend_on_dataset_ids() -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 1, n_splits=3, shuffle_seed=9)
    renamed = replace(fold, dataset_ids=("9991", "9992", "9993"))
    fit_predict = ut.build_hist_gradient_boosting_fit_predict(
        params={"max_iter": 8, "max_leaf_nodes": 7, "min_samples_leaf": 2},
        random_state=11,
    )

    pred_a = run_strict_unified_cv_fold(fold, fit_predict)
    pred_b = run_strict_unified_cv_fold(renamed, fit_predict)

    np.testing.assert_allclose(pred_a.val_pred_sec, pred_b.val_pred_sec, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pred_a.test_pred_sec, pred_b.test_pred_sec, rtol=1e-6, atol=1e-6)
    assert "dataset_id_feature" in pred_a.model_info["selection_excludes"]


def test_hist_gradient_boosting_builder_is_run_strict_fold_compatible() -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 2, n_splits=3, shuffle_seed=13)
    fit_predict = ut.build_hist_gradient_boosting_fit_predict(
        params={"max_iter": 10, "max_leaf_nodes": 7, "min_samples_leaf": 2},
        clip_predictions=True,
        clip_margin_scales=[0.0, 0.5],
        random_state=17,
    )

    pred = run_strict_unified_cv_fold(fold, fit_predict)

    assert isinstance(pred, UnifiedCVFoldPrediction)
    assert pred.val_meta == fold.val_meta
    assert pred.test_meta == fold.test_meta
    assert pred.model_info["learner"] == "hist_gradient_boosting"
    assert pred.model_info["n_model"] == 1
    assert pred.model_info["clip"]["enabled"] is True
    assert pred.model_info["clip"]["selected_margin_scale"] in {0.0, 0.5}
    assert len(pred.val_pred_sec) == len(fold.y_val_sec)
    assert len(pred.test_pred_sec) == len(fold.y_test_sec)


def test_default_unified_cv_learner_can_select_tabular_head(monkeypatch) -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 0, n_splits=3, shuffle_seed=19)
    CountingRegressor.fit_calls = []
    monkeypatch.setattr(ut, "ExtraTreesRegressor", CountingRegressor)

    fit_predict = _default_fit_predict(
        {
            "UNIFIED_CV_LEARNER": "tabular",
            "UNIFIED_TABULAR_LEARNER": "extra_trees",
            "UNIFIED_TABULAR_EXTRA_TREES": {"n_estimators": 5},
            "STRICT_UNIFIED_FEATURE_CONFIG": {
                "enable_robust_preprocessing": True,
                "include_interactions": False,
                "include_condition_residuals": False,
            },
        }
    )
    pred = run_strict_unified_cv_fold(fold, fit_predict)

    assert len(CountingRegressor.fit_calls) == 1
    assert pred.model_info["learner"] == "extra_trees"
    assert pred.model_info["n_model"] == 1
    assert pred.model_info["feature_source"] == "unified_features.build_strict_unified_fold_features"
