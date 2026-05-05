from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rtbench.bench.unified_cv import UnifiedCVFoldPrediction
from rtbench.bench.unified_cv import UnifiedCVRowMeta
from rtbench.bench.unified_selection import (
    UnifiedCandidate,
    ridge_candidate_grid,
    select_strict_unified_learner,
    strict_unified_selector_fit_predict,
)


class ConstantRegressor:
    def __init__(self, value: float, fit_log: list[int] | None = None) -> None:
        self.value = float(value)
        self.fit_log = fit_log

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantRegressor":
        if self.fit_log is not None:
            self.fit_log.append(int(len(y)))
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float32)


class ColumnRegressor:
    def __init__(self, column: int) -> None:
        self.column = int(column)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ColumnRegressor":
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)[:, self.column]


class PoisonedTestFold:
    fold_id = 4
    X_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train_sec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    X_val = np.array([[3.0], [4.0]], dtype=np.float32)
    y_val_sec = np.array([0.0, 0.0], dtype=np.float32)
    X_test = np.array([[5.0], [6.0], [7.0]], dtype=np.float32)
    val_meta = (
        UnifiedCVRowMeta("0001", "v0", 0, "val", 4),
        UnifiedCVRowMeta("0001", "v1", 1, "val", 4),
    )

    @property
    def y_test_sec(self) -> np.ndarray:
        raise AssertionError("selector must not read y_test_sec")

    @property
    def test_meta(self) -> tuple[Any, ...]:
        raise AssertionError("selector must not read test_meta")

    @property
    def dataset_indices(self) -> dict[str, Any]:
        raise AssertionError("selector must not read per-dataset indices")


def _constant_candidate(name: str, value: float, fit_log: list[int] | None = None) -> UnifiedCandidate:
    return UnifiedCandidate(
        name=name,
        kind="constant",
        params={"value": float(value)},
        standardize=False,
        estimator_factory=lambda params, _seed: ConstantRegressor(float(params["value"]), fit_log),
    )


def _column_candidate(name: str, column: int) -> UnifiedCandidate:
    return UnifiedCandidate(
        name=name,
        kind="column",
        params={"column": int(column)},
        standardize=False,
        estimator_factory=lambda params, _seed: ColumnRegressor(int(params["column"])),
    )


def _meta_for(dataset_id: str, count: int) -> tuple[UnifiedCVRowMeta, ...]:
    return tuple(UnifiedCVRowMeta(dataset_id, f"{dataset_id}-v{i}", i, "val", 0) for i in range(count))


def test_selector_uses_validation_only_and_does_not_read_test_labels_or_meta() -> None:
    fold = PoisonedTestFold()
    candidates = [_constant_candidate("val-perfect", 0.0), _constant_candidate("test-would-prefer", 100.0)]

    result = select_strict_unified_learner(fold, candidates, metric="mae", refit_policy="train_only")

    assert result.model_info["selected_candidate_name"] == "val-perfect"
    assert result.test_pred_sec.tolist() == [0.0, 0.0, 0.0]
    assert "y_test_sec" in result.model_info["selection_excludes"]
    assert "test_meta" in result.model_info["selection_excludes"]
    assert result.model_info["selection_uses"] == ["X_train", "y_train_sec", "X_val", "y_val_sec"]


def test_selector_records_pooled_candidate_grid_metric_tiebreak_and_refit_policy() -> None:
    fit_log: list[int] = []
    fold = SimpleNamespace(
        fold_id=2,
        X_train=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y_train_sec=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        X_val=np.array([[3.0], [4.0]], dtype=np.float32),
        y_val_sec=np.array([1.0, 2.0], dtype=np.float32),
        X_test=np.array([[5.0]], dtype=np.float32),
    )
    first = _constant_candidate("first-tied", 1.5, fit_log)
    second = _constant_candidate("second-tied", 1.5, fit_log)

    result = select_strict_unified_learner(fold, [first, second], metric="mae", refit_policy="train_val")

    assert result.model_info["selected_candidate_name"] == "first-tied"
    assert result.model_info["tie_break_policy"] == "validation metric, then candidate_index order"
    assert result.model_info["validation_metric"] == "mae"
    assert result.model_info["refit_policy"] == "train_val"
    assert result.model_info["n_final_models"] == 1
    assert result.model_info["n_models_per_dataset"] == 0
    assert result.model_info["n_pooled_candidate_fits"] == 2
    assert result.model_info["final_fit_scope"] == "pooled_train_val"
    assert result.model_info["candidate_grid"][0]["params"] == {"value": 1.5}
    assert fit_log == [3, 3, 5]


def test_selector_rejects_dataset_level_overrides_lookup_hilic_pool_and_local_fast() -> None:
    bad_candidates = [
        UnifiedCandidate(name="ridge", kind="ridge", params={"dataset_overrides": {"0001": {"alpha": 1.0}}}),
        UnifiedCandidate(name="lookup-ridge", kind="ridge", params={"alpha": 1.0}),
        UnifiedCandidate(name="ridge", kind="hilic_pool", params={"alpha": 1.0}),
        UnifiedCandidate(name="ridge", kind="local_fast", params={"alpha": 1.0}),
    ]
    fold = PoisonedTestFold()

    for candidate in bad_candidates:
        with pytest.raises(ValueError, match="strict unified fairness"):
            select_strict_unified_learner(fold, [candidate])


def test_selector_allows_hilic_mode_candidate_names_without_pool() -> None:
    fold = PoisonedTestFold()
    result = select_strict_unified_learner(
        fold,
        [_constant_candidate("hilic-mode-pooled-ridge", 0.0)],
        metric="mae",
        refit_policy="train_only",
    )

    assert result.model_info["selected_candidate_name"] == "hilic-mode-pooled-ridge"


def test_ridge_grid_selects_by_pooled_validation_metric_not_dataset_override() -> None:
    fold = SimpleNamespace(
        fold_id=0,
        X_train=np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        y_train_sec=np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
        X_val=np.array([[4.0], [5.0]], dtype=np.float32),
        y_val_sec=np.array([9.0, 11.0], dtype=np.float32),
        X_test=np.array([[6.0]], dtype=np.float32),
    )

    result = select_strict_unified_learner(
        fold,
        ridge_candidate_grid([0.0, 1000.0]),
        metric="mae",
        refit_policy="train_only",
    )

    assert result.model_info["selected_params"] == {"alpha": 0.0}
    assert result.model_info["candidate_fit_scope"] == "pooled_train_only"
    assert result.model_info["selection_excludes"].count("per_dataset_overrides") == 1


def test_dataset_balanced_objective_groups_validation_by_dataset_and_avoids_row_count_dominance() -> None:
    big_y = np.arange(50, dtype=np.float32)
    small_y = np.array([0.0, 100.0], dtype=np.float32)
    y_val = np.concatenate([big_y, small_y])
    big_dominates = np.concatenate([big_y, np.array([50.0, 50.0], dtype=np.float32)])
    balanced = np.concatenate([big_y + 5.0, small_y])
    fold = SimpleNamespace(
        fold_id=0,
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train_sec=np.zeros(2, dtype=np.float32),
        X_val=np.stack([big_dominates, balanced], axis=1).astype(np.float32),
        y_val_sec=y_val,
        X_test=np.zeros((1, 2), dtype=np.float32),
        val_meta=_meta_for("0001", len(big_y)) + _meta_for("0002", len(small_y)),
    )
    candidates = [_column_candidate("pooled-big-dataset", 0), _column_candidate("balanced-datasets", 1)]

    pooled = select_strict_unified_learner(fold, candidates, metric="mae", refit_policy="train_only")
    grouped = select_strict_unified_learner(
        fold,
        candidates,
        objective="dataset_balanced_r2",
        refit_policy="train_only",
        mean_dataset_mae_slack=100.0,
    )

    assert pooled.model_info["selected_candidate_name"] == "pooled-big-dataset"
    assert grouped.model_info["selected_candidate_name"] == "balanced-datasets"
    assert grouped.model_info["selection_uses"] == ["X_train", "y_train_sec", "X_val", "y_val_sec", "val_meta"]
    selected = grouped.model_info["selected_balanced_validation_metrics"]
    assert selected["n_datasets"] == 2
    assert selected["datasets"][0]["n_val_rows"] == 50
    assert selected["datasets"][1]["n_val_rows"] == 2
    assert selected["mean_dataset_r2"] > grouped.model_info["validation_results"][0]["balanced_metrics"]["mean_dataset_r2"]


def test_dataset_balanced_metrics_include_beat_both_summary_when_baselines_supplied() -> None:
    fold = SimpleNamespace(
        fold_id=0,
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train_sec=np.zeros(2, dtype=np.float32),
        X_val=np.array([[0.0, 10.0], [10.0, 0.0], [0.0, 50.0], [100.0, 50.0]], dtype=np.float32),
        y_val_sec=np.array([0.0, 10.0, 0.0, 100.0], dtype=np.float32),
        X_test=np.zeros((1, 2), dtype=np.float32),
        val_meta=_meta_for("0001", 2) + _meta_for("0002", 2),
    )

    result = select_strict_unified_learner(
        fold,
        [_column_candidate("perfect", 0), _column_candidate("bad", 1)],
        objective="dataset_balanced_r2",
        refit_policy="train_only",
        baseline_metrics={"1": {"mae": 1.0, "r2": 0.9}, "0002": {"mae": 1.0, "r2": 0.9}},
    )

    balanced = result.model_info["selected_balanced_validation_metrics"]
    assert balanced["baseline_dataset_count"] == 2
    assert balanced["beat_mae_count"] == 2
    assert balanced["beat_r2_count"] == 2
    assert balanced["beat_both_count"] == 2
    assert balanced["beat_both_rate"] == 1.0


def test_dataset_balanced_objective_respects_mean_dataset_mae_constraint() -> None:
    fold = SimpleNamespace(
        fold_id=0,
        X_train=np.zeros((2, 2), dtype=np.float32),
        y_train_sec=np.zeros(2, dtype=np.float32),
        X_val=np.array([[100.0, 0.0], [1100.0, 1000.0], [0.0, 5.0], [10.0, 5.0]], dtype=np.float32),
        y_val_sec=np.array([0.0, 1000.0, 0.0, 10.0], dtype=np.float32),
        X_test=np.zeros((1, 2), dtype=np.float32),
        val_meta=_meta_for("0001", 2) + _meta_for("0002", 2),
    )

    result = select_strict_unified_learner(
        fold,
        [_column_candidate("higher-r2-over-mae-limit", 0), _column_candidate("lower-r2-within-mae-limit", 1)],
        objective="dataset_balanced_r2",
        refit_policy="train_only",
        max_mean_dataset_mae=10.0,
    )

    assert result.model_info["selected_candidate_name"] == "lower-r2-within-mae-limit"
    records = result.model_info["validation_results"]
    assert records[0]["balanced_metrics"]["mean_dataset_r2"] > records[1]["balanced_metrics"]["mean_dataset_r2"]
    assert records[0]["balanced_metrics"]["mean_dataset_mae"] > result.model_info["balanced_objective"]["max_mean_dataset_mae"]
    assert records[1]["balanced_metrics"]["mean_dataset_mae"] <= result.model_info["balanced_objective"]["max_mean_dataset_mae"]


def test_dataset_balanced_metrics_emit_low_dynamic_range_diagnostics() -> None:
    fold = SimpleNamespace(
        fold_id=0,
        X_train=np.zeros((2, 1), dtype=np.float32),
        y_train_sec=np.zeros(2, dtype=np.float32),
        X_val=np.array([[10.0], [10.0], [10.0], [0.0], [10.0], [20.0]], dtype=np.float32),
        y_val_sec=np.array([10.0, 10.0, 10.0, 0.0, 10.0, 20.0], dtype=np.float32),
        X_test=np.zeros((1, 1), dtype=np.float32),
        val_meta=_meta_for("0001", 3) + _meta_for("0002", 3),
    )

    result = select_strict_unified_learner(
        fold,
        [_column_candidate("perfect", 0)],
        objective="dataset_balanced_r2",
        refit_policy="train_only",
        low_dynamic_range_threshold=0.01,
    )

    balanced = result.model_info["selected_balanced_validation_metrics"]
    assert balanced["low_dynamic_range_count"] == 1
    assert balanced["low_dynamic_range_datasets"] == ["0001"]
    low_range_row = next(row for row in balanced["datasets"] if row["dataset"] == "0001")
    assert low_range_row["low_dynamic_range"] is True
    assert low_range_row["y_range"] == 0.0


def test_dataset_balanced_objective_ignores_test_metadata() -> None:
    fold = PoisonedTestFold()

    result = select_strict_unified_learner(
        fold,
        [_constant_candidate("val-perfect", 0.0), _constant_candidate("test-would-prefer", 100.0)],
        objective="dataset_balanced_r2",
        refit_policy="train_only",
    )

    assert result.model_info["selected_candidate_name"] == "val-perfect"
    assert "test_meta" in result.model_info["selection_excludes"]
    assert result.model_info["selection_uses"] == ["X_train", "y_train_sec", "X_val", "y_val_sec", "val_meta"]


def test_runner_compatible_wrapper_attaches_metadata_after_core_selection() -> None:
    fold = SimpleNamespace(
        fold_id=1,
        X_train=np.array([[0.0], [1.0]], dtype=np.float32),
        y_train_sec=np.array([2.0, 2.0], dtype=np.float32),
        X_val=np.array([[2.0]], dtype=np.float32),
        y_val_sec=np.array([2.0], dtype=np.float32),
        X_test=np.array([[3.0], [4.0]], dtype=np.float32),
        val_meta=("val-row",),
        test_meta=("test-row-1", "test-row-2"),
    )

    pred = strict_unified_selector_fit_predict(
        fold,
        [_constant_candidate("constant", 2.0)],
        metric="mae",
        refit_policy="train_only",
    )

    assert isinstance(pred, UnifiedCVFoldPrediction)
    assert pred.val_meta == ("val-row",)
    assert pred.test_meta == ("test-row-1", "test-row-2")
    assert pred.model_info["selector"] == "strict_unified_pooled_validation"
