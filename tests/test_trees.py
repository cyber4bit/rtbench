from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest

import rtbench.models.trees as trees


class _DummyTreeRegressor:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pred_queue = [
            np.asarray(kwargs.pop("dummy_val_pred"), dtype=np.float32),
            np.asarray(kwargs.pop("dummy_test_pred"), dtype=np.float32),
        ]
        self.feature_importances_ = np.asarray(kwargs.pop("feature_importances", [1.0, 1.0]), dtype=np.float32)

    def fit(self, *args: Any, **kwargs: Any) -> "_DummyTreeRegressor":
        return self

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._pred_queue.pop(0)

    def set_params(self, **kwargs: Any) -> "_DummyTreeRegressor":
        return self


class _EarlyStopXGBRegressor:
    instances: list["_EarlyStopXGBRegressor"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pred_queue = [
            np.asarray(kwargs.pop("dummy_val_pred"), dtype=np.float32),
            np.asarray(kwargs.pop("dummy_test_pred"), dtype=np.float32),
        ]
        self.best_iteration = kwargs.pop("best_iteration", None)
        self.raise_on_set_params = bool(kwargs.pop("raise_on_set_params", False))
        self.feature_importances_ = np.asarray(kwargs.pop("feature_importances", [1.0, 1.0]), dtype=np.float32)
        self.fit_calls: list[dict[str, Any]] = []
        self.predict_calls: list[dict[str, Any]] = []
        self.set_param_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def fit(self, *args: Any, **kwargs: Any) -> "_EarlyStopXGBRegressor":
        self.fit_calls.append(kwargs)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.predict_calls.append(kwargs)
        return self._pred_queue.pop(0)

    def set_params(self, **kwargs: Any) -> "_EarlyStopXGBRegressor":
        self.set_param_calls.append(kwargs)
        if self.raise_on_set_params:
            raise RuntimeError("set_params failed")
        return self


class _EarlyStopLGBMRegressor:
    instances: list["_EarlyStopLGBMRegressor"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._pred_queue = [
            np.asarray(kwargs.pop("dummy_val_pred"), dtype=np.float32),
            np.asarray(kwargs.pop("dummy_test_pred"), dtype=np.float32),
        ]
        self.best_iteration_ = kwargs.pop("best_iteration_", 0)
        self.feature_importances_ = np.asarray(kwargs.pop("feature_importances", [1.0, 1.0]), dtype=np.float32)
        self.fit_calls: list[dict[str, Any]] = []
        self.predict_calls: list[dict[str, Any]] = []
        self.__class__.instances.append(self)

    def fit(self, *args: Any, **kwargs: Any) -> "_EarlyStopLGBMRegressor":
        self.fit_calls.append(kwargs)
        return self

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self.predict_calls.append(kwargs)
        return self._pred_queue.pop(0)


def _tree_model_cfg() -> dict[str, Any]:
    return {
        "EARLY_STOPPING_ROUNDS": 0,
        "XGB_A": {"dummy_val_pred": [1.0, 2.0], "dummy_test_pred": [3.0, 4.0]},
        "XGB_B": {"dummy_val_pred": [2.0, 3.0], "dummy_test_pred": [4.0, 5.0]},
        "LGBM_A": {"dummy_val_pred": [3.0, 4.0], "dummy_test_pred": [5.0, 6.0]},
        "LGBM_B": {"dummy_val_pred": [4.0, 5.0], "dummy_test_pred": [6.0, 7.0]},
    }


def test_fit_tree_models_returns_all_candidates_with_prefix() -> None:
    X_src = np.zeros((2, 3), dtype=np.float32)
    X_t_train = np.ones((2, 3), dtype=np.float32)
    X_val = np.ones((2, 3), dtype=np.float32)
    X_test = np.ones((2, 3), dtype=np.float32)
    y_src = np.array([1.0, 2.0], dtype=np.float32)
    y_t_train = np.array([3.0, 4.0], dtype=np.float32)
    y_val_sec = np.array([10.0, 20.0], dtype=np.float32)

    with (
        mock.patch.object(trees, "XGBRegressor", _DummyTreeRegressor),
        mock.patch.object(trees, "LGBMRegressor", _DummyTreeRegressor),
    ):
        outputs = trees._fit_tree_models(
            model_cfg=_tree_model_cfg(),
            X_src=X_src,
            y_src=y_src,
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_val=X_val,
            y_val_used=np.array([1.0, 2.0], dtype=np.float32),
            y_val_sec=y_val_sec,
            X_test=X_test,
            seed=7,
            source_weight=0.2,
            target_weight=1.0,
            name_prefix="PFX_",
            target_transform="gradient_norm",
            target_inv_scale=10.0,
            target_t0_sec=1.0,
        )

    assert [out.name for out in outputs] == ["PFX_XGB_A", "PFX_XGB_B", "PFX_LGBM_A", "PFX_LGBM_B"]
    assert np.allclose(outputs[0].val_pred, np.array([10.0, 20.0], dtype=np.float32))
    assert np.allclose(outputs[0].test_pred, np.array([30.0, 40.0], dtype=np.float32))


def test_fit_branch_tree_models_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown algo 'svm'"):
        trees._fit_branch_tree_models(
            branch_name="FULL",
            specs=[("BAD", "svm", {})],
            X_src=np.zeros((1, 2), dtype=np.float32),
            y_src=np.zeros(1, dtype=np.float32),
            X_t_train=np.zeros((1, 2), dtype=np.float32),
            y_t_train=np.zeros(1, dtype=np.float32),
            X_val=np.zeros((1, 2), dtype=np.float32),
            y_val_used=np.zeros(1, dtype=np.float32),
            y_val_sec=np.zeros(1, dtype=np.float32),
            X_test=np.zeros((1, 2), dtype=np.float32),
            seed=0,
            source_weight=0.2,
            target_weight=1.0,
        )


def test_split_helpers_validate_ratios_and_fallback_when_stratify_fails() -> None:
    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        trees.random_split(y=np.arange(4, dtype=np.float32), seed=0, train=0.6, val=0.3, test=0.3)

    side_effects = [
        ValueError("first split failed"),
        (np.array([0, 1]), np.array([2, 3])),
        ValueError("second split failed"),
        (np.array([2]), np.array([3])),
    ]
    with mock.patch.object(trees, "train_test_split", side_effect=side_effects):
        split = trees.stratified_split(y=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), seed=11, train=0.5, val=0.25, test=0.25)

    assert split.train_idx.tolist() == [0, 1]
    assert split.val_idx.tolist() == [2]
    assert split.test_idx.tolist() == [3]


def test_target_transform_helpers_cover_identity_and_clipping() -> None:
    y_sec = np.array([1.0, 5.0], dtype=np.float32)

    assert trees._normalize_target_transform("identity") == "none"
    assert trees._normalize_target_transform("mystery") == "none"
    assert np.array_equal(trees._forward_target(y_sec, "none", 1.0, 1.0), y_sec)
    assert np.array_equal(trees._inverse_target(y_sec, "none", 1.0, 1.0), y_sec)
    assert np.isfinite(trees._inverse_target(np.array([1e9], dtype=np.float32), "logk", 1.0, 10.0)).all()
    assert np.isfinite(trees._inverse_target(np.array([1e9], dtype=np.float32), "log1p", 1.0, 1.0)).all()


def test_fit_tree_models_early_stopping_uses_iteration_controls() -> None:
    _EarlyStopXGBRegressor.instances = []
    _EarlyStopLGBMRegressor.instances = []
    X_src = np.zeros((2, 3), dtype=np.float32)
    X_t_train = np.ones((2, 3), dtype=np.float32)
    X_val = np.ones((2, 3), dtype=np.float32)
    X_test = np.ones((2, 3), dtype=np.float32)
    y_src = np.array([1.0, 2.0], dtype=np.float32)
    y_t_train = np.array([3.0, 4.0], dtype=np.float32)
    y_val_used = np.array([0.1, 0.2], dtype=np.float32)
    y_val_sec = np.array([10.0, 20.0], dtype=np.float32)
    cfg = {
        "EARLY_STOPPING_ROUNDS": 5,
        "XGB_A": {"dummy_val_pred": [1.0, 2.0], "dummy_test_pred": [3.0, 4.0], "best_iteration": 2},
        "XGB_B": {
            "dummy_val_pred": [2.0, 3.0],
            "dummy_test_pred": [4.0, 5.0],
            "best_iteration": None,
            "raise_on_set_params": True,
        },
        "LGBM_A": {"dummy_val_pred": [3.0, 4.0], "dummy_test_pred": [5.0, 6.0], "best_iteration_": 3},
        "LGBM_B": {"dummy_val_pred": [4.0, 5.0], "dummy_test_pred": [6.0, 7.0], "best_iteration_": 0},
    }

    with (
        mock.patch.object(trees, "XGBRegressor", _EarlyStopXGBRegressor),
        mock.patch.object(trees, "LGBMRegressor", _EarlyStopLGBMRegressor),
        mock.patch.object(trees, "early_stopping", side_effect=lambda stopping_rounds, verbose=False: ("es", stopping_rounds, verbose)),
    ):
        outputs = trees._fit_tree_models(
            model_cfg=cfg,
            X_src=X_src,
            y_src=y_src,
            X_t_train=X_t_train,
            y_t_train=y_t_train,
            X_val=X_val,
            y_val_used=y_val_used,
            y_val_sec=y_val_sec,
            X_test=X_test,
            seed=5,
            source_weight=0.2,
            target_weight=1.0,
            source_sample_weights=np.array([0.3, 0.4], dtype=np.float32),
            name_prefix="ES_",
            target_transform="gradient_norm",
            target_inv_scale=10.0,
            target_t0_sec=1.0,
        )

    assert [out.name for out in outputs] == ["ES_XGB_A", "ES_XGB_B", "ES_LGBM_A", "ES_LGBM_B"]
    assert _EarlyStopXGBRegressor.instances[0].predict_calls[0]["iteration_range"] == (0, 3)
    assert "iteration_range" not in _EarlyStopXGBRegressor.instances[1].predict_calls[0]
    assert _EarlyStopLGBMRegressor.instances[0].fit_calls[0]["callbacks"] == [("es", 5, False)]
    assert _EarlyStopLGBMRegressor.instances[0].predict_calls[0]["num_iteration"] == 3
    assert _EarlyStopLGBMRegressor.instances[1].predict_calls[0]["num_iteration"] is None


def test_fit_tree_models_validates_source_weight_length() -> None:
    with pytest.raises(ValueError, match="source_sample_weights length mismatch"):
        trees._fit_tree_models(
            model_cfg=_tree_model_cfg(),
            X_src=np.zeros((2, 2), dtype=np.float32),
            y_src=np.zeros(2, dtype=np.float32),
            X_t_train=np.zeros((2, 2), dtype=np.float32),
            y_t_train=np.zeros(2, dtype=np.float32),
            X_val=np.zeros((1, 2), dtype=np.float32),
            y_val_used=np.zeros(1, dtype=np.float32),
            y_val_sec=np.zeros(1, dtype=np.float32),
            X_test=np.zeros((1, 2), dtype=np.float32),
            seed=0,
            source_weight=0.2,
            target_weight=1.0,
            source_sample_weights=np.array([0.2], dtype=np.float32),
        )


def test_fit_branch_tree_models_handles_xgb_and_lgbm_paths() -> None:
    _EarlyStopXGBRegressor.instances = []
    _EarlyStopLGBMRegressor.instances = []

    with (
        mock.patch.object(trees, "XGBRegressor", _EarlyStopXGBRegressor),
        mock.patch.object(trees, "LGBMRegressor", _EarlyStopLGBMRegressor),
        mock.patch.object(trees, "early_stopping", side_effect=lambda stopping_rounds, verbose=False: ("es", stopping_rounds, verbose)),
    ):
        outputs = trees._fit_branch_tree_models(
            branch_name="BR",
            specs=[
                ("X1", "xgb", {"dummy_val_pred": [1.0, 1.5], "dummy_test_pred": [2.0, 2.5], "best_iteration": 0}),
                ("L1", "lgbm", {"dummy_val_pred": [3.0, 3.5], "dummy_test_pred": [4.0, 4.5], "best_iteration_": 2}),
            ],
            X_src=np.zeros((2, 2), dtype=np.float32),
            y_src=np.zeros(2, dtype=np.float32),
            X_t_train=np.ones((2, 2), dtype=np.float32),
            y_t_train=np.ones(2, dtype=np.float32),
            X_val=np.ones((2, 2), dtype=np.float32),
            y_val_used=np.array([0.5, 0.6], dtype=np.float32),
            y_val_sec=np.array([10.0, 12.0], dtype=np.float32),
            X_test=np.ones((2, 2), dtype=np.float32),
            seed=3,
            source_weight=0.2,
            target_weight=1.0,
            early_stopping_rounds=4,
            target_transform="log1p",
            target_inv_scale=1.0,
            target_t0_sec=1.0,
        )

    assert [out.name for out in outputs] == ["BR_X1", "BR_L1"]
    assert _EarlyStopXGBRegressor.instances[0].predict_calls[0]["iteration_range"] == (0, 1)
    assert _EarlyStopLGBMRegressor.instances[0].predict_calls[0]["num_iteration"] == 2
    assert np.allclose(outputs[0].val_pred, np.expm1(np.array([1.0, 1.5], dtype=np.float32)))


def test_fit_branch_tree_models_validates_source_weight_length() -> None:
    with pytest.raises(ValueError, match="source_sample_weights length mismatch"):
        trees._fit_branch_tree_models(
            branch_name="FULL",
            specs=[("X1", "xgb", {"dummy_val_pred": [1.0], "dummy_test_pred": [2.0]})],
            X_src=np.zeros((2, 2), dtype=np.float32),
            y_src=np.zeros(2, dtype=np.float32),
            X_t_train=np.zeros((2, 2), dtype=np.float32),
            y_t_train=np.zeros(2, dtype=np.float32),
            X_val=np.zeros((1, 2), dtype=np.float32),
            y_val_used=np.zeros(1, dtype=np.float32),
            y_val_sec=np.zeros(1, dtype=np.float32),
            X_test=np.zeros((1, 2), dtype=np.float32),
            seed=0,
            source_weight=0.2,
            target_weight=1.0,
            source_sample_weights=np.array([0.2], dtype=np.float32),
        )


def test_mdl_feature_subset_keeps_descriptors_maccs_and_cp() -> None:
    X = np.arange(2 * (2 + 200 + 3), dtype=np.float32).reshape(2, 205)
    subset = trees._mdl_feature_subset(X, {"descriptor": 2, "fingerprint": 200, "cp": 3})

    assert subset.shape[1] == 2 + 166 + 3
    assert np.array_equal(subset[:, :2], X[:, :2])
    assert np.array_equal(subset[:, -3:], X[:, -3:])

    X_extra = np.arange(2 * (2 + 200 + 5 + 3), dtype=np.float32).reshape(2, 210)
    subset_extra = trees._mdl_feature_subset(
        X_extra,
        {"descriptor": 2, "fingerprint": 200, "mol_text": 2, "mol_seq": 3, "cp": 3},
    )
    assert subset_extra.shape[1] == 2 + 166 + 3
    assert np.array_equal(subset_extra[:, :2], X_extra[:, :2])
    assert np.array_equal(subset_extra[:, -3:], X_extra[:, -3:])


def test_mdl_feature_subset_returns_input_for_invalid_shapes_and_empty_right_block() -> None:
    one_dim = np.arange(4, dtype=np.float32)
    invalid = trees._mdl_feature_subset(one_dim, {"descriptor": 1, "fingerprint": 2})
    assert np.array_equal(invalid, one_dim)

    X = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(trees._mdl_feature_subset(X, {"descriptor": -1, "fingerprint": 2}), X)

    no_right = trees._mdl_feature_subset(X, {"descriptor": 1, "fingerprint": 2})
    assert np.array_equal(no_right, X)
