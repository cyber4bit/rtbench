from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import Ridge

from rtbench.bench.unified_cv import evaluate_strict_unified_cv
from rtbench.bench.unified_features import (
    StrictUnifiedFeatureConfig,
    build_strict_unified_features,
    build_strict_unified_fold_features,
    fit_strict_unified_feature_encoder,
)


def test_strict_unified_feature_encoder_fits_scalers_on_pooled_train_only() -> None:
    X_mol_train = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, np.nan]], dtype=np.float32)
    X_cp_train = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, np.nan], [5.0, 6.0, 9.0]], dtype=np.float32)
    X_mol_val = np.array([[1000.0, -1000.0]], dtype=np.float32)
    X_cp_val = np.array([[1000.0, -1000.0, 500.0]], dtype=np.float32)
    X_mol_test = np.array([[2000.0, -2000.0]], dtype=np.float32)
    X_cp_test = np.array([[2000.0, -2000.0, 900.0]], dtype=np.float32)

    features = build_strict_unified_features(
        X_mol_train=X_mol_train,
        X_cp_train=X_cp_train,
        X_mol_val=X_mol_val,
        X_cp_val=X_cp_val,
        X_mol_test=X_mol_test,
        X_cp_test=X_cp_test,
    )

    enc = features.encoder
    np.testing.assert_allclose(enc.mol_mean_, np.array([2.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(enc.cp_mean_, np.array([3.0, 4.0, 6.0], dtype=np.float32))

    leaked_mol_mean = np.nanmean(np.vstack([X_mol_train, X_mol_val, X_mol_test]), axis=0)
    leaked_cp_mean = np.nanmean(np.vstack([X_cp_train, X_cp_val, X_cp_test]), axis=0)
    assert not np.allclose(enc.mol_mean_, leaked_mol_mean)
    assert not np.allclose(enc.cp_mean_, leaked_cp_mean)

    assert features.X_train.shape[0] == 3
    assert features.X_val.shape[0] == 1
    assert features.X_test.shape[0] == 1


def test_strict_unified_features_have_stable_dimensions_and_names() -> None:
    rng = np.random.default_rng(1)
    X_mol_train = rng.normal(size=(9, 4)).astype(np.float32)
    X_cp_train = rng.normal(size=(9, 5)).astype(np.float32)
    X_mol_val = rng.normal(size=(3, 4)).astype(np.float32)
    X_cp_val = rng.normal(size=(3, 5)).astype(np.float32)
    X_mol_test = rng.normal(size=(2, 4)).astype(np.float32)
    X_cp_test = rng.normal(size=(2, 5)).astype(np.float32)
    config = StrictUnifiedFeatureConfig(cp_interaction_components=2)

    features = build_strict_unified_features(
        X_mol_train=X_mol_train,
        X_cp_train=X_cp_train,
        X_mol_val=X_mol_val,
        X_cp_val=X_cp_val,
        X_mol_test=X_mol_test,
        X_cp_test=X_cp_test,
        config=config,
    )

    expected_dim = 4 + 5 + 8 + 4 * (2 + 8) + 4
    assert features.X_train.shape == (9, expected_dim)
    assert features.X_val.shape == (3, expected_dim)
    assert features.X_test.shape == (2, expected_dim)
    assert len(features.feature_names) == expected_dim
    assert len(set(features.feature_names)) == expected_dim
    assert features.encoder.n_features_out_ == expected_dim


def test_robust_preprocessing_fits_quantiles_on_pooled_train_only_and_clips_heldout_extremes() -> None:
    X_mol_train = np.array([[0.0, -30.0], [10.0, -20.0], [20.0, -10.0], [30.0, 0.0]], dtype=np.float32)
    X_cp_train = np.array([[100.0], [110.0], [120.0], [130.0]], dtype=np.float32)
    X_mol_val = np.array([[10000.0, -10000.0]], dtype=np.float32)
    X_cp_val = np.array([[10000.0]], dtype=np.float32)
    X_mol_test = np.array([[-5000.0, 5000.0]], dtype=np.float32)
    X_cp_test = np.array([[-5000.0]], dtype=np.float32)
    config = StrictUnifiedFeatureConfig(
        include_cp_summary=False,
        include_interactions=False,
        include_condition_residuals=False,
        enable_robust_preprocessing=True,
        robust_features=False,
        robust_scale=False,
        robust_quantile_low=0.25,
        robust_quantile_high=0.75,
    )

    features = build_strict_unified_features(
        X_mol_train=X_mol_train,
        X_cp_train=X_cp_train,
        X_mol_val=X_mol_val,
        X_cp_val=X_cp_val,
        X_mol_test=X_mol_test,
        X_cp_test=X_cp_test,
        config=config,
    )
    enc = features.encoder

    np.testing.assert_allclose(enc.mol_robust_lower_, np.array([7.5, -22.5], dtype=np.float32))
    np.testing.assert_allclose(enc.mol_robust_upper_, np.array([22.5, -7.5], dtype=np.float32))
    np.testing.assert_allclose(enc.cp_robust_lower_, np.array([107.5], dtype=np.float32))
    np.testing.assert_allclose(enc.cp_robust_upper_, np.array([122.5], dtype=np.float32))
    assert not np.allclose(enc.mol_robust_upper_, np.quantile(np.vstack([X_mol_train, X_mol_val, X_mol_test]), 0.75, axis=0))
    assert not np.allclose(enc.cp_robust_upper_, np.quantile(np.vstack([X_cp_train, X_cp_val, X_cp_test]), 0.75, axis=0))

    np.testing.assert_allclose(enc.mol_robust_preprocessor_.transform(X_mol_val), [[22.5, -22.5]])
    np.testing.assert_allclose(enc.cp_robust_preprocessor_.transform(X_cp_val), [[122.5]])
    np.testing.assert_allclose(enc.mol_robust_preprocessor_.transform(X_mol_test), [[7.5, -7.5]])
    np.testing.assert_allclose(enc.cp_robust_preprocessor_.transform(X_cp_test), [[107.5]])
    assert features.X_train.shape == (4, 3)
    assert features.X_val.shape == (1, 3)
    assert features.X_test.shape == (1, 3)
    assert np.isfinite(features.X_val).all()
    assert np.isfinite(features.X_test).all()


def test_strict_unified_feature_config_accepts_mapping_for_runner_configs() -> None:
    X_mol_train = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    X_cp_train = np.array([[100.0], [110.0], [120.0], [130.0]], dtype=np.float32)
    X_mol_val = np.array([[10000.0]], dtype=np.float32)
    X_cp_val = np.array([[10000.0]], dtype=np.float32)

    features = build_strict_unified_features(
        X_mol_train=X_mol_train,
        X_cp_train=X_cp_train,
        X_mol_val=X_mol_val,
        X_cp_val=X_cp_val,
        X_mol_test=X_mol_val,
        X_cp_test=X_cp_val,
        config={
            "include_cp_summary": False,
            "include_interactions": False,
            "include_condition_residuals": False,
            "enable_robust_preprocessing": True,
            "robust_scale": False,
            "robust_features": False,
            "robust_quantile_low": 0.25,
            "robust_quantile_high": 0.75,
        },
    )

    assert features.encoder.config.enable_robust_preprocessing is True
    np.testing.assert_allclose(features.encoder.mol_robust_upper_, [22.5])
    np.testing.assert_allclose(features.encoder.cp_robust_upper_, [122.5])
    assert np.isfinite(features.X_val).all()


def test_robust_feature_preprocessing_keeps_dimensions_stable_and_outputs_finite_values() -> None:
    X_mol_train = np.array(
        [[0.0, 1.0, np.nan], [2.0, np.inf, 3.0], [4.0, 5.0, -np.inf], [1.0e20, -1.0e20, 6.0]],
        dtype=np.float32,
    )
    X_cp_train = np.array(
        [[1.0, 2.0], [np.nan, 4.0], [5.0, np.inf], [-1.0e20, 1.0e20]],
        dtype=np.float32,
    )
    X_mol_val = np.array([[np.inf, -np.inf, 1.0e30], [9.0, 8.0, 7.0]], dtype=np.float32)
    X_cp_val = np.array([[np.inf, -np.inf], [6.0, 7.0]], dtype=np.float32)
    X_mol_test = np.array([[-1.0e30, 1.0e30, np.nan]], dtype=np.float32)
    X_cp_test = np.array([[np.nan, 1.0e30]], dtype=np.float32)
    config = StrictUnifiedFeatureConfig(
        cp_interaction_components=2,
        enable_robust_preprocessing=True,
        robust_quantile_low=0.1,
        robust_quantile_high=0.9,
    )

    features = build_strict_unified_features(
        X_mol_train=X_mol_train,
        X_cp_train=X_cp_train,
        X_mol_val=X_mol_val,
        X_cp_val=X_cp_val,
        X_mol_test=X_mol_test,
        X_cp_test=X_cp_test,
        config=config,
    )

    expected_dim = 3 + 2 + 8 + 3 * (2 + 8) + 3
    assert features.X_train.shape == (4, expected_dim)
    assert features.X_val.shape == (2, expected_dim)
    assert features.X_test.shape == (1, expected_dim)
    assert len(features.feature_names) == expected_dim
    assert features.encoder.feature_robust_lower_.shape == (expected_dim,)
    assert np.isfinite(features.X_train).all()
    assert np.isfinite(features.X_val).all()
    assert np.isfinite(features.X_test).all()


def test_strict_unified_feature_transform_has_no_dataset_or_target_override() -> None:
    X_mol_train = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]], dtype=np.float32)
    X_cp_train = np.array([[10.0, 0.0], [11.0, 1.0], [12.0, 0.0], [13.0, 1.0]], dtype=np.float32)
    enc = fit_strict_unified_feature_encoder(X_mol_train, X_cp_train)

    row_mol = np.array([[8.0, 9.0]], dtype=np.float32)
    row_cp = np.array([[20.0, 2.0]], dtype=np.float32)
    first = enc.transform(row_mol, row_cp)
    second = enc.transform(np.vstack([row_mol, row_mol]), np.vstack([row_cp, row_cp]))

    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(second[0], second[1])


def test_condition_residuals_use_train_relation_when_transforming_heldout_rows() -> None:
    cp_train = np.arange(1.0, 7.0, dtype=np.float32).reshape(-1, 1)
    mol_train = np.concatenate([2.0 * cp_train + 5.0, -3.0 * cp_train + 1.0], axis=1).astype(np.float32)
    cp_test = np.array([[20.0], [30.0]], dtype=np.float32)
    mol_test = np.array([[100.0, -100.0], [200.0, -200.0]], dtype=np.float32)

    enc = fit_strict_unified_feature_encoder(
        mol_train,
        cp_train,
        config=StrictUnifiedFeatureConfig(cp_interaction_components=1, include_interactions=False),
    )
    transformed_train = enc.transform(mol_train, cp_train)
    transformed_test = enc.transform(mol_test, cp_test)
    residual_start = enc.n_features_out_ - enc.n_mol_features_

    np.testing.assert_allclose(transformed_train[:, residual_start:], 0.0, atol=2e-5)
    assert np.max(np.abs(transformed_test[:, residual_start:])) > 1.0


def _matrix(dataset_id: str, n_rows: int, offset: int) -> SimpleNamespace:
    base = np.arange(n_rows, dtype=np.float32) + float(offset)
    cp = np.stack(
        [
            np.full(n_rows, 1.0 + 0.1 * offset, dtype=np.float32),
            np.linspace(0.0, 1.0, n_rows, dtype=np.float32),
        ],
        axis=1,
    )
    mol = np.stack([base, np.sin(base / 3.0), np.cos(base / 4.0)], axis=1).astype(np.float32)
    return SimpleNamespace(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}-row-{i}" for i in range(n_rows)],
        X=np.concatenate([mol, cp], axis=1),
        X_mol=mol,
        X_cp=cp,
        y_sec=(0.7 * mol[:, 0] - 2.0 * mol[:, 1] + 4.0 * cp[:, 0]).astype(np.float32),
    )


def test_unified_feature_builder_smoke_fits_one_pooled_model_per_fold() -> None:
    mats = {
        "0001": _matrix("0001", 18, 0),
        "0002": _matrix("0002", 20, 10),
        "0003": _matrix("0003", 22, 20),
    }
    calls: list[int] = []

    def fit_predict(fold):
        calls.append(fold.fold_id)
        features = build_strict_unified_fold_features(
            fold,
            config=StrictUnifiedFeatureConfig(cp_interaction_components=1),
        )
        model = Ridge(alpha=0.01)
        model.fit(features.X_train, fold.y_train_sec)
        return model.predict(features.X_val), model.predict(features.X_test)

    predictions = evaluate_strict_unified_cv(mats, mats.keys(), fit_predict, n_splits=3, shuffle_seed=4)

    assert calls == [0, 1, 2]
    assert len(predictions) == 3
    for pred in predictions:
        assert len(pred.val_pred_sec) == len(pred.val_meta)
        assert len(pred.test_pred_sec) == len(pred.test_meta)
