from __future__ import annotations

import numpy as np
import pytest

from rtbench.models.ridge import _fit_ridge_models


def test_fit_ridge_models_returns_transfer_and_local_candidates() -> None:
    X_src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    y_src = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    X_t_train = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)
    y_t_train = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    X_val = np.array([[6.0, 6.0], [7.0, 7.0]], dtype=np.float32)
    X_test = np.array([[8.0, 8.0], [9.0, 9.0]], dtype=np.float32)
    y_val_sec = np.array([6.0, 7.0], dtype=np.float32)

    outputs = _fit_ridge_models(
        model_cfg={"RIDGE_TL_ALPHA": 0.1, "RIDGE_LOCAL_ALPHA": 0.1},
        X_src=X_src,
        y_src=y_src,
        X_t_train=X_t_train,
        y_t_train=y_t_train,
        X_val=X_val,
        y_val_sec=y_val_sec,
        X_test=X_test,
        seed=3,
        source_weight=0.2,
        target_weight=1.0,
        name_prefix="RID_",
    )

    assert [out.name for out in outputs] == ["RID_RIDGE_TL", "RID_RIDGE_LOCAL"]
    assert all(out.val_pred.shape == (2,) for out in outputs)
    assert all(np.isfinite(out.val_metrics["mae"]) for out in outputs)


def test_fit_ridge_models_validates_source_weight_length() -> None:
    with pytest.raises(ValueError, match="source_sample_weights length mismatch"):
        _fit_ridge_models(
            model_cfg={},
            X_src=np.zeros((2, 2), dtype=np.float32),
            y_src=np.zeros(2, dtype=np.float32),
            X_t_train=np.zeros((2, 2), dtype=np.float32),
            y_t_train=np.zeros(2, dtype=np.float32),
            X_val=np.zeros((1, 2), dtype=np.float32),
            y_val_sec=np.zeros(1, dtype=np.float32),
            X_test=np.zeros((1, 2), dtype=np.float32),
            seed=0,
            source_weight=0.2,
            target_weight=1.0,
            source_sample_weights=np.array([0.2], dtype=np.float32),
        )
