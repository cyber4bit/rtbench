from __future__ import annotations

import numpy as np

from rtbench.models.candidate_builder import CandidateBuildContext, build_candidates, collect_candidates
from rtbench.models.mlp import train_mlp


def _mlp_arrays() -> dict[str, np.ndarray]:
    return {
        "X_src": np.array(
            [
                [np.nan, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "y_src": np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        "X_t_train": np.array(
            [
                [0.5, 1.0, 1.5],
                [1.0, np.nan, 2.0],
                [1.5, 2.0, 2.5],
                [2.0, 2.5, 3.0],
            ],
            dtype=np.float32,
        ),
        "y_t_train": np.array([1.2, 1.7, 2.2, 2.7], dtype=np.float32),
        "X_val": np.array([[0.7, 1.2, np.nan], [1.8, 2.2, 2.6]], dtype=np.float32),
        "y_val_used": np.array([1.3, 2.4], dtype=np.float32),
        "y_val_sec": np.array([13.0, 24.0], dtype=np.float32),
        "X_test": np.array([[0.9, 1.4, 1.9], [np.nan, 2.4, 2.9]], dtype=np.float32),
    }


def test_train_mlp_default_finetune_path_returns_finite_predictions() -> None:
    arrays = _mlp_arrays()
    out = train_mlp(
        model_cfg={
            "MLP_TL": {
                "hidden": [8, 4],
                "dropout": 0.0,
                "lr_pretrain": 0.01,
                "epochs_pretrain": 2,
                "batch_size": 2,
                "patience": 2,
                "epochs_finetune": 3,
                "lr_finetune": 0.01,
            }
        },
        seed=3,
        target_transform="gradient_norm",
        target_inv_scale=10.0,
        target_t0_sec=1.0,
        **arrays,
    )

    assert out.name == "MLP_TL"
    assert out.val_pred.shape == (2,)
    assert out.test_pred.shape == (2,)
    assert np.isfinite(out.val_pred).all()
    assert np.isfinite(out.test_pred).all()
    assert np.isfinite(list(out.val_metrics.values())).all()


def test_train_mlp_search_finetune_and_mdl_style_paths() -> None:
    arrays = _mlp_arrays()
    out = train_mlp(
        model_cfg={
            "MLP_TL": {
                "style": "mdl",
                "dropout": 0.0,
                "lr_pretrain": 0.01,
                "epochs_pretrain": 2,
                "batch_size": 2,
                "patience": 2,
                "epochs_finetune": 2,
                "lr_finetune": 0.01,
                "search_finetune": True,
                "finetune_batch_sizes": [2],
                "finetune_lrs": [0.001, 0.005],
                "finetune_epochs_grid": [1, 2],
            }
        },
        seed=5,
        target_transform="log1p",
        target_inv_scale=1.0,
        target_t0_sec=1.0,
        **arrays,
    )

    assert out.name == "MLP_TL"
    assert out.val_pred.shape == (2,)
    assert out.test_pred.shape == (2,)
    assert np.isfinite(out.val_pred).all()
    assert np.isfinite(out.test_pred).all()


def test_candidate_builder_exports_candidate_api() -> None:
    assert CandidateBuildContext is not None
    assert build_candidates is collect_candidates
