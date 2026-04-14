from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rtbench.bench.weighting import build_adaptive_source_weights, normalize_target_transform
from rtbench.models.trees import _forward_target, _inverse_target


def test_build_adaptive_source_weights_uses_cosine_similarity_and_topk() -> None:
    mats = {
        "0001": SimpleNamespace(X_cp=np.array([[1.0, 0.0]], dtype=np.float32), y_sec=np.array([10.0, 11.0])),
        "0002": SimpleNamespace(X_cp=np.array([[0.5, 0.5]], dtype=np.float32), y_sec=np.array([12.0, 13.0, 14.0])),
        "0003": SimpleNamespace(X_cp=np.array([[-1.0, 0.0]], dtype=np.float32), y_sec=np.array([15.0])),
        "0100": SimpleNamespace(X_cp=np.array([[1.0, 0.0]], dtype=np.float32), y_sec=np.array([20.0])),
    }
    source_row_dataset_ids = np.array(["0001", "0002", "0002", "0003"], dtype=object)

    weights = build_adaptive_source_weights(
        pretrain_ids=["0001", "0002", "0003"],
        mats=mats,
        source_row_dataset_ids=source_row_dataset_ids,
        target_ds="0100",
        base_source_weight=0.2,
        similarity_power=1.0,
        min_scale=0.25,
        max_scale=2.0,
        mode="per_sample",
        top_k_sources=2,
    )

    assert weights[0] > weights[1]
    assert weights[1] == weights[2]
    assert weights[3] == 0.0


def test_build_adaptive_source_weights_supports_per_dataset_scaling() -> None:
    mats = {
        "0001": SimpleNamespace(X_cp=np.array([[1.0, 0.0]], dtype=np.float32), y_sec=np.array([10.0, 11.0])),
        "0002": SimpleNamespace(X_cp=np.array([[1.0, 0.0]], dtype=np.float32), y_sec=np.array([12.0, 13.0, 14.0, 15.0])),
        "0100": SimpleNamespace(X_cp=np.array([[1.0, 0.0]], dtype=np.float32), y_sec=np.array([20.0])),
    }
    source_row_dataset_ids = np.array(["0001", "0001", "0002", "0002", "0002", "0002"], dtype=object)

    weights = build_adaptive_source_weights(
        pretrain_ids=["0001", "0002"],
        mats=mats,
        source_row_dataset_ids=source_row_dataset_ids,
        target_ds="0100",
        base_source_weight=0.4,
        similarity_power=1.0,
        min_scale=1.0,
        max_scale=1.0,
        mode="per_dataset",
    )

    assert np.allclose(weights[:2], np.array([0.2, 0.2], dtype=np.float32))
    assert np.allclose(weights[2:], np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32))


def test_normalize_target_transform_aliases() -> None:
    assert normalize_target_transform({"target_normalize": True}) == "gradient_norm"
    assert normalize_target_transform({"target_transform": "gradnorm"}) == "gradient_norm"
    assert normalize_target_transform({"target_transform": "log_k"}) == "logk"
    assert normalize_target_transform({"target_transform": "logrt"}) == "log1p"
    assert normalize_target_transform({"target_transform": "none"}) == "none"


def test_target_transform_roundtrip_consistency() -> None:
    y_sec = np.array([30.0, 60.0, 120.0], dtype=np.float32)
    for transform, inv_scale, t0_sec in (
        ("gradient_norm", 240.0, 10.0),
        ("logk", 1.0, 12.0),
        ("log1p", 1.0, 1.0),
    ):
        used = _forward_target(y_sec, transform, inv_scale, t0_sec)
        restored = _inverse_target(used, transform, inv_scale, t0_sec)
        assert np.allclose(restored, y_sec, atol=1e-5)
