from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rtbench.bench.unified_cv import (
    assemble_unified_cv_fold,
    evaluate_strict_unified_cv,
    hyper_tl_validation_index_kwargs,
    iter_unified_cv_folds,
    prediction_frame,
    run_unified_cv,
)
from rtbench.config import Config
from rtbench.models import kfold_split


def _matrix(dataset_id: str, n_rows: int, offset: int) -> SimpleNamespace:
    base = np.arange(n_rows, dtype=np.float32) + float(offset)
    return SimpleNamespace(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}-row-{i}" for i in range(n_rows)],
        X=np.stack([base, base + 100.0, base + 200.0], axis=1),
        X_mol=np.stack([base + 10.0, base + 20.0], axis=1),
        X_cp=np.stack([base + 30.0], axis=1),
        y_sec=base + 1.0,
    )


def _mats() -> dict[str, SimpleNamespace]:
    return {
        "0001": _matrix("0001", 21, 0),
        "0002": _matrix("0002", 23, 1000),
        "0003": _matrix("0003", 25, 2000),
    }


def test_assemble_unified_cv_fold_matches_kfold_split_semantics() -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, ["1", "0002", "0003"], 3, shuffle_seed=123)

    assert fold.dataset_ids == ("0001", "0002", "0003")
    assert fold.fold_id == 3
    assert fold.X_train.shape[0] == len(fold.train_meta) == len(fold.y_train_sec)
    assert fold.X_val.shape[0] == len(fold.val_meta) == len(fold.y_val_sec)
    assert fold.X_test.shape[0] == len(fold.test_meta) == len(fold.y_test_sec)
    assert {row.split_name for row in fold.train_meta} == {"train"}
    assert {row.split_name for row in fold.val_meta} == {"val"}
    assert {row.split_name for row in fold.test_meta} == {"test"}

    for dataset_id in fold.dataset_ids:
        expected = kfold_split(mats[dataset_id].y_sec, seed=3, n_splits=10, shuffle_seed=123)
        actual = fold.dataset_indices[dataset_id]
        assert np.array_equal(actual.train_idx, expected.train_idx)
        assert np.array_equal(actual.val_idx, expected.val_idx)
        assert np.array_equal(actual.test_idx, expected.test_idx)

        ds_slice = fold.test_slices[dataset_id]
        meta_idx = [row.local_row_index for row in fold.test_meta[ds_slice]]
        assert meta_idx == expected.test_idx.tolist()


def test_iter_unified_cv_folds_covers_each_dataset_test_row_once() -> None:
    mats = _mats()
    seen = {dataset_id: [] for dataset_id in mats}

    for fold in iter_unified_cv_folds(mats, mats.keys(), shuffle_seed=11):
        for row in fold.test_meta:
            seen[row.dataset_id].append(row.local_row_index)

    for dataset_id, rows in seen.items():
        assert sorted(rows) == list(range(len(mats[dataset_id].y_sec)))


def test_evaluate_strict_unified_cv_calls_fit_predict_once_per_fold() -> None:
    mats = _mats()
    calls: list[int] = []

    def fit_predict(fold):
        calls.append(fold.fold_id)
        assert fold.X_train.shape[0] == len(fold.y_train_sec)
        assert fold.X_val.shape[0] == len(fold.y_val_sec)
        assert fold.X_test.shape[0] == len(fold.y_test_sec)
        return {
            "val_pred_sec": np.full(len(fold.y_val_sec), float(np.mean(fold.y_train_sec)), dtype=np.float32),
            "test_pred_sec": np.arange(len(fold.y_test_sec), dtype=np.float32),
            "model_info": {"kind": "single-pooled-callback"},
        }

    predictions = evaluate_strict_unified_cv(mats, mats.keys(), fit_predict, shuffle_seed=7)
    assert calls == list(range(10))
    assert len(predictions) == 10
    assert all(pred.model_info["kind"] == "single-pooled-callback" for pred in predictions)

    df = prediction_frame(predictions)
    assert len(df) == sum(len(mat.y_sec) for mat in mats.values())
    assert sorted(df["dataset"].unique().tolist()) == sorted(mats)
    assert set(df["split"].unique().tolist()) == {"test"}


def test_hyper_tl_validation_index_kwargs_address_concatenated_train_val_pool() -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 0)
    kwargs = hyper_tl_validation_index_kwargs(fold)

    assert np.array_equal(kwargs["train_idx"], np.arange(len(fold.y_train_sec)))
    assert np.array_equal(
        kwargs["val_idx"],
        np.arange(len(fold.y_train_sec), len(fold.y_train_sec) + len(fold.y_val_sec)),
    )


def test_run_unified_cv_writes_report_compatible_per_seed(tmp_path: Path) -> None:
    mats = _mats()
    cfg = Config(
        data={},
        datasets={"external": list(mats)},
        split={"strategy": "kfold", "folds": 10, "shuffle_seed": 5},
        models={"UNIFIED_CV_RIDGE_ALPHA": 0.5},
        transfer_weights={},
        seeds={"default": "0:9"},
        metrics={"unirt_mode": "RPLC"},
        stats={},
        outputs={"root": str(tmp_path / "unused")},
    )
    prepared = SimpleNamespace(external_ids=list(mats), mats=mats)

    result = run_unified_cv(
        sheet="S4",
        mode="RPLC",
        config=cfg,
        prepared=prepared,
        output_root=tmp_path / "unified",
        no_download=True,
    )

    assert result.n_model == 1
    assert result.per_seed_path.exists()
    assert result.predictions_path.exists()
    assert set(["dataset", "seed", "mae", "medae", "mre", "r2"]).issubset(result.per_seed_df.columns)
    assert len(result.per_seed_df) == len(mats) * 10

    reloaded = pd.read_csv(result.per_seed_path, dtype={"dataset": str}, encoding="utf-8")
    assert reloaded["dataset"].str.len().eq(4).all()
    assert sorted(reloaded["seed"].unique().tolist()) == list(range(10))
