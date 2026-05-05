from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import rtbench.bench.unified_hypertl as unified_hypertl
from rtbench.bench.unified_cv import assemble_unified_cv_fold, run_unified_cv
from rtbench.config import Config


def _matrix(dataset_id: str, n_rows: int, offset: int) -> SimpleNamespace:
    base = np.arange(n_rows, dtype=np.float32) + float(offset)
    return SimpleNamespace(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}-row-{i}" for i in range(n_rows)],
        X=np.stack([base, base + 10.0, base + 20.0], axis=1),
        X_mol=np.stack([base + 100.0, base + 200.0], axis=1),
        X_cp=np.stack([base + 300.0, np.full(n_rows, offset, dtype=np.float32)], axis=1),
        y_sec=base + 1.0,
    )


def _mats() -> dict[str, SimpleNamespace]:
    return {
        "0001": _matrix("0001", 9, 0),
        "0002": _matrix("0002", 9, 1000),
    }


def test_unified_hypertl_pretrains_once_on_pooled_train_val_and_predicts_pooled_splits(monkeypatch) -> None:
    mats = _mats()
    fold = assemble_unified_cv_fold(mats, mats.keys(), 1, n_splits=3, shuffle_seed=17)
    calls: dict[str, Any] = {"pretrain": [], "predict": []}

    class FakeBundle:
        pass

    def fake_pretrain_hyper_tl(
        X_src_mol,
        X_src_cp,
        y_src,
        cfg,
        seed=0,
        train_idx=None,
        val_idx=None,
        **kwargs,
    ):
        calls["pretrain"].append(
            {
                "X_src_mol": np.asarray(X_src_mol).copy(),
                "X_src_cp": np.asarray(X_src_cp).copy(),
                "y_src": np.asarray(y_src).copy(),
                "cfg": dict(cfg),
                "seed": seed,
                "train_idx": np.asarray(train_idx).copy(),
                "val_idx": np.asarray(val_idx).copy(),
                "kwargs": kwargs,
            }
        )
        return FakeBundle()

    def fake_predict_hypertl_bundle(bundle, X_mol, X_cp, *, batch_size):
        calls["predict"].append(
            {
                "X_mol": np.asarray(X_mol).copy(),
                "X_cp": np.asarray(X_cp).copy(),
                "batch_size": batch_size,
            }
        )
        return np.asarray(X_cp, dtype=np.float32)[:, 0] * 0.0 + len(calls["predict"])

    monkeypatch.setattr(unified_hypertl, "pretrain_hyper_tl", fake_pretrain_hyper_tl)
    monkeypatch.setattr(unified_hypertl, "predict_hypertl_bundle", fake_predict_hypertl_bundle)

    fit_predict = unified_hypertl.build_unified_hypertl_fit_predict(
        {
            "UNIFIED_CV_HYPER_TL": {
                "epochs": 2,
                "batch_size": 4,
                "use_task_adapters": True,
                "use_cross_stitch": True,
            },
            "UNIFIED_CV_FORCE_CP_ONLY": True,
        },
        seed=100,
    )
    pred = fit_predict(fold)

    assert len(calls["pretrain"]) == 1
    pretrain = calls["pretrain"][0]
    assert np.array_equal(pretrain["X_src_mol"], np.concatenate([fold.X_train_mol, fold.X_val_mol], axis=0))
    assert np.array_equal(pretrain["X_src_cp"], np.concatenate([fold.X_train_cp, fold.X_val_cp], axis=0))
    assert np.array_equal(pretrain["y_src"], np.concatenate([fold.y_train_sec, fold.y_val_sec], axis=0))
    assert np.array_equal(pretrain["train_idx"], np.arange(len(fold.y_train_sec)))
    assert np.array_equal(
        pretrain["val_idx"],
        np.arange(len(fold.y_train_sec), len(fold.y_train_sec) + len(fold.y_val_sec)),
    )
    assert pretrain["cfg"]["n_models"] == 1
    assert pretrain["cfg"]["use_task_adapters"] is False
    assert pretrain["cfg"]["use_cross_stitch"] is False
    assert pretrain["seed"] == 101

    assert len(calls["predict"]) == 2
    assert np.array_equal(calls["predict"][0]["X_mol"], fold.X_val_mol)
    assert np.array_equal(calls["predict"][1]["X_mol"], fold.X_test_mol)
    assert np.array_equal(pred.val_pred_sec, np.ones(len(fold.y_val_sec), dtype=np.float32))
    assert np.array_equal(pred.test_pred_sec, np.full(len(fold.y_test_sec), 2.0, dtype=np.float32))
    assert pred.model_info["learner"] == "unified_hypertl"
    assert pred.model_info["n_model"] == 1


def test_run_unified_cv_can_select_strict_unified_hypertl_from_config(tmp_path, monkeypatch) -> None:
    mats = _mats()
    pretrain_calls: list[dict[str, Any]] = []

    class FakeBundle:
        pass

    def fake_pretrain_hyper_tl(X_src_mol, X_src_cp, y_src, cfg, seed=0, train_idx=None, val_idx=None, **kwargs):
        pretrain_calls.append(
            {
                "n_rows": len(y_src),
                "train_idx": np.asarray(train_idx).copy(),
                "val_idx": np.asarray(val_idx).copy(),
                "seed": seed,
                "cfg": dict(cfg),
            }
        )
        return FakeBundle()

    def fake_predict_hypertl_bundle(bundle, X_mol, X_cp, *, batch_size):
        return np.asarray(X_cp, dtype=np.float32)[:, 0]

    monkeypatch.setattr(unified_hypertl, "pretrain_hyper_tl", fake_pretrain_hyper_tl)
    monkeypatch.setattr(unified_hypertl, "predict_hypertl_bundle", fake_predict_hypertl_bundle)

    cfg = Config(
        data={},
        datasets={"external": list(mats)},
        split={"strategy": "kfold", "folds": 3, "shuffle_seed": 5},
        models={
            "ONLY_HYPER_TL": True,
            "ENABLE_HYPER_TL": True,
            "UNIFIED_CV_HYPER_TL": {"epochs": 1, "batch_size": 8},
        },
        transfer_weights={},
        seeds={"default": "0:2"},
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
    assert len(pretrain_calls) == 3
    assert all(call["cfg"]["n_models"] == 1 for call in pretrain_calls)
    assert all(call["n_rows"] == len(call["train_idx"]) + len(call["val_idx"]) for call in pretrain_calls)
    assert result.predictions_path.exists()
    predictions = pd.read_csv(result.predictions_path, dtype={"dataset": str})
    assert len(predictions) == sum(len(mat.y_sec) for mat in mats.values())
