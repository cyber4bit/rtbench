from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from rtbench.config import Config
from rtbench.data import DatasetMatrix


prepare_module = importlib.import_module("rtbench.bench.prepare")


class _FakeCPModel:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def cp_vector_for_dataset(self, ds_root: Path, ds: str) -> np.ndarray:
        return np.asarray(self._vectors[ds], dtype=np.float32)


def _make_cfg(tmp_path: Path, *, cpvec_enabled: bool = False, use_all_studies: bool = True) -> Config:
    baseline_csv = tmp_path / "baseline.csv"
    pd.DataFrame([{"dataset": "0003", "paper_mae": 1.0, "paper_r2": 0.5}]).to_csv(
        baseline_csv,
        index=False,
        encoding="utf-8",
    )
    data_cfg: dict[str, Any] = {
        "repo_url": "https://example.com/repo",
        "commit": "deadbeef",
        "local_root": str(tmp_path / "repoRT"),
        "baseline_csv": str(baseline_csv),
        "gradient_points": 20,
    }
    if cpvec_enabled:
        data_cfg["cpvec"] = {"enabled": True, "use_all_studies": use_all_studies}
    return Config(
        data=data_cfg,
        datasets={"pretrain": ["0001", "0002"], "external": ["0003"], "expected_pretrain_count": 5},
        split={"train": 0.8, "val": 0.1, "test": 0.1},
        models={},
        transfer_weights={"source": 0.2, "target": 1.0},
        seeds={"default": "0:1"},
        metrics={"paper_avg_mae": 100.0, "paper_avg_r2": 0.5, "required_win_both": 0},
        stats={"fdr_q": 0.05},
        outputs={"root": str(tmp_path / "outputs"), "resume": True},
    )


def _make_matrix(dataset_id: str, rows: int, width: int = 5, cp_dim: int = 2) -> DatasetMatrix:
    X_mol = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) + float(int(dataset_id))
    X_cp = np.tile(np.linspace(1.0, float(cp_dim), cp_dim, dtype=np.float32), (rows, 1))
    X = np.concatenate([X_mol, np.full((rows, width - X_mol.shape[1]), 0.5, dtype=np.float32)], axis=1)
    return DatasetMatrix(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}_{i}" for i in range(rows)],
        mol_keys=[f"mol-{dataset_id}-{i}" for i in range(rows)],
        X=X,
        X_mol=X_mol,
        X_cp=X_cp,
        y_sec=np.linspace(10.0, 10.0 + rows - 1, rows, dtype=np.float32),
        y_scale_sec=120.0,
        t0_sec=10.0,
    )


def test_prepare_keeps_schema_and_concatenates_source_matrices(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    schema = SimpleNamespace(group_sizes={"descriptor": 2, "fingerprint": 2, "meta": 1})
    mats = {
        "0001": _make_matrix("0001", rows=2),
        "0002": _make_matrix("0002", rows=3),
        "0003": _make_matrix("0003", rows=4),
    }

    with (
        mock.patch.object(prepare_module, "ensure_repo_data"),
        mock.patch.object(prepare_module, "validate_required_inputs"),
        mock.patch.object(prepare_module, "build_all_matrices", return_value=(schema, mats)) as build_mock,
        mock.patch.object(prepare_module, "pretrain_count_14", return_value=5),
    ):
        prep = prepare_module.prepare(cfg, no_download=True)

    assert prep.schema is schema
    assert prep.all_ids == ["0001", "0002", "0003"]
    assert prep.X_src.shape == (5, mats["0001"].X.shape[1])
    assert prep.X_src_mol.shape == (5, mats["0001"].X_mol.shape[1])
    assert prep.X_src_cp.shape == (5, mats["0001"].X_cp.shape[1])
    assert prep.source_row_dataset_ids.tolist() == ["0001", "0001", "0002", "0002", "0002"]
    assert prep.source_row_mol_keys.tolist()[0] == "mol-0001-0"
    assert prep.source_mol_key_set == set(prep.source_row_mol_keys.tolist())
    assert prep.baseline_df["dataset"].tolist() == ["0003"]
    assert build_mock.call_args.kwargs["gradient_points"] == 20
    assert build_mock.call_args.kwargs["cpvec_map"] is None


def test_prepare_builds_cpvec_map_for_cache_miss_style_flow(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, cpvec_enabled=True, use_all_studies=True)
    schema = SimpleNamespace(group_sizes={"descriptor": 2, "fingerprint": 2, "meta": 1, "cpvec": 2})
    mats = {
        "0001": _make_matrix("0001", rows=2),
        "0002": _make_matrix("0002", rows=3),
        "0003": _make_matrix("0003", rows=4),
    }
    cp_model = _FakeCPModel(
        {
            "0001": np.array([1.0, 0.0], dtype=np.float32),
            "0002": np.array([0.0, 1.0], dtype=np.float32),
            "0003": np.array([1.0, 1.0], dtype=np.float32),
        }
    )

    with (
        mock.patch.object(prepare_module, "ensure_repo_data"),
        mock.patch.object(prepare_module, "validate_required_inputs"),
        mock.patch.object(prepare_module, "list_all_study_ids", return_value=["0001", "0002", "0003", "0999"]) as list_mock,
        mock.patch.object(prepare_module, "ensure_cp_inputs") as ensure_cp_inputs_mock,
        mock.patch.object(prepare_module, "load_or_train_cpvec", return_value=(cp_model, 2)) as cpvec_mock,
        mock.patch.object(prepare_module, "build_all_matrices", return_value=(schema, mats)) as build_mock,
        mock.patch.object(prepare_module, "pretrain_count_14", return_value=5),
    ):
        prep = prepare_module.prepare(cfg, no_download=True)

    assert prep.X_src.shape[0] == 5
    list_mock.assert_called_once()
    ensure_cp_inputs_mock.assert_called_once()
    assert ensure_cp_inputs_mock.call_args.kwargs["dataset_ids"] == ["0001", "0002", "0003", "0999"]
    cpvec_mock.assert_called_once()
    cpvec_map = build_mock.call_args.kwargs["cpvec_map"]
    assert set(cpvec_map.keys()) == {"0001", "0002", "0003"}
    assert np.array_equal(cpvec_map["0003"], np.array([1.0, 1.0], dtype=np.float32))


def test_prepare_builds_cpvec_map_for_cache_hit_style_flow(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, cpvec_enabled=True, use_all_studies=False)
    schema = SimpleNamespace(group_sizes={"descriptor": 2, "fingerprint": 2, "meta": 1, "cpvec": 2})
    mats = {
        "0001": _make_matrix("0001", rows=2),
        "0002": _make_matrix("0002", rows=3),
        "0003": _make_matrix("0003", rows=4),
    }
    cp_model = _FakeCPModel(
        {
            "0001": np.array([2.0, 0.0], dtype=np.float32),
            "0002": np.array([0.0, 2.0], dtype=np.float32),
            "0003": np.array([2.0, 2.0], dtype=np.float32),
        }
    )

    with (
        mock.patch.object(prepare_module, "ensure_repo_data"),
        mock.patch.object(prepare_module, "validate_required_inputs"),
        mock.patch.object(prepare_module, "list_all_study_ids") as list_mock,
        mock.patch.object(prepare_module, "ensure_cp_inputs") as ensure_cp_inputs_mock,
        mock.patch.object(prepare_module, "load_or_train_cpvec", return_value=(cp_model, 2)),
        mock.patch.object(prepare_module, "build_all_matrices", return_value=(schema, mats)) as build_mock,
        mock.patch.object(prepare_module, "pretrain_count_14", return_value=5),
    ):
        prepare_module.prepare(cfg, no_download=True)

    list_mock.assert_not_called()
    assert ensure_cp_inputs_mock.call_args.kwargs["dataset_ids"] == ["0001", "0002", "0003"]
    cpvec_map = build_mock.call_args.kwargs["cpvec_map"]
    assert np.array_equal(cpvec_map["0001"], np.array([2.0, 0.0], dtype=np.float32))
