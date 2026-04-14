from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import rtbench.bench.runner as runner
import rtbench.models.candidates as candidates_module
import rtbench.models.ensemble as ensemble_module
from rtbench.config import Config
from rtbench.data import DatasetMatrix, FINGERPRINT_SIZES, build_all_matrices
from rtbench.experiments import garbage_collect_experiments, load_registry
from rtbench.models import CandidateOutput, SplitData, stratified_split
from rtbench.models.candidates import build_candidates
from rtbench.models.candidates.common import CandidateBuildContext
from rtbench.models.ensemble import EnsembleOutput


def _empty_candidate_context() -> CandidateBuildContext:
    return CandidateBuildContext(
        model_cfg={
            "ENABLE_HYPER_TL": False,
            "ONLY_HYPER_TL": False,
            "ENABLE_MLP": False,
            "ENABLE_MDL_SUBSET_CANDIDATES": False,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": False,
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": False,
            "ENABLE_ANCHOR_TL": False,
        },
        X_src=np.zeros((0, 2), dtype=np.float32),
        X_src_cp=np.zeros((0, 1), dtype=np.float32),
        y_src=np.zeros(0, dtype=np.float32),
        y_src_sec=np.zeros(0, dtype=np.float32),
        X_train=np.zeros((0, 2), dtype=np.float32),
        X_train_mol=np.zeros((0, 1), dtype=np.float32),
        X_train_cp=np.zeros((0, 1), dtype=np.float32),
        y_train=np.zeros(0, dtype=np.float32),
        y_train_sec=np.zeros(0, dtype=np.float32),
        X_val=np.zeros((0, 2), dtype=np.float32),
        X_val_mol=np.zeros((0, 1), dtype=np.float32),
        y_val_used=np.zeros(0, dtype=np.float32),
        y_val_sec=np.zeros(0, dtype=np.float32),
        X_test=np.zeros((0, 2), dtype=np.float32),
        X_test_mol=np.zeros((0, 1), dtype=np.float32),
        seed=0,
        source_weight=0.2,
        target_weight=1.0,
        group_sizes={"mol": 1, "cp": 1},
    )


def _ensemble_inputs() -> dict[str, object]:
    split = SplitData(
        train_idx=np.array([0, 1], dtype=int),
        val_idx=np.array([2, 3], dtype=int),
        test_idx=np.array([4, 5], dtype=int),
    )
    y_target = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0], dtype=np.float32)
    return {
        "model_cfg": {
            "ENABLE_HYPER_TL": False,
            "ONLY_HYPER_TL": False,
            "ENABLE_MLP": False,
            "ENABLE_MDL_SUBSET_CANDIDATES": False,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": False,
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": False,
            "ENABLE_ANCHOR_TL": False,
            "FUSION_TOP_K": 2,
            "FUSION_OBJECTIVE": "mae",
            "FUSION_RANK": "mae",
            "CALIBRATE": False,
            "CLIP_MULT": float("nan"),
        },
        "X_src": np.zeros((2, 2), dtype=np.float32),
        "X_src_mol": np.zeros((2, 2), dtype=np.float32),
        "X_src_cp": np.zeros((2, 1), dtype=np.float32),
        "y_src": np.array([90.0, 95.0], dtype=np.float32),
        "X_target": np.arange(12, dtype=np.float32).reshape(6, 2),
        "X_target_mol": np.arange(12, dtype=np.float32).reshape(6, 2),
        "X_target_cp": np.arange(6, dtype=np.float32).reshape(6, 1),
        "y_target": y_target,
        "y_target_sec": y_target,
        "split": split,
        "seed": 7,
        "source_weight": 0.2,
        "target_weight": 1.0,
        "group_sizes": {"mol": 1, "cp": 1},
    }


def _make_cfg(tmp_path: Path, *, resume: bool = True) -> Config:
    return Config(
        data={},
        datasets={"failed_override": []},
        split={"train": 0.5, "val": 0.25, "test": 0.25, "strategy": "random"},
        models={"ENABLE_FAIL_TUNING": True},
        transfer_weights={"source": 0.2, "target": 1.0, "adaptive_source": False},
        seeds={"default": "0:1"},
        metrics={"paper_avg_mae": 999.0, "paper_avg_r2": -999.0, "required_win_both": 0},
        stats={"fdr_q": 0.05},
        outputs={"root": str(tmp_path / "outputs"), "resume": resume},
    )


def _make_matrix(dataset_id: str, rows: int, mol_dim: int = 2, cp_dim: int = 2) -> DatasetMatrix:
    x_mol = np.arange(rows * mol_dim, dtype=np.float32).reshape(rows, mol_dim) + float(int(dataset_id))
    x_cp = np.tile(np.linspace(1.0, float(cp_dim), cp_dim, dtype=np.float32), (rows, 1))
    x = np.concatenate([x_mol, x_cp], axis=1)
    return DatasetMatrix(
        dataset_id=dataset_id,
        ids=[f"{dataset_id}_{i}" for i in range(rows)],
        mol_keys=[f"mol-{dataset_id}-{i}" for i in range(rows)],
        X=x,
        X_mol=x_mol,
        X_cp=x_cp,
        y_sec=np.linspace(10.0, 10.0 + rows - 1, rows, dtype=np.float32),
        y_scale_sec=120.0,
        t0_sec=10.0,
    )


def _make_prep() -> SimpleNamespace:
    src = _make_matrix("0001", rows=3)
    tgt = _make_matrix("0002", rows=6)
    return SimpleNamespace(
        external_ids=["0002"],
        pretrain_ids=["0001"],
        mats={"0001": src, "0002": tgt},
        X_src=src.X,
        X_src_mol=src.X_mol,
        X_src_cp=src.X_cp,
        y_src_sec=src.y_sec,
        source_row_dataset_ids=np.array(["0001"] * len(src.y_sec), dtype=object),
        source_row_mol_keys=np.array(src.mol_keys, dtype=object),
        source_mol_key_set=set(src.mol_keys),
        baseline_df=pd.DataFrame([{"dataset": "0002", "paper_mae": 100.0, "paper_r2": 0.5}]),
        schema=SimpleNamespace(group_sizes={"mol": src.X_mol.shape[1], "cp": src.X_cp.shape[1]}),
        hyper_cache={},
    )


def _split() -> SplitData:
    return SplitData(
        train_idx=np.array([0, 1], dtype=int),
        val_idx=np.array([2, 3], dtype=int),
        test_idx=np.array([4, 5], dtype=int),
    )


def _ensemble_output() -> EnsembleOutput:
    return EnsembleOutput(
        pred_test=np.array([42.0, 43.0], dtype=np.float32),
        pred_val=np.array([40.0, 41.0], dtype=np.float32),
        top_models=["TREE_A", "RIDGE_B"],
        weights=[0.75, 0.25],
        feature_group_importance={"mol": 0.6, "cp": 0.4},
    )


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "0002",
                "paper_mae": 100.0,
                "paper_r2": 0.5,
                "our_mae_mean": 5.0,
                "our_r2_mean": 0.9,
                "delta_mae": 95.0,
                "delta_r2": 0.4,
                "p_mae": 0.1,
                "p_r2": 0.1,
                "p_adj_mae": 0.1,
                "p_adj_r2": 0.1,
                "win_both": True,
            }
        ]
    )


def test_build_candidates_empty_dataset() -> None:
    with pytest.raises(ValueError, match="Empty target dataset"):
        build_candidates(_empty_candidate_context())


def test_build_candidates_respects_name_filters() -> None:
    ctx = CandidateBuildContext(
        model_cfg={
            "ENABLE_HYPER_TL": False,
            "ONLY_HYPER_TL": False,
            "ENABLE_MLP": False,
            "ENABLE_MDL_SUBSET_CANDIDATES": False,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": False,
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": False,
            "ENABLE_ANCHOR_TL": False,
            "CANDIDATE_NAME_ALLOWLIST": [r"^KEEP_"],
            "CANDIDATE_NAME_DENYLIST": [r"DROP$"],
        },
        X_src=np.zeros((1, 2), dtype=np.float32),
        X_src_cp=np.zeros((1, 1), dtype=np.float32),
        y_src=np.zeros(1, dtype=np.float32),
        y_src_sec=np.zeros(1, dtype=np.float32),
        X_train=np.zeros((1, 2), dtype=np.float32),
        X_train_mol=np.zeros((1, 1), dtype=np.float32),
        X_train_cp=np.zeros((1, 1), dtype=np.float32),
        y_train=np.zeros(1, dtype=np.float32),
        y_train_sec=np.zeros(1, dtype=np.float32),
        X_val=np.zeros((1, 2), dtype=np.float32),
        X_val_mol=np.zeros((1, 1), dtype=np.float32),
        y_val_used=np.zeros(1, dtype=np.float32),
        y_val_sec=np.zeros(1, dtype=np.float32),
        X_test=np.zeros((1, 2), dtype=np.float32),
        X_test_mol=np.zeros((1, 1), dtype=np.float32),
        seed=0,
        source_weight=0.2,
        target_weight=1.0,
        group_sizes={"mol": 1, "cp": 1},
    )
    fake_candidates = [
        CandidateOutput("KEEP_ALPHA", np.zeros(1), np.zeros(1), {"mae": 1.0}, {"kind": "a"}),
        CandidateOutput("KEEP_BETA_DROP", np.zeros(1), np.zeros(1), {"mae": 2.0}, {"kind": "b"}),
        CandidateOutput("SKIP_GAMMA", np.zeros(1), np.zeros(1), {"mae": 3.0}, {"kind": "c"}),
    ]
    with (
        mock.patch.object(candidates_module, "build_hyper_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_transfer_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_ridge_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_mdl_transfer_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_fail_tune_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_local_tree_candidates", return_value=fake_candidates),
        mock.patch.object(candidates_module, "build_mdl_local_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_anchor_candidates", return_value=[]),
    ):
        out = candidates_module.build_candidates(ctx)
    assert [candidate.name for candidate in out] == ["KEEP_ALPHA"]


def test_build_candidates_raises_when_name_filters_drop_everything() -> None:
    ctx = CandidateBuildContext(
        model_cfg={
            "ENABLE_HYPER_TL": False,
            "ONLY_HYPER_TL": False,
            "ENABLE_MLP": False,
            "ENABLE_MDL_SUBSET_CANDIDATES": False,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": False,
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": False,
            "ENABLE_ANCHOR_TL": False,
            "CANDIDATE_NAME_ALLOWLIST": [r"^NO_MATCH$"],
        },
        X_src=np.zeros((1, 2), dtype=np.float32),
        X_src_cp=np.zeros((1, 1), dtype=np.float32),
        y_src=np.zeros(1, dtype=np.float32),
        y_src_sec=np.zeros(1, dtype=np.float32),
        X_train=np.zeros((1, 2), dtype=np.float32),
        X_train_mol=np.zeros((1, 1), dtype=np.float32),
        X_train_cp=np.zeros((1, 1), dtype=np.float32),
        y_train=np.zeros(1, dtype=np.float32),
        y_train_sec=np.zeros(1, dtype=np.float32),
        X_val=np.zeros((1, 2), dtype=np.float32),
        X_val_mol=np.zeros((1, 1), dtype=np.float32),
        y_val_used=np.zeros(1, dtype=np.float32),
        y_val_sec=np.zeros(1, dtype=np.float32),
        X_test=np.zeros((1, 2), dtype=np.float32),
        X_test_mol=np.zeros((1, 1), dtype=np.float32),
        seed=0,
        source_weight=0.2,
        target_weight=1.0,
        group_sizes={"mol": 1, "cp": 1},
    )
    fake_candidates = [CandidateOutput("KEEP_ALPHA", np.zeros(1), np.zeros(1), {"mae": 1.0}, {"kind": "a"})]
    with (
        mock.patch.object(candidates_module, "build_hyper_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_transfer_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_ridge_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_mdl_transfer_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_fail_tune_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_local_tree_candidates", return_value=fake_candidates),
        mock.patch.object(candidates_module, "build_mdl_local_tree_candidates", return_value=[]),
        mock.patch.object(candidates_module, "build_anchor_candidates", return_value=[]),
    ):
        with pytest.raises(ValueError, match="No ensemble candidates were produced"):
            candidates_module.build_candidates(ctx)


def test_stratified_split_single_sample() -> None:
    with pytest.raises(ValueError, match="at least 2 samples"):
        stratified_split(np.array([1.0], dtype=np.float32), seed=0, train=0.8, val=0.1, test=0.1)


def test_ensemble_all_models_fail() -> None:
    bad_candidates = [
        CandidateOutput(
            name="BAD_NAN",
            val_pred=np.array([100.0, 100.0], dtype=np.float32),
            test_pred=np.array([101.0, 101.0], dtype=np.float32),
            val_metrics={"mae": float("nan"), "r2": 0.0},
            model={"kind": "bad"},
        ),
        CandidateOutput(
            name="BAD_INF",
            val_pred=np.array([110.0, 110.0], dtype=np.float32),
            test_pred=np.array([111.0, 111.0], dtype=np.float32),
            val_metrics={"mae": float("inf"), "r2": 0.0},
            model={"kind": "bad"},
        ),
    ]

    with (
        mock.patch.object(ensemble_module, "build_candidates", return_value=bad_candidates),
        pytest.raises(ValueError, match="No valid ensemble candidates"),
    ):
        ensemble_module.train_and_ensemble(**_ensemble_inputs())


def test_registry_csv_corrupted(tmp_path: Path) -> None:
    registry_path = tmp_path / "experiments" / "registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text('"run_dir,status\n"unterminated', encoding="utf-8")
    frame, resolved_path = load_registry(tmp_path, refresh=False)
    assert resolved_path == registry_path
    assert frame.empty
    assert "run_dir" in frame.columns
    assert "status" in frame.columns


def test_config_sha1_mismatch_blocks_resume(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, resume=True)
    prep = _make_prep()
    out_root = Path(cfg.outputs["root"])
    metrics_root = out_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset": "0002",
                "seed": 0,
                "mae": 1.0,
                "medae": 1.0,
                "mre": 0.1,
                "medre": 0.1,
                "r2": 0.9,
                "rmse": 1.0,
            }
        ]
    ).to_csv(metrics_root / "per_seed.csv", index=False, encoding="utf-8")
    (out_root / "config.sha1").write_text("old-sha", encoding="utf-8")

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "train_and_ensemble", return_value=_ensemble_output()) as train_mock,
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df()),
        mock.patch.object(runner, "write_report"),
    ):
        runner.run_trial(prep, cfg, seeds=[0], config_sha1="new-sha", write_predictions=False)

    assert train_mock.call_count == 1
    assert (out_root / "config.sha1").read_text(encoding="utf-8").strip() == "new-sha"


def test_resolve_dataset_model_cfg_applies_auto_policy() -> None:
    y_sec = np.concatenate(
        [
            np.linspace(11.0, 13.4, 12, dtype=np.float32),
            np.array([70.0, 72.0, 74.0], dtype=np.float32),
        ]
    )
    mat = DatasetMatrix(
        dataset_id="0002",
        ids=[f"id-{i}" for i in range(len(y_sec))],
        mol_keys=[f"m{i}" for i in range(len(y_sec))],
        X=np.zeros((len(y_sec), 2), dtype=np.float32),
        X_mol=np.zeros((len(y_sec), 1), dtype=np.float32),
        X_cp=np.zeros((len(y_sec), 1), dtype=np.float32),
        y_sec=y_sec,
        y_scale_sec=120.0,
        t0_sec=10.0,
    )
    base_cfg = {
        "CALIBRATE": True,
        "LOCAL_LGBM": {"n_estimators": 2000},
        "SINGLE_TASK_AUTO_POLICY": {
            "enabled": True,
            "small_n_max": 20,
            "high_outlier_rate": 0.10,
            "small_outlier": {
                "CALIBRATE": False,
                "CANDIDATE_NAME_ALLOWLIST": [r"^LOCAL_LGBM$"],
                "LOCAL_LGBM": {"n_estimators": 800},
            },
        },
    }

    resolved_cfg, stats = runner.resolve_dataset_model_cfg(base_cfg, mat)

    assert stats["selected_rule"] == "small_outlier"
    assert resolved_cfg["CALIBRATE"] is False
    assert resolved_cfg["LOCAL_LGBM"]["n_estimators"] == 800
    assert resolved_cfg["CANDIDATE_NAME_ALLOWLIST"] == [r"^LOCAL_LGBM$"]


def test_build_all_matrices_with_missing_fingerprint_file_uses_zero_vectors(
    tmp_path: Path,
    synthetic_repo_factory,
) -> None:
    processed_root = synthetic_repo_factory(
        tmp_path / "data",
        {"0001": 4},
        missing_fingerprints={"0001": set(FINGERPRINT_SIZES.keys())},
    )
    schema, mats = build_all_matrices(processed_root=processed_root, dataset_ids=["0001"], gradient_points=20)
    desc_dim = len(schema.descriptor_cols)
    fp_dim = sum(FINGERPRINT_SIZES.values())
    fp_slice = mats["0001"].X_mol[:, desc_dim : desc_dim + fp_dim]
    assert np.allclose(fp_slice, 0.0)
    assert np.isfinite(mats["0001"].X).all()


def test_build_all_matrices_with_zero_gradient_points(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root = synthetic_repo_factory(tmp_path / "data", {"0001": 4, "0002": 4})
    schema, mats = build_all_matrices(processed_root=processed_root, dataset_ids=["0001", "0002"], gradient_points=0)
    assert schema.gradient_points == 0
    assert schema.group_sizes["gradient_program"] == 0
    assert mats["0001"].X_cp.shape[1] == schema.cp_size
    assert np.isfinite(mats["0001"].X).all()


def test_build_all_matrices_imputes_all_nan_descriptors(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root = synthetic_repo_factory(
        tmp_path / "data",
        {"0001": 4},
        descriptor_all_nan={"0001"},
    )
    schema, mats = build_all_matrices(processed_root=processed_root, dataset_ids=["0001"], gradient_points=20)
    desc_slice = mats["0001"].X_mol[:, : len(schema.descriptor_cols)]
    assert np.allclose(desc_slice, 0.0)
    assert np.isfinite(mats["0001"].X).all()


def test_gc_no_candidates_returns_empty_payload(tmp_path: Path) -> None:
    payload = garbage_collect_experiments(tmp_path, status="tmp", dry_run=True)
    assert payload["candidate_count"] == 0
    assert payload["matching_run_count"] == 0
    assert payload["deleted_count"] == 0
    assert payload["candidate_roots"] == []
