from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import rtbench.bench.runner as runner
from rtbench.config import Config
from rtbench.data import DatasetMatrix
from rtbench.models import SplitData
from rtbench.models.ensemble import EnsembleOutput


def _make_cfg(tmp_path: Path, *, resume: bool = True, failed_override: list[str] | None = None) -> Config:
    return Config(
        data={},
        datasets={"failed_override": failed_override or []},
        split={"train": 0.5, "val": 0.25, "test": 0.25, "strategy": "random"},
        models={"ENABLE_FAIL_TUNING": True},
        transfer_weights={"source": 0.2, "target": 1.0, "adaptive_source": False},
        seeds={"default": "0:1"},
        metrics={"paper_avg_mae": 999.0, "paper_avg_r2": -999.0, "required_win_both": 0},
        stats={"fdr_q": 0.05},
        outputs={"root": str(tmp_path / "outputs"), "resume": resume},
    )


def _make_matrix(dataset_id: str, rows: int, mol_dim: int = 2, cp_dim: int = 2) -> DatasetMatrix:
    X_mol = np.arange(rows * mol_dim, dtype=np.float32).reshape(rows, mol_dim) + float(int(dataset_id))
    X_cp = np.tile(np.linspace(1.0, float(cp_dim), cp_dim, dtype=np.float32), (rows, 1))
    X = np.concatenate([X_mol, X_cp], axis=1)
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


def _make_prep() -> Any:
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


def _make_multi_prep(
    *,
    pretrain_specs: list[tuple[str, int]] | None = None,
    external_specs: list[tuple[str, int]] | None = None,
    mol_dim: int = 2,
) -> Any:
    pretrain_specs = pretrain_specs or [("0001", 3)]
    external_specs = external_specs or [("0002", 6)]

    mats = {}
    for dataset_id, rows in pretrain_specs + external_specs:
        mats[dataset_id] = _make_matrix(dataset_id, rows=rows, mol_dim=mol_dim, cp_dim=2)

    pretrain_ids = [dataset_id for dataset_id, _ in pretrain_specs]
    external_ids = [dataset_id for dataset_id, _ in external_specs]
    src_mats = [mats[dataset_id] for dataset_id in pretrain_ids]

    return SimpleNamespace(
        external_ids=external_ids,
        pretrain_ids=pretrain_ids,
        mats=mats,
        X_src=np.concatenate([mat.X for mat in src_mats], axis=0),
        X_src_mol=np.concatenate([mat.X_mol for mat in src_mats], axis=0),
        X_src_cp=np.concatenate([mat.X_cp for mat in src_mats], axis=0),
        y_src_sec=np.concatenate([mat.y_sec for mat in src_mats], axis=0),
        source_row_dataset_ids=np.concatenate(
            [np.array([mat.dataset_id] * len(mat.y_sec), dtype=object) for mat in src_mats],
            axis=0,
        ),
        source_row_mol_keys=np.concatenate([np.array(mat.mol_keys, dtype=object) for mat in src_mats], axis=0),
        source_mol_key_set={key for mat in src_mats for key in mat.mol_keys},
        baseline_df=pd.DataFrame(
            [{"dataset": dataset_id, "paper_mae": 100.0, "paper_r2": 0.5} for dataset_id in external_ids]
        ),
        schema=SimpleNamespace(group_sizes={"descriptor": 2, "cp": 2}),
        hyper_cache={},
    )


def _summary_df(win_both: bool = True) -> pd.DataFrame:
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
                "win_both": win_both,
            }
        ]
    )


def _metrics(mae: float = 1.0, r2: float = 0.9) -> dict[str, float]:
    return {
        "mae": mae,
        "medae": mae,
        "mre": 0.1,
        "medre": 0.1,
        "r2": r2,
        "rmse": mae + 0.5,
    }


def _ensemble_output() -> EnsembleOutput:
    return EnsembleOutput(
        pred_test=np.array([42.0, 43.0], dtype=np.float32),
        pred_val=np.array([40.0, 41.0], dtype=np.float32),
        top_models=["TREE_A", "RIDGE_B"],
        weights=[0.75, 0.25],
        feature_group_importance={"mol": 0.6, "cp": 0.4},
    )


def _split() -> SplitData:
    return SplitData(
        train_idx=np.array([0, 1]),
        val_idx=np.array([2, 3]),
        test_idx=np.array([4, 5]),
    )


def test_runner_helper_functions_cover_edge_cases(tmp_path: Path) -> None:
    assert runner.aggregate_group_importance([]) == {}
    assert runner.aggregate_group_importance([{"mol": 0.2, "cp": 0.4}, {"mol": 0.4}]) == {"cp": 0.2, "mol": 0.30000000000000004}

    missing_path = tmp_path / "missing.csv"
    assert runner.load_previous_failed(missing_path, ["0002"]) == set()

    bad_path = tmp_path / "bad.csv"
    bad_path.write_text('"unterminated', encoding="utf-8")
    assert runner.load_previous_failed(bad_path, ["0002"]) == set()

    missing_cols = tmp_path / "missing_cols.csv"
    pd.DataFrame([{"dataset": "0002", "status": "failed"}]).to_csv(missing_cols, index=False, encoding="utf-8")
    assert runner.load_previous_failed(missing_cols, ["0002"]) == set()

    valid_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {"dataset": "2", "win_both": False},
            {"dataset": "9", "win_both": True},
            {"dataset": "8", "win_both": False},
        ]
    ).to_csv(valid_path, index=False, encoding="utf-8")
    assert runner.load_previous_failed(valid_path, ["0002", "0009"]) == {"0002"}

    empty_df = pd.DataFrame(columns=["dataset", "seed", "mae", "medae", "mre", "medre", "r2", "rmse"])
    assert runner.write_per_seed_csv(empty_df, tmp_path / "metrics" / "per_seed.csv", ["0002"]).empty

    written = runner.write_per_seed_csv(
        pd.DataFrame(
            [
                {"dataset": "2", "seed": 1, "mae": 2.0, "medae": 2.0, "mre": 0.2, "medre": 0.2, "r2": 0.7, "rmse": 2.5, "top_models": "OLD"},
                {"dataset": "0002", "seed": 0, "mae": 1.0, "medae": 1.0, "mre": 0.1, "medre": 0.1, "r2": 0.8, "rmse": 1.5, "top_models": "A"},
                {"dataset": "0002", "seed": 1, "mae": 0.5, "medae": 0.5, "mre": 0.05, "medre": 0.05, "r2": 0.9, "rmse": 1.0, "top_models": "NEW"},
                {"dataset": "0004", "seed": 0, "mae": 9.0, "medae": 9.0, "mre": 0.9, "medre": 0.9, "r2": 0.1, "rmse": 9.5, "top_models": "DROP"},
            ]
        ),
        tmp_path / "metrics" / "per_seed.csv",
        ["0002"],
    )
    assert written["dataset"].tolist() == ["0002", "0002"]
    assert written["seed"].tolist() == [0, 1]
    assert written["top_models"].tolist() == ["A", "NEW"]


def test_run_trial_normal_flow_writes_outputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, resume=False)
    prep = _make_prep()
    summary = _summary_df()

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "train_and_ensemble", return_value=_ensemble_output()) as train_mock,
        mock.patch.object(runner, "summarize_vs_paper", return_value=summary),
        mock.patch.object(runner, "write_report") as report_mock,
    ):
        result = runner.run_trial(prep, cfg, seeds=[0], config_sha1="sha-normal", write_predictions=True)

    out_root = Path(cfg.outputs["root"])
    per_seed = pd.read_csv(out_root / "metrics" / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    assert train_mock.call_count == 1
    assert report_mock.call_count == 1
    assert result.out_root == out_root
    assert list(per_seed["dataset"]) == ["0002"]
    assert list(per_seed["seed"]) == [0]
    assert per_seed.loc[0, "top_models"] == "TREE_A,RIDGE_B"
    assert (out_root / "predictions" / "0002" / "seed_0.csv").exists()
    assert (out_root / "metrics" / "summary_vs_paper.csv").exists()
    assert (out_root / "config.sha1").read_text(encoding="utf-8").strip() == "sha-normal"


def test_run_trial_resume_only_runs_pending_seeds(tmp_path: Path) -> None:
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
    (out_root / "config.sha1").write_text("sha-resume", encoding="utf-8")

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "train_and_ensemble", return_value=_ensemble_output()) as train_mock,
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df()),
        mock.patch.object(runner, "write_report"),
    ):
        runner.run_trial(prep, cfg, seeds=[0, 1], config_sha1="sha-resume", write_predictions=False)

    per_seed = pd.read_csv(metrics_root / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    assert train_mock.call_count == 1
    assert sorted(per_seed["seed"].tolist()) == [0, 1]


def test_run_trial_uses_fail_tuning_for_marked_datasets(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, resume=False, failed_override=["0002"])
    prep = _make_prep()

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df(win_both=False)),
        mock.patch.object(runner, "write_report"),
    ):
        seen: list[bool] = []

        def _train(**kwargs: Any) -> EnsembleOutput:
            seen.append(bool(kwargs["fail_tune"]))
            return _ensemble_output()

        with mock.patch.object(runner, "train_and_ensemble", side_effect=_train):
            runner.run_trial(prep, cfg, seeds=[0], config_sha1="sha-fail", write_predictions=False)

    assert seen == [True]


def test_run_trial_disables_resume_when_config_sha_changes(tmp_path: Path) -> None:
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
                "mae": 99.0,
                "medae": 99.0,
                "mre": 9.9,
                "medre": 9.9,
                "r2": -9.0,
                "rmse": 99.0,
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

    per_seed = pd.read_csv(metrics_root / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    assert train_mock.call_count == 1
    assert per_seed["mae"].tolist() != [99.0]
    assert (out_root / "config.sha1").read_text(encoding="utf-8").strip() == "new-sha"


def test_run_trial_recovers_from_corrupt_resume_and_uses_gradient_transform(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, resume=True)
    cfg.split["strategy"] = "stratified"
    cfg.transfer_weights.update(
        {
            "source_weight_mode": "surprising-mode",
            "top_k_sources": "not-an-int",
            "target_transform": "gradient_norm",
        }
    )
    prep = _make_prep()
    out_root = Path(cfg.outputs["root"])
    metrics_root = out_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.sha1").write_text("sha-grad", encoding="utf-8")

    train_calls: list[dict[str, Any]] = []

    def _train(**kwargs: Any) -> EnsembleOutput:
        train_calls.append(kwargs)
        return _ensemble_output()

    with (
        mock.patch.object(runner.pd, "read_csv", side_effect=ValueError("bad resume csv")),
        mock.patch.object(runner, "stratified_split", return_value=_split()),
        mock.patch.object(runner, "train_and_ensemble", side_effect=_train),
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df()),
        mock.patch.object(runner, "write_report"),
    ):
        runner.run_trial(prep, cfg, seeds=[0], config_sha1="sha-grad", write_predictions=False)

    assert len(train_calls) == 1
    assert np.allclose(train_calls[0]["source_sample_weights"], np.full(3, 0.2, dtype=np.float32))
    assert np.allclose(train_calls[0]["y_src"], prep.y_src_sec / prep.mats["0001"].y_scale_sec)
    assert np.allclose(train_calls[0]["y_target"], prep.mats["0002"].y_sec / prep.mats["0002"].y_scale_sec)
    assert train_calls[0]["target_inv_scale"] == prep.mats["0002"].y_scale_sec


def test_run_trial_uses_per_dataset_weights_and_log1p_targets(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, resume=False)
    cfg.transfer_weights.update({"source_weight_mode": "per_dataset", "target_transform": "log1p"})
    prep = _make_prep()

    train_calls: list[dict[str, Any]] = []

    def _train(**kwargs: Any) -> EnsembleOutput:
        train_calls.append(kwargs)
        return _ensemble_output()

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "train_and_ensemble", side_effect=_train),
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df()),
        mock.patch.object(runner, "write_report"),
    ):
        runner.run_trial(prep, cfg, seeds=[0], config_sha1="sha-log1p", write_predictions=False)

    assert len(train_calls) == 1
    assert np.allclose(train_calls[0]["source_sample_weights"], np.full(3, 0.2 / 3.0, dtype=np.float32))
    assert np.allclose(train_calls[0]["y_src"], np.log1p(prep.y_src_sec))
    assert np.allclose(train_calls[0]["y_target"], np.log1p(prep.mats["0002"].y_sec))
    assert train_calls[0]["target_inv_scale"] == 1.0


def test_run_trial_hyper_overlap_logk_and_cache_reuse(tmp_path: Path) -> None:
    prep = _make_multi_prep(
        pretrain_specs=[("0001", 3), ("0004", 2)],
        external_specs=[("0002", 6), ("0003", 6)],
        mol_dim=170,
    )
    cfg = _make_cfg(tmp_path, resume=False)
    cfg.outputs["root"] = str(tmp_path / "hyper_first")
    cfg.datasets["failed_override"] = []
    cfg.models.update(
        {
            "ENABLE_FAIL_TUNING": False,
            "ENABLE_HYPER_TL": True,
            "HYPER_TL": {
                "n_models": 2,
                "use_mdl_subset_mol": True,
                "balance_pretrain_by_dataset": True,
                "ridge_lambdas": [0.5, 1.5],
                "ridge_lambda_b": 0.25,
            },
        }
    )
    cfg.transfer_weights.update(
        {
            "source": 0.4,
            "adaptive_source": True,
            "top_k_sources": 1,
            "target_transform": "logk",
            "overlap_adaptive_source": True,
            "overlap_ref": 0.5,
            "overlap_power": 1.0,
            "overlap_min_scale": 0.1,
            "overlap_max_scale": 2.0,
            "overlap_disable_threshold": 0.6,
        }
    )
    cfg.metrics.update({"paper_avg_mae": 100.0, "paper_avg_r2": 0.7, "required_win_both": 0})

    pretrain_calls: list[dict[str, Any]] = []
    train_calls: list[dict[str, Any]] = []

    def _fake_pretrain_hyper_tl(**kwargs: Any) -> Any:
        pretrain_calls.append(kwargs)
        return SimpleNamespace(
            model=f"hyper-{kwargs['seed']}",
            device="cpu",
            mol_mean=np.zeros(kwargs["X_src_mol"].shape[1], dtype=np.float32),
            mol_std=np.ones(kwargs["X_src_mol"].shape[1], dtype=np.float32),
            cp_mean=np.zeros(kwargs["X_src_cp"].shape[1], dtype=np.float32),
            cp_std=np.ones(kwargs["X_src_cp"].shape[1], dtype=np.float32),
            ridge_lambdas=[9.0],
            ridge_lambda_b=9.0,
        )

    def _fake_train(**kwargs: Any) -> EnsembleOutput:
        train_calls.append(kwargs)
        return _ensemble_output()

    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "build_adaptive_source_weights", return_value=np.full(prep.X_src.shape[0], 0.4, dtype=np.float32)) as weight_mock,
        mock.patch.object(runner, "pretrain_hyper_tl", side_effect=_fake_pretrain_hyper_tl),
        mock.patch.object(runner, "train_and_ensemble", side_effect=_fake_train),
        mock.patch.object(runner, "compute_metrics", return_value=_metrics(mae=10.0, r2=0.1)),
        mock.patch.object(
            runner,
            "summarize_vs_paper",
            return_value=_summary_df(win_both=False).assign(dataset=["0002"]),
        ),
        mock.patch.object(runner, "write_report"),
    ):
        result = runner.run_trial(prep, cfg, seeds=[0], config_sha1="sha-hyper", write_predictions=False, early_stop=True)

    assert result.early_stop_reason.startswith("early_stop: avg_r2 cannot beat paper")
    assert len(pretrain_calls) == 2
    assert len(train_calls) == 1
    assert weight_mock.call_count == 1
    expected_balance = np.array([0.8333333, 0.8333333, 0.8333333, 1.25, 1.25], dtype=np.float32)
    assert np.allclose(pretrain_calls[0]["sample_weights"], expected_balance, atol=1e-6)
    assert pretrain_calls[0]["X_src_mol"].shape[1] == 168
    assert train_calls[0]["X_src_mol"].shape[1] == 168
    assert train_calls[0]["X_target_mol"].shape[1] == 168
    assert np.allclose(train_calls[0]["source_sample_weights"], np.zeros(prep.X_src.shape[0], dtype=np.float32))
    expected_y_src = np.log(np.clip((prep.y_src_sec - 10.0) / 10.0, 1e-6, None)).astype(np.float32)
    expected_y_tgt = np.log(np.clip((prep.mats["0002"].y_sec - 10.0) / 10.0, 1e-6, None)).astype(np.float32)
    assert np.allclose(train_calls[0]["y_src"], expected_y_src)
    assert np.allclose(train_calls[0]["y_target"], expected_y_tgt)
    assert len(train_calls[0]["hyper_bundle"]) == 2
    assert train_calls[0]["hyper_bundle"][0].ridge_lambdas == [0.5, 1.5]
    assert train_calls[0]["hyper_bundle"][0].ridge_lambda_b == 0.25

    cfg.outputs["root"] = str(tmp_path / "hyper_second")
    with (
        mock.patch.object(runner, "random_split", return_value=_split()),
        mock.patch.object(runner, "build_adaptive_source_weights", return_value=np.full(prep.X_src.shape[0], 0.4, dtype=np.float32)),
        mock.patch.object(runner, "pretrain_hyper_tl") as pretrain_mock,
        mock.patch.object(runner, "train_and_ensemble", return_value=_ensemble_output()),
        mock.patch.object(runner, "compute_metrics", return_value=_metrics(mae=1.0, r2=0.9)),
        mock.patch.object(runner, "summarize_vs_paper", return_value=_summary_df()),
        mock.patch.object(runner, "write_report"),
    ):
        runner.run_trial(prep, cfg, seeds=[0], external_ids=["0002"], config_sha1="sha-hyper-cache", write_predictions=False)

    pretrain_mock.assert_not_called()
