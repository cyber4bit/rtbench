from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from rtbench.bench import config_from_raw
from rtbench.experimental import supp_combo, sweep


def _baseline_columns(mae_base: float, r2_base: float) -> dict[str, float]:
    return {
        "Uni_RT_mae": mae_base + 0.3,
        "Uni_RT_r2": r2_base - 0.1,
        "MDL_TL_mae": mae_base,
        "MDL_TL_r2": r2_base,
        "GNN_RT_mae": mae_base + 0.2,
        "GNN_RT_r2": r2_base - 0.05,
        "DeepGCN_RT_mae": mae_base + 0.1,
        "DeepGCN_RT_r2": r2_base - 0.02,
    }


def test_supp_combo_main_writes_combined_outputs(tmp_path: Path, monkeypatch) -> None:
    run_a = tmp_path / "run_a.csv"
    run_b = tmp_path / "run_b.csv"
    out_dir = tmp_path / "combo_out"
    policy_path = tmp_path / "policy.yaml"

    pd.DataFrame(
        [
            {"dataset": "0001", "our_mae_mean": 1.00, "our_r2_mean": 0.82, "p_mae": 0.02, "p_r2": 0.03, **_baseline_columns(1.10, 0.75)},
            {"dataset": "0002", "our_mae_mean": 1.20, "our_r2_mean": 0.80, "p_mae": 0.04, "p_r2": 0.05, **_baseline_columns(1.25, 0.74)},
        ]
    ).to_csv(run_a, index=False, encoding="utf-8")
    pd.DataFrame(
        [
            {"dataset": "0001", "our_mae_mean": 0.95, "our_r2_mean": 0.84, "p_mae": 0.01, "p_r2": 0.02, **_baseline_columns(1.10, 0.75)},
            {"dataset": "0002", "our_mae_mean": 1.10, "our_r2_mean": 0.83, "p_mae": 0.03, "p_r2": 0.04, **_baseline_columns(1.25, 0.74)},
        ]
    ).to_csv(run_b, index=False, encoding="utf-8")
    policy_path.write_text(
        yaml.safe_dump(
            {
                "runs": {"run_a": run_a.as_posix(), "run_b": run_b.as_posix()},
                "policy": {"0001": "run_b", "0002": "run_a"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["supp_combo", "--policy", str(policy_path), "--out-dir", str(out_dir), "--log-level", "WARNING"],
    )
    supp_combo.main()

    comparison = pd.read_csv(out_dir / "comparison.csv", dtype={"dataset": str}, encoding="utf-8").fillna("")
    summary_text = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert comparison["dataset"].tolist() == ["0001", "0002"]
    assert comparison["picked_run"].tolist() == ["run_b", "run_a"]
    assert "p_adj_mae" in comparison.columns
    assert "better_both_vs_MDL_TL" in comparison.columns
    assert "# Supplement Combo Summary" in summary_text
    assert "- 0001: run_b" in summary_text
    assert "- 0002: run_a" in summary_text


def test_sweep_main_writes_summary_and_snapshots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "base.yaml"
    out_dir = tmp_path / "outputs_sweep_test"
    base_raw = {
        "data": {
            "repo_url": "https://example.com/repo",
            "commit": "abc123",
            "local_root": "data/repoRT",
            "baseline_csv": "data/baseline/demo.csv",
            "gradient_points": 20,
        },
        "datasets": {
            "pretrain": ["0019", "0052"],
            "external": ["0050", "0233"],
            "expected_pretrain_count": 2,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
        "models": {
            "FUSION_TOP_K": 1,
            "CALIBRATE": False,
            "ONLY_HYPER_TL": False,
            "XGB_A": {"n_estimators": 10},
            "XGB_B": {"n_estimators": 10},
            "LGBM_A": {"n_estimators": 10},
            "LGBM_B": {"n_estimators": 10},
            "HYPER_TL": {},
        },
        "transfer_weights": {"source": 0.2, "target": 1.0, "adaptive_source": False, "target_transform": "none"},
        "seeds": {"default": "0:0"},
        "metrics": {"paper_avg_mae": 5.0, "paper_avg_r2": 0.5, "required_win_both": 1},
        "stats": {"test": "wilcoxon", "correction": "bh_fdr", "fdr_q": 0.05},
        "outputs": {"root": "outputs_demo", "resume": False},
    }
    config_path.write_text(yaml.safe_dump(base_raw, sort_keys=False), encoding="utf-8")
    resolved = SimpleNamespace(config=config_from_raw(base_raw), raw=copy.deepcopy(base_raw))

    monkeypatch.setattr(sweep, "resolve_config", lambda path: resolved)
    monkeypatch.setattr(sweep, "prepare", lambda cfg, external_ids, no_download: SimpleNamespace(cfg=cfg, external_ids=external_ids))

    counter = {"n": 0}

    def _fake_run_trial(
        prep,
        cfg,
        *,
        seeds,
        external_ids,
        config_sha1,
        resume_enabled,
        write_predictions,
        early_stop,
    ):
        del prep, config_sha1, resume_enabled, write_predictions, early_stop
        counter["n"] += 1
        run_dir = Path(cfg.outputs["root"])
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        avg_mae = 10.0 + counter["n"] / 100.0
        avg_r2 = 0.7 + counter["n"] / 1000.0
        summary_df = pd.DataFrame(
            [
                {
                    "dataset": ds,
                    "our_mae_mean": avg_mae,
                    "our_r2_mean": avg_r2,
                    "delta_mae": avg_mae - 5.0,
                    "delta_r2": avg_r2 - 0.5,
                    "win_both": True,
                }
                for ds in external_ids
            ]
        )
        summary_df.to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
        pd.DataFrame([{"dataset": external_ids[0], "seed": int(seeds[0]), "mae": avg_mae, "r2": avg_r2}]).to_csv(
            metrics_dir / "per_seed.csv",
            index=False,
            encoding="utf-8",
        )
        return SimpleNamespace(
            out_root=run_dir,
            summary_df=summary_df,
            avg_mae=avg_mae,
            avg_r2=avg_r2,
            wins=1,
            success=True,
            early_stop_reason="",
        )

    monkeypatch.setattr(sweep, "run_trial", _fake_run_trial)

    recorded: list[dict[str, object]] = []

    def _record_experiment(project_root, **kwargs):
        del project_root
        recorded.append(kwargs)

    monkeypatch.setattr(sweep, "record_experiment", _record_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--config",
            str(config_path),
            "--seeds",
            "0:0",
            "--out-dir",
            str(out_dir),
            "--no-download",
            "--log-level",
            "WARNING",
        ],
    )

    sweep.main()

    summary = pd.read_csv(out_dir / "sweep_summary.csv", encoding="utf-8")
    assert len(summary) == 36
    assert counter["n"] == 36
    assert len(recorded) == 36
    assert summary.iloc[0]["success"]
    first_trial = summary.iloc[0]["trial"]
    assert (out_dir / first_trial / "config.resolved.yaml").exists()
    assert (out_dir / first_trial / "metrics" / "summary_vs_paper.csv").exists()
