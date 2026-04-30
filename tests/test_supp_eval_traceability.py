from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from rtbench.experimental import supp_eval


def _base_raw(tmp_path: Path) -> dict[str, object]:
    return {
        "data": {
            "local_root": (tmp_path / "data" / "repoRT").as_posix(),
            "repo_url": "https://example.com/repoRT.git",
            "commit": "abc123",
            "baseline_csv": "",
            "cpvec": {"enabled": False},
        },
        "datasets": {
            "pretrain": ["0002"],
            "external": ["0001"],
            "expected_pretrain_count": 0,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
        "models": {"FUSION_TOP_K": 3, "CALIBRATE": True},
        "transfer_weights": {"source": 0.2, "target": 1.0, "adaptive_source": True},
        "seeds": {"default": "0:1"},
        "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
        "stats": {"fdr_q": 0.05},
        "outputs": {"root": "outputs_placeholder", "resume": False},
    }


def test_supp_eval_persists_traceable_subrun_configs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sheet_df = pd.DataFrame(
        [
            {"dataset": "0001", "MDL_TL_mae": 1.10, "MDL_TL_r2": 0.61, "Uni_RT_mae": 1.20, "Uni_RT_r2": 0.58},
            {"dataset": "0002", "MDL_TL_mae": 1.20, "MDL_TL_r2": 0.62, "Uni_RT_mae": 1.30, "Uni_RT_r2": 0.57},
        ]
    )
    counts = {"0001": 11, "0002": 13}
    expected_baseline = "data/baseline/supp_s4_mdl_tl.csv"
    out_root = Path("outputs_supp_eval_trace")
    seeds = [0, 1]

    monkeypatch.setattr(
        supp_eval,
        "parse_supp_table",
        lambda xlsx_path, sheet_name: (sheet_df.copy(), ["MDL_TL", "Uni_RT"]),
    )
    monkeypatch.setattr(supp_eval, "_rt_count_for_dataset", lambda processed_root, ds: counts[str(ds)])
    monkeypatch.setattr(supp_eval, "prepare", lambda cfg, no_download: SimpleNamespace(cfg=cfg))

    def _fake_run_trial(
        prep,
        cfg,
        *,
        seeds,
        config_sha1,
        resume_enabled,
        write_predictions,
        early_stop,
    ):
        del prep, resume_enabled, write_predictions, early_stop
        run_dir = Path(cfg.outputs["root"])
        resolved_path = run_dir / "config.resolved.yaml"
        assert resolved_path.exists()
        assert (run_dir / "config.sha1").read_text(encoding="utf-8").strip() == hashlib.sha1(
            resolved_path.read_bytes()
        ).hexdigest()
        assert config_sha1 == hashlib.sha1(resolved_path.read_bytes()).hexdigest()

        resolved_raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        dataset = str(cfg.datasets["external"][0]).zfill(4)
        assert resolved_raw["data"]["baseline_csv"] == cfg.data["baseline_csv"] == expected_baseline
        assert resolved_raw["datasets"]["external"] == [dataset]
        assert resolved_raw["outputs"]["root"] == run_dir.as_posix()

        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        per_seed_df = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "mae": 1.0 + int(dataset) / 1000.0 + int(seed) * 0.01,
                    "medae": 1.0,
                    "mre": 0.1,
                    "medre": 0.1,
                    "r2": 0.7,
                    "rmse": 1.2,
                }
                for seed in seeds
            ]
        )
        per_seed_df.to_csv(metrics_dir / "per_seed.csv", index=False, encoding="utf-8")

        summary_df = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "our_mae_mean": float(per_seed_df["mae"].mean()),
                    "our_r2_mean": 0.70 + int(dataset) / 10000.0,
                    "delta_mae": 0.15,
                    "delta_r2": 0.06,
                    "p_mae": 0.01,
                    "p_r2": 0.02,
                    "win_both": True,
                }
            ]
        )
        summary_df.to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
        return SimpleNamespace(out_root=run_dir, summary_df=summary_df)

    monkeypatch.setattr(supp_eval, "run_trial", _fake_run_trial)

    supp_eval.run_sheet(
        base_raw=_base_raw(tmp_path),
        xlsx_path=tmp_path / "supp.xlsx",
        sheet_name="S4",
        seeds=seeds,
        out_root=out_root,
        no_download=True,
    )

    baseline_path = tmp_path / expected_baseline
    assert baseline_path.exists()

    registry = pd.read_csv(tmp_path / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
    success_rows = registry.loc[registry["status"] == "success"].sort_values("run_dir").reset_index(drop=True)
    assert success_rows["run_dir"].tolist() == [
        "outputs_supp_eval_trace/S4/0001",
        "outputs_supp_eval_trace/S4/0002",
    ]

    for row in success_rows.to_dict(orient="records"):
        run_dir = tmp_path / row["run_dir"]
        resolved_path = run_dir / "config.resolved.yaml"
        assert resolved_path.exists()
        resolved_raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        assert resolved_raw["data"]["baseline_csv"] == expected_baseline
        assert resolved_raw["datasets"]["external"] == [run_dir.name]
        assert resolved_raw["datasets"]["pretrain"] == sorted(
            ds for ds in sheet_df["dataset"].astype(str).str.zfill(4).tolist() if ds != run_dir.name
        )
        assert resolved_raw["outputs"]["root"] == row["run_dir"]
        assert row["effective_config_path"] == f"{row['run_dir']}/config.resolved.yaml"
        assert row["effective_config_sha1"] == hashlib.sha1(resolved_path.read_bytes()).hexdigest()
        assert row["config_sha1"] == row["effective_config_sha1"]
        assert row["config_hash_type"] == "file_sha1"

        registry_resolved_path = tmp_path / row["effective_config_path"]
        assert registry_resolved_path == resolved_path
        replay_cfg = yaml.safe_load(registry_resolved_path.read_text(encoding="utf-8"))
        per_seed_path = tmp_path / replay_cfg["outputs"]["root"] / "metrics" / "per_seed.csv"
        per_seed_df = pd.read_csv(per_seed_path, dtype={"dataset": str}, encoding="utf-8").fillna("")
        assert per_seed_df["dataset"].astype(str).str.zfill(4).unique().tolist() == replay_cfg["datasets"]["external"]
        assert per_seed_df["seed"].astype(int).tolist() == seeds
