from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_cli_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "v7/scripts/run_unified_cv.py"
    spec = importlib.util.spec_from_file_location("run_unified_cv_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_cv_parser_accepts_required_shape() -> None:
    cli = _load_cli_module()
    args = cli.build_parser().parse_args(
        [
            "--sheet",
            "S4",
            "--mode",
            "RPLC",
            "--config",
            "v7/configs/v7_unified.yaml",
            "--folds",
            "10",
            "--shuffle-seed",
            "20260505",
            "--output-root",
            "outputs_v7_unified_cv10_S4",
            "--no-download",
        ]
    )

    assert args.sheet == "S4"
    assert args.mode == "RPLC"
    assert args.config == "v7/configs/v7_unified.yaml"
    assert args.folds == 10
    assert args.shuffle_seed == 20260505
    assert args.no_download is True


def test_unified_cv_cli_calls_core_and_writes_unirt_outputs(tmp_path: Path, monkeypatch) -> None:
    cli = _load_cli_module()
    baseline = tmp_path / "unirt_sota_28.csv"
    pd.DataFrame(
        [
            {
                "mode": "RPLC",
                "dataset": "0001",
                "method": "Uni-RT",
                "n_model": 1,
                "mae": 11.0,
                "medae": 5.5,
                "mre": 0.11,
                "r2": 0.80,
            }
        ]
    ).to_csv(baseline, index=False, encoding="utf-8")
    config_path = tmp_path / "v7" / "configs" / "v7_unified.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "data: {}",
                "datasets: {}",
                "split: {}",
                "models: {}",
                "transfer_weights: {}",
                "seeds: {}",
                "metrics: {}",
                "stats: {}",
                "outputs: {}",
            ]
        ),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    def fake_run_unified_cv(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "dataset": "0001",
                    "seed": fold,
                    "mae": 10.0 + fold * 0.01,
                    "medae": 5.0,
                    "mre": 0.10,
                    "r2": 0.90,
                }
                for fold in range(10)
            ]
        )

    fake_core = types.ModuleType("rtbench.bench.unified_cv")
    fake_core.run_unified_cv = fake_run_unified_cv
    monkeypatch.setitem(sys.modules, "rtbench.bench.unified_cv", fake_core)

    out_root = tmp_path / "outputs_v7_unified_cv10_S4"
    args = cli.build_parser().parse_args(
        [
            "--sheet",
            "S4",
            "--mode",
            "RPLC",
            "--config",
            "v7/configs/v7_unified.yaml",
            "--folds",
            "10",
            "--shuffle-seed",
            "20260505",
            "--output-root",
            str(out_root),
            "--baseline",
            str(baseline),
            "--repo-root",
            str(tmp_path),
            "--no-download",
        ]
    )

    paths = cli.run(args)

    assert seen["sheet"] == "S4"
    assert seen["mode"] == "RPLC"
    assert seen["folds"] == 10
    assert seen["shuffle_seed"] == 20260505
    assert seen["no_download"] is True
    assert paths["per_seed"].exists()
    assert paths["report"].exists()
    assert paths["cross_summary"].exists()
    assert paths["dataset_comparison"].exists()

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "nModel=1" in report_text
    assert "observed seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], expected=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" in report_text

    summary = pd.read_csv(paths["cross_summary"], encoding="utf-8")
    nmodel = summary.loc[summary["Metrics"] == "nModel"].iloc[0]
    assert str(nmodel["RTBench strict unified CV (avg +/- Std)"]) == "1"
