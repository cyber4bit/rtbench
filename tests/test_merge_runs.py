from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import rtbench.merge_runs as merge_runs


def _write_run(root: Path, rows: list[dict[str, object]], prediction_text: str, *, dataset: str) -> None:
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "predictions" / dataset).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "metrics" / "per_seed.csv", index=False, encoding="utf-8")
    (root / "predictions" / dataset / "seed_0.csv").write_text(prediction_text, encoding="utf-8")


def test_parse_override_requires_dataset_assignment() -> None:
    with pytest.raises(ValueError, match="Invalid override"):
        merge_runs.parse_override("0001")


def test_merge_runs_combines_outputs_and_recomputes_summary(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    override_root = tmp_path / "override"
    out_root = tmp_path / "merged"
    baseline_csv = tmp_path / "baseline.csv"

    _write_run(
        base_root,
        rows=[
            {"dataset": "0001", "seed": 0, "mae": 10.0, "medae": 10.0, "mre": 0.1, "medre": 0.1, "r2": 0.50, "rmse": 11.0},
            {"dataset": "0002", "seed": 0, "mae": 20.0, "medae": 20.0, "mre": 0.2, "medre": 0.2, "r2": 0.30, "rmse": 21.0},
        ],
        prediction_text="base\n",
        dataset="0001",
    )
    (base_root / "predictions" / "0002").mkdir(parents=True, exist_ok=True)
    (base_root / "predictions" / "0002" / "seed_0.csv").write_text("base-override\n", encoding="utf-8")
    _write_run(
        override_root,
        rows=[
            {"dataset": "0002", "seed": 0, "mae": 5.0, "medae": 5.0, "mre": 0.05, "medre": 0.05, "r2": 0.90, "rmse": 6.0},
        ],
        prediction_text="override\n",
        dataset="0002",
    )
    pd.DataFrame(
        [
            {"dataset": "0001", "paper_mae": 12.0, "paper_r2": 0.45},
            {"dataset": "0002", "paper_mae": 12.0, "paper_r2": 0.45},
        ]
    ).to_csv(baseline_csv, index=False, encoding="utf-8")
    summary_df = pd.DataFrame(
        [
            {
                "dataset": "0001",
                "paper_mae": 12.0,
                "paper_r2": 0.45,
                "our_mae_mean": 10.0,
                "our_r2_mean": 0.50,
                "delta_mae": 2.0,
                "delta_r2": 0.05,
                "p_mae": 0.1,
                "p_r2": 0.1,
                "p_adj_mae": 0.1,
                "p_adj_r2": 0.1,
                "win_both": True,
            },
            {
                "dataset": "0002",
                "paper_mae": 12.0,
                "paper_r2": 0.45,
                "our_mae_mean": 5.0,
                "our_r2_mean": 0.90,
                "delta_mae": 7.0,
                "delta_r2": 0.45,
                "p_mae": 0.1,
                "p_r2": 0.1,
                "p_adj_mae": 0.1,
                "p_adj_r2": 0.1,
                "win_both": True,
            },
        ]
    )

    argv = [
        "rtbench.merge_runs",
        "--base",
        str(base_root),
        "--override",
        f"0002={override_root}",
        "--out",
        str(out_root),
        "--baseline-csv",
        str(baseline_csv),
        "--paper-avg-mae",
        "12.0",
        "--paper-avg-r2",
        "0.45",
        "--required-win-both",
        "1",
        "--fdr-q",
        "0.05",
        "--log-level",
        "WARNING",
    ]

    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(merge_runs, "summarize_vs_paper", return_value=summary_df) as summarize_mock,
        mock.patch.object(merge_runs, "write_report") as report_mock,
        mock.patch.object(merge_runs, "record_experiment") as record_mock,
    ):
        merge_runs.main()

    merged = pd.read_csv(out_root / "metrics" / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    merged_0002 = merged.loc[merged["dataset"] == "0002"].reset_index(drop=True)
    assert merged_0002.loc[0, "mae"] == 5.0
    assert (out_root / "predictions" / "0001" / "seed_0.csv").read_text(encoding="utf-8") == "base\n"
    assert (out_root / "predictions" / "0002" / "seed_0.csv").read_text(encoding="utf-8") == "override\n"
    assert summarize_mock.call_count == 1
    recomputed = summarize_mock.call_args.kwargs["per_seed_df"]
    assert recomputed.loc[recomputed["dataset"] == "0002", "mae"].tolist() == [5.0]
    assert report_mock.call_count == 1
    assert record_mock.call_count == 1
    assert (out_root / "metrics" / "summary_vs_paper.csv").exists()
