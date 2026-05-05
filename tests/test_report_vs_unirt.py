from __future__ import annotations

from pathlib import Path

import pandas as pd

from rtbench.report_vs_unirt import write_unirt_report


def test_unirt_report_includes_assessment_diagnostics_and_figures(tmp_path: Path) -> None:
    per_seed = pd.DataFrame(
        [
            {
                "dataset": ds,
                "seed": seed,
                "mae": mae + seed * 0.1,
                "medae": mae / 2.0,
                "mre": 0.1,
                "r2": r2,
            }
            for ds, mae, r2 in (("0001", 10.0, 0.90), ("0002", 30.0, 0.70))
            for seed in range(10)
        ]
    )
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {"mode": "RPLC", "dataset": "0001", "method": "Uni-RT", "n_model": 1, "mae": 12.0, "medae": 6.0, "mre": 0.12, "r2": 0.88},
            {"mode": "RPLC", "dataset": "0002", "method": "Uni-RT", "n_model": 1, "mae": 20.0, "medae": 10.0, "mre": 0.15, "r2": 0.80},
        ]
    ).to_csv(baseline, index=False, encoding="utf-8")

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    diagnostics = metrics_dir / "diagnostics_vs_unirt.csv"
    pd.DataFrame(
        [
            {
                "Dataset": "0001",
                "status": "win_both",
                "win_both": True,
                "n_rows": 120,
                "source_overlap_rate": 0.5,
                "fingerprint_diversity": 0.7,
                "nearest_cp_zdist": 1.0,
                "outlier_rate": 0.01,
                "small_dataset_lt100": False,
                "low_source_overlap": False,
                "high_structure_diversity": False,
                "chromatographic_mismatch": False,
                "high_outlier_rate": False,
            },
            {
                "Dataset": "0002",
                "status": "loss_both",
                "win_both": False,
                "n_rows": 80,
                "source_overlap_rate": 0.02,
                "fingerprint_diversity": 0.9,
                "nearest_cp_zdist": 5.0,
                "outlier_rate": 0.2,
                "small_dataset_lt100": True,
                "low_source_overlap": True,
                "high_structure_diversity": True,
                "chromatographic_mismatch": True,
                "high_outlier_rate": True,
            },
        ]
    ).to_csv(diagnostics, index=False, encoding="utf-8")

    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    for name in ("predicted_vs_true_rt.png", "mre_by_dataset.png", "summary_boxplots.png"):
        (fig_dir / name).write_bytes(b"png")

    out = tmp_path / "report_vs_unirt.md"
    write_unirt_report(
        out_path=out,
        per_seed=per_seed,
        baseline_csv=baseline,
        mode="RPLC",
        n_model=2,
        expected_seeds=list(range(10)),
        split_cfg={"train": 0.8, "val": 0.1, "test": 0.1},
        output_dir=metrics_dir,
    )

    text = out.read_text(encoding="utf-8")
    assert "Ours (avg +/- Std)" in text
    assert "## Overall Assessment" in text
    assert "## Dataset-Level Diagnosis" in text
    assert "low source overlap" in text
    assert "## Visualizations" in text
    assert "predicted_vs_true_rt.png" in text
    assert "uses nModel=2" in text


def test_unirt_report_uses_matching_fairness_note_for_single_unified_model(tmp_path: Path) -> None:
    per_seed = pd.DataFrame(
        [
            {"dataset": "0001", "seed": seed, "mae": 10.0, "medae": 5.0, "mre": 0.1, "r2": 0.9}
            for seed in range(10)
        ]
    )
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        [
            {
                "mode": "RPLC",
                "dataset": "0001",
                "method": "Uni-RT",
                "n_model": 1,
                "mae": 12.0,
                "medae": 6.0,
                "mre": 0.12,
                "r2": 0.88,
            }
        ]
    ).to_csv(baseline, index=False, encoding="utf-8")

    out = tmp_path / "report_vs_unirt.md"
    write_unirt_report(
        out_path=out,
        per_seed=per_seed,
        baseline_csv=baseline,
        mode="RPLC",
        n_model=1,
        expected_seeds=list(range(10)),
    )

    text = out.read_text(encoding="utf-8")
    assert "reports nModel=1" in text
    assert "uses nModel=14" not in text
