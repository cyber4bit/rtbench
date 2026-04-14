from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import rtbench.run as run_entry


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_rplc_14x14_smoke_runs_end_to_end_and_registers_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "phase9_project"
    project_root.mkdir()
    outputs_root = project_root / "outputs_smoke"
    config_path = REPO_ROOT / "configs" / "rplc_14x14_smoke.yaml"
    data_root = REPO_ROOT / "data" / "repoRT"
    baseline_csv = REPO_ROOT / "data" / "baseline" / "paper_table2_3_external_rplc.csv"

    argv = [
        "rtbench.run",
        "--config",
        str(config_path),
        "--seeds",
        "0:0",
        "--override",
        f"data.local_root={data_root.as_posix()}",
        f"data.baseline_csv={baseline_csv.as_posix()}",
        f"outputs.root={outputs_root.as_posix()}",
        "--no-download",
        "--log-level",
        "INFO",
    ]

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(sys, "argv", argv)

    run_entry.main()

    per_seed_path = outputs_root / "metrics" / "per_seed.csv"
    summary_path = outputs_root / "metrics" / "summary_vs_paper.csv"
    report_path = outputs_root / "report.md"
    log_path = outputs_root / "logs" / "run.jsonl"
    resolved_config_path = outputs_root / "config.resolved.yaml"
    registry_path = project_root / "experiments" / "registry.csv"

    for path in (per_seed_path, summary_path, report_path, log_path, resolved_config_path, registry_path):
        assert path.exists(), f"missing artifact: {path}"

    per_seed = pd.read_csv(per_seed_path, dtype={"dataset": str}, encoding="utf-8")
    summary = pd.read_csv(summary_path, dtype={"dataset": str}, encoding="utf-8")
    registry = pd.read_csv(registry_path, dtype=str, encoding="utf-8").fillna("")

    per_seed["dataset"] = per_seed["dataset"].astype(str).str.zfill(4)
    summary["dataset"] = summary["dataset"].astype(str).str.zfill(4)

    assert set(per_seed["dataset"]) == {"0050", "0233"}
    assert set(summary["dataset"]) == {"0050", "0233"}
    assert set(per_seed["seed"].astype(int)) == {0}
    assert "win_both" in summary.columns
    assert (outputs_root / "predictions" / "0050" / "seed_0.csv").exists()
    assert (outputs_root / "predictions" / "0233" / "seed_0.csv").exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "# RepoRT 14+14 Benchmark Report" in report_text
    assert "## Final Verdict" in report_text

    log_lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert log_lines
    assert any(line.get("message") == "[3/3] Run completed." for line in log_lines)

    row = registry.loc[registry["run_dir"] == "outputs_smoke"].reset_index(drop=True)
    assert len(row) == 1
    assert row.loc[0, "status"] == "success"
    assert row.loc[0, "summary_path"] == "outputs_smoke/metrics/summary_vs_paper.csv"
    assert row.loc[0, "dataset_count"] == "2"
    assert row.loc[0, "seed_count"] == "1"
    assert row.loc[0, "effective_config_path"] == "outputs_smoke/config.resolved.yaml"
    assert row.loc[0, "effective_config_sha1"]
    assert row.loc[0, "config_path"].replace("\\", "/").lower().endswith("/configs/rplc_14x14_smoke.yaml")
