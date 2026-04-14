from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks._support import benchmark_callable, temp_repo
from rtbench.bench.prepare import prepare
from rtbench.config import Config


def run_benchmark(*, repeats: int = 3) -> dict[str, float | int]:
    dataset_ids = ["0001", "0002", "0003"]
    tmpdir, data_root, processed_root = temp_repo(dataset_ids=dataset_ids, rows_per_dataset=12)
    try:
        baseline_csv = Path(tmpdir.name) / "baseline.csv"
        pd.DataFrame([{"dataset": "0003", "paper_mae": 1.0, "paper_r2": 0.5}]).to_csv(
            baseline_csv,
            index=False,
            encoding="utf-8",
        )
        cfg = Config(
            data={
                "repo_url": "https://example.com/repo",
                "commit": "deadbeefcafebabe",
                "local_root": str(data_root),
                "baseline_csv": str(baseline_csv),
                "gradient_points": 20,
            },
            datasets={"pretrain": ["0001", "0002"], "external": ["0003"], "expected_pretrain_count": 24},
            split={"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
            models={},
            transfer_weights={"source": 0.2, "target": 1.0},
            seeds={"default": "0:1"},
            metrics={"paper_avg_mae": 1.0, "paper_avg_r2": 0.5, "required_win_both": 0},
            stats={"fdr_q": 0.05},
            outputs={"root": str(Path(tmpdir.name) / "outputs_bench"), "resume": False},
        )

        prepared = prepare(cfg, no_download=True)

        def _run() -> int:
            bench = prepare(cfg, no_download=True)
            return int(bench.X_src.shape[0] + sum(len(bench.mats[ds].ids) for ds in bench.external_ids))

        metrics = benchmark_callable(_run, repeats=repeats, warmups=1)
    finally:
        tmpdir.cleanup()

    metrics["dataset_count"] = int(len(dataset_ids))
    metrics["source_rows"] = int(prepared.X_src.shape[0])
    metrics["external_rows"] = int(sum(len(prepared.mats[ds].ids) for ds in prepared.external_ids))
    metrics["feature_width"] = int(prepared.X_src.shape[1])
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
