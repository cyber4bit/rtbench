from __future__ import annotations

from benchmarks._support import benchmark_callable, small_cpvec_cfg, temp_repo
from rtbench import cpvec


def run_benchmark(*, cold_repeats: int = 2, warm_repeats: int = 3) -> dict[str, float | int]:
    dataset_ids = ["0001", "0002", "0003", "0004"]
    cfg = small_cpvec_cfg()

    def _cold_run() -> int:
        tmpdir, data_root, processed_root = temp_repo(dataset_ids=dataset_ids, rows_per_dataset=10)
        try:
            encoder, cp_dim = cpvec.load_or_train_cpvec(
                data_root=data_root,
                processed_root=processed_root,
                repo_url="https://example.com/repo",
                commit="deadbeefcafebabe",
                cfg=cfg,
                download=False,
                dataset_ids=dataset_ids,
            )
            _ = encoder.cp_vector_for_dataset(processed_root / "0001", "0001")
            return int(cp_dim)
        finally:
            tmpdir.cleanup()

    tmpdir, data_root, processed_root = temp_repo(dataset_ids=dataset_ids, rows_per_dataset=10)
    try:
        encoder, cp_dim = cpvec.load_or_train_cpvec(
            data_root=data_root,
            processed_root=processed_root,
            repo_url="https://example.com/repo",
            commit="deadbeefcafebabe",
            cfg=cfg,
            download=False,
            dataset_ids=dataset_ids,
        )
        _ = encoder.cp_vector_for_dataset(processed_root / "0001", "0001")

        def _warm_run() -> int:
            enc, dim = cpvec.load_or_train_cpvec(
                data_root=data_root,
                processed_root=processed_root,
                repo_url="https://example.com/repo",
                commit="deadbeefcafebabe",
                cfg=cfg,
                download=False,
                dataset_ids=dataset_ids,
            )
            _ = enc.cp_vector_for_dataset(processed_root / "0002", "0002")
            return int(dim)

        cold_metrics = benchmark_callable(_cold_run, repeats=cold_repeats, warmups=0)
        warm_metrics = benchmark_callable(_warm_run, repeats=warm_repeats, warmups=1)
    finally:
        tmpdir.cleanup()

    return {
        "cp_dim": int(cp_dim),
        "study_count": int(len(dataset_ids)),
        "cold_mean_ms": float(cold_metrics["mean_ms"]),
        "cold_min_ms": float(cold_metrics["min_ms"]),
        "cold_peak_python_bytes": int(cold_metrics["peak_python_bytes"]),
        "warm_mean_ms": float(warm_metrics["mean_ms"]),
        "warm_min_ms": float(warm_metrics["min_ms"]),
        "warm_peak_python_bytes": int(warm_metrics["peak_python_bytes"]),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
