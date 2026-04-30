from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.bench_candidate_builder import run_benchmark as run_candidate_builder_benchmark
from benchmarks.bench_cpvec import run_benchmark as run_cpvec_benchmark
from benchmarks.bench_data_loading import run_benchmark as run_data_loading_benchmark
from benchmarks._support import benchmark_header, write_json


DEFAULT_OUTPUT = Path("benchmarks/baseline.json")


def run_all_benchmarks() -> dict[str, Any]:
    return {
        **benchmark_header(),
        "benchmarks": {
            "candidate_builder": run_candidate_builder_benchmark(),
            "cpvec": run_cpvec_benchmark(),
            "data_loading": run_data_loading_benchmark(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTBench synthetic performance baselines")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to write the benchmark baseline JSON")
    args = parser.parse_args()

    payload = run_all_benchmarks()
    out_path = Path(args.output)
    write_json(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
