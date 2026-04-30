from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from rtbench.data import FINGERPRINT_SIZES


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def benchmark_callable(
    func: Callable[[], Any],
    *,
    repeats: int = 3,
    warmups: int = 1,
) -> dict[str, float]:
    for _ in range(max(0, int(warmups))):
        func()

    times_ms: list[float] = []
    peaks: list[int] = []
    for _ in range(max(1, int(repeats))):
        tracemalloc.start()
        started = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times_ms.append(float(elapsed_ms))
        peaks.append(int(peak))

    return {
        "repeats": int(max(1, int(repeats))),
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "peak_python_bytes": int(max(peaks) if peaks else 0),
    }


def benchmark_header() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def create_synthetic_report_repo(
    root: Path,
    *,
    dataset_ids: list[str],
    rows_per_dataset: int = 12,
) -> tuple[Path, Path]:
    data_root = root / "repoRT"
    processed_root = data_root / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": dataset_ids}).to_csv(processed_root / "studies.tsv", sep="\t", index=False, encoding="utf-8")
    for offset, dataset_id in enumerate(dataset_ids, start=1):
        _write_dataset(processed_root, dataset_id, rows=rows_per_dataset, offset=offset)
    return data_root, processed_root


def small_cpvec_cfg() -> dict[str, float | int]:
    return {
        "seed": 0,
        "col_w2v_dim": 5,
        "col_w2v_window": 2,
        "col_w2v_neg": 2,
        "col_w2v_epochs": 10,
        "col_w2v_lr": 0.02,
        "col_w2v_batch_size": 16,
        "col_w2v_min_count": 1,
        "ae1_latent": 4,
        "ae1_hidden": 16,
        "ae1_dropout": 0.0,
        "ae1_epochs": 40,
        "ae1_lr": 0.02,
        "ae1_batch_size": 8,
        "ae2_latent": 3,
        "ae2_hidden": 12,
        "ae2_dropout": 0.0,
        "ae2_epochs": 40,
        "ae2_lr": 0.02,
        "ae2_batch_size": 4,
    }


def temp_repo(
    *,
    dataset_ids: list[str],
    rows_per_dataset: int = 12,
) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    tmpdir = tempfile.TemporaryDirectory()
    data_root, processed_root = create_synthetic_report_repo(
        Path(tmpdir.name),
        dataset_ids=dataset_ids,
        rows_per_dataset=rows_per_dataset,
    )
    return tmpdir, data_root, processed_root


def _write_dataset(processed_root: Path, dataset_id: str, *, rows: int, offset: int) -> None:
    ds_root = processed_root / dataset_id
    ds_root.mkdir(parents=True, exist_ok=True)

    ids = [f"{dataset_id}_{i:03d}" for i in range(rows)]
    rt_df = pd.DataFrame(
        {
            "id": ids,
            "rt": np.linspace(1.0 + 0.1 * offset, 5.0 + 0.1 * offset, rows),
            "name": [f"mol-{i % 4}" for i in range(rows)],
            "smiles.std": [f"C{i % 5}H{i % 7}" for i in range(rows)],
        }
    )
    rt_df.to_csv(ds_root / f"{dataset_id}_rtdata_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    desc = pd.DataFrame({"id": ids})
    for j in range(8):
        desc[f"desc_{j}"] = np.linspace(j, j + rows - 1, rows) + offset
    desc.to_csv(ds_root / f"{dataset_id}_descriptors_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    for fp_name, fp_size in FINGERPRINT_SIZES.items():
        fp_df = pd.DataFrame(
            {
                "id": ids,
                "bits.on": [_bits_string(fp_size, offset + i) for i in range(rows)],
            }
        )
        fp_df.to_csv(ds_root / f"{dataset_id}_fingerprints_{fp_name}_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    meta = pd.DataFrame(
        [
            {
                "column.name": f"Benchmark C18 {offset}",
                "column.usp.code": "L1" if offset % 2 else "L7",
                "column.temperature": 30.0 + offset,
                "column.flowrate": 0.30 + 0.01 * offset,
                "column.length": 100.0 + offset,
                "column.id": 2.1,
                "column.particle.size": 1.7,
                "column.t0": 1.0 + 0.05 * offset,
                "eluent.a": 60.0 - offset,
                "eluent.a.unit": "",
                "eluent.b": 40.0 + offset,
                "eluent.b.unit": "",
                "gradient.start.A": 95.0,
                "gradient.start.B": 5.0,
                "gradient.start.C": 0.0,
                "gradient.start.D": 0.0,
                "gradient.end.A": 60.0,
                "gradient.end.B": 40.0,
                "gradient.end.C": 0.0,
                "gradient.end.D": 0.0,
            }
        ]
    )
    meta.to_csv(ds_root / f"{dataset_id}_metadata.tsv", sep="\t", index=False, encoding="utf-8")

    grad = pd.DataFrame(
        {
            "time": [0.0, 2.5, 5.0],
            "A [%]": [95.0, 80.0 - offset, 60.0 - offset],
            "B [%]": [5.0, 20.0 + offset, 40.0 + offset],
            "flow [mL/min]": [0.30 + 0.01 * offset, 0.32 + 0.01 * offset, 0.34 + 0.01 * offset],
        }
    )
    grad.to_csv(ds_root / f"{dataset_id}_gradient.tsv", sep="\t", index=False, encoding="utf-8")

    pd.DataFrame([{"dataset": dataset_id, "note": "synthetic"}]).to_csv(
        ds_root / f"{dataset_id}_info.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )


def _bits_string(size: int, seed: int) -> str:
    idxs = sorted({1 + ((seed * 17 + j * 97) % size) for j in range(6)})
    return ",".join(str(x) for x in idxs)
