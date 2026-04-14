from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rtbench.data import FINGERPRINT_SIZES


def _write_synthetic_dataset(
    processed_root: Path,
    dataset_id: str,
    *,
    row_count: int,
    descriptor_all_nan: bool = False,
    missing_fingerprints: set[str] | None = None,
) -> None:
    missing_fingerprints = set(missing_fingerprints or set())
    ds_root = processed_root / dataset_id
    ds_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "id": dataset_id,
                "lab.temp": 30.0 + int(dataset_id[-1]),
                "column.name": "C18",
                "column.usp.code": "L1",
                "column.length": 100.0,
                "column.id": 2.1,
                "column.t0": 1.5,
                "column.temperature": 35.0,
                "eluent.A": 95.0,
                "eluent.B": 5.0,
                "gradient.start.A": 95.0,
                "gradient.start.B": 5.0,
                "gradient.start.C": 0.0,
                "gradient.start.D": 0.0,
                "gradient.end.A": 5.0,
                "gradient.end.B": 95.0,
                "gradient.end.C": 0.0,
                "gradient.end.D": 0.0,
            }
        ]
    ).to_csv(ds_root / f"{dataset_id}_metadata.tsv", sep="\t", index=False, encoding="utf-8")

    pd.DataFrame(
        [
            {"time": 0.0, "A [%]": 95.0, "B [%]": 5.0, "C [%]": 0.0, "D [%]": 0.0, "flow [mL/min]": 0.30},
            {"time": 5.0, "A [%]": 50.0, "B [%]": 50.0, "C [%]": 0.0, "D [%]": 0.0, "flow [mL/min]": 0.35},
            {"time": 10.0, "A [%]": 5.0, "B [%]": 95.0, "C [%]": 0.0, "D [%]": 0.0, "flow [mL/min]": 0.40},
        ]
    ).to_csv(ds_root / f"{dataset_id}_gradient.tsv", sep="\t", index=False, encoding="utf-8")

    pd.DataFrame(
        [{"id": f"{dataset_id}_{i}", "rt": 1.0 + i, "smiles.std": f"C{i}", "name": f"mol-{dataset_id}-{i}"} for i in range(row_count)]
    ).to_csv(ds_root / f"{dataset_id}_rtdata_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    desc_a_values = [float("nan")] * row_count if descriptor_all_nan else [float(i) + 0.1 for i in range(row_count)]
    desc_b_values = [float("nan")] * row_count if descriptor_all_nan else [float(i) + 1.0 for i in range(row_count)]
    pd.DataFrame(
        [
            {"id": f"{dataset_id}_{i}", "desc_a": desc_a_values[i], "desc_b": desc_b_values[i]}
            for i in range(row_count)
        ]
    ).to_csv(ds_root / f"{dataset_id}_descriptors_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    for fp_name in FINGERPRINT_SIZES:
        if fp_name in missing_fingerprints:
            continue
        pd.DataFrame(
            [
                {"id": f"{dataset_id}_{i}", "bits.on": "1,2,5" if i % 2 == 0 else "3,4,6"}
                for i in range(row_count)
            ]
        ).to_csv(ds_root / f"{dataset_id}_fingerprints_{fp_name}_canonical_success.tsv", sep="\t", index=False, encoding="utf-8")

    (ds_root / f"{dataset_id}_info.tsv").write_text("key\tvalue\nsource\tsynthetic\n", encoding="utf-8")


@pytest.fixture
def synthetic_repo_factory():
    def _make(
        root: Path,
        dataset_rows: dict[str, int],
        *,
        descriptor_all_nan: set[str] | None = None,
        missing_fingerprints: dict[str, set[str]] | None = None,
    ) -> Path:
        processed_root = root / "repoRT" / "processed_data"
        processed_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": sorted(dataset_rows.keys())}).to_csv(processed_root / "studies.tsv", sep="\t", index=False, encoding="utf-8")
        descriptor_all_nan = set(descriptor_all_nan or set())
        missing_fingerprints = dict(missing_fingerprints or {})
        for dataset_id, row_count in sorted(dataset_rows.items()):
            _write_synthetic_dataset(
                processed_root,
                dataset_id,
                row_count=int(row_count),
                descriptor_all_nan=dataset_id in descriptor_all_nan,
                missing_fingerprints=set(missing_fingerprints.get(dataset_id, set())),
            )
        return processed_root

    return _make
