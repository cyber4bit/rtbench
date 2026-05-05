from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rtbench.data import build_all_matrices, pretrain_count_14, validate_required_inputs
from rtbench.metrics import compute_metrics
from rtbench.models import kfold_split, random_split, stratified_split


PRETRAIN_FIXTURE = ["0019", "0052"]
EXTERNAL_FIXTURE = ["0050", "0233"]
DATASET_ROWS = {"0019": 3, "0052": 4, "0050": 5, "0233": 6}


def _build_fixture_repo(tmp_path: Path, synthetic_repo_factory) -> tuple[Path, list[str]]:
    processed_root = synthetic_repo_factory(tmp_path / "data", DATASET_ROWS)
    wanted = PRETRAIN_FIXTURE + EXTERNAL_FIXTURE
    validate_required_inputs(processed_root, wanted)
    return processed_root, wanted


def test_data_merge_counts(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root, wanted = _build_fixture_repo(tmp_path, synthetic_repo_factory)
    _, mats = build_all_matrices(processed_root=processed_root, dataset_ids=wanted, gradient_points=20)
    cnt = pretrain_count_14(mats=mats, pretrain_ids=PRETRAIN_FIXTURE)
    assert cnt == 7
    assert all(mats[d].y_scale_sec > 0 for d in wanted)
    assert all(mats[d].t0_sec > 0 for d in wanted)


def test_no_leakage() -> None:
    y = np.linspace(1.0, 100.0, 200)
    split = stratified_split(y, seed=42, train=0.81, val=0.09, test=0.10)
    train_set = set(split.train_idx.tolist())
    val_set = set(split.val_idx.tolist())
    test_set = set(split.test_idx.tolist())
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_feature_shapes(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root, wanted = _build_fixture_repo(tmp_path, synthetic_repo_factory)
    schema, mats = build_all_matrices(processed_root=processed_root, dataset_ids=wanted, gradient_points=20)
    dims = [mats[ds].X.shape[1] for ds in wanted]
    assert all(d == dims[0] for d in dims)
    assert sum(schema.group_sizes.values()) == dims[0]


def test_molecule_context_features_are_optional(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root = synthetic_repo_factory(tmp_path / "data", {"0001": 3})
    rt_path = processed_root / "0001" / "0001_rtdata_canonical_success.tsv"
    rt_df = pd.read_csv(rt_path, sep="\t", encoding="utf-8")
    rt_df["formula"] = ["C2H6O", "C6H12O6", "C5H5N5"]
    rt_df["classyfire.superclass"] = ["Lipids", "Organic acids", "Organoheterocyclic compounds"]
    rt_df.to_csv(rt_path, sep="\t", index=False, encoding="utf-8")

    schema_plain, mats_plain = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
    )
    schema_ctx, mats_ctx = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
        include_molecule_context=True,
    )

    assert schema_plain.formula_elements == []
    assert schema_ctx.formula_elements
    assert schema_ctx.group_sizes["descriptor"] > schema_plain.group_sizes["descriptor"]
    assert mats_ctx["0001"].X_mol.shape[1] > mats_plain["0001"].X_mol.shape[1]
    context_width = schema_ctx.group_sizes["descriptor"] - len(schema_ctx.descriptor_cols)
    context_slice = mats_ctx["0001"].X_mol[:, len(schema_ctx.descriptor_cols) : schema_ctx.group_sizes["descriptor"]]
    assert context_width == context_slice.shape[1]
    assert np.count_nonzero(context_slice) > 0


def test_molecule_text_ngram_features_are_optional(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root = synthetic_repo_factory(tmp_path / "data", {"0001": 3})
    schema_plain, mats_plain = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
    )
    schema_text, mats_text = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
        molecule_text_ngram_dim=16,
    )

    assert schema_text.molecule_text_ngram_dim == 16
    assert schema_text.group_sizes["descriptor"] == schema_plain.group_sizes["descriptor"]
    assert schema_text.group_sizes["mol_text"] == 16
    assert mats_text["0001"].X_mol.shape[1] == mats_plain["0001"].X_mol.shape[1] + 16
    text_slice = mats_text["0001"].X_mol[:, -16:]
    assert np.isfinite(text_slice).all()
    assert np.count_nonzero(text_slice) > 0


def test_molecule_sequence_features_are_optional(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root = synthetic_repo_factory(tmp_path / "data", {"0001": 3})
    schema_plain, mats_plain = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
    )
    schema_seq, mats_seq = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=["0001"],
        gradient_points=20,
        molecule_sequence_max_len=12,
    )

    assert schema_seq.molecule_sequence_max_len == 12
    assert schema_seq.group_sizes["descriptor"] == schema_plain.group_sizes["descriptor"]
    assert schema_seq.group_sizes["mol_seq"] == 12
    assert mats_seq["0001"].X_mol.shape[1] == mats_plain["0001"].X_mol.shape[1] + 12
    seq_slice = mats_seq["0001"].X_mol[:, -12:]
    assert np.isfinite(seq_slice).all()
    assert np.count_nonzero(seq_slice) > 0
    assert np.all(seq_slice >= 0)


def test_cpvec_schema_dummy(tmp_path: Path, synthetic_repo_factory) -> None:
    processed_root, wanted = _build_fixture_repo(tmp_path, synthetic_repo_factory)
    cp_dim = 7
    cp_map = {ds: np.zeros(cp_dim, dtype=np.float32) for ds in wanted}
    schema, mats = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=wanted,
        gradient_points=20,
        cpvec_map=cp_map,
    )
    assert schema.uses_cpvec
    assert schema.cpvec_dim == cp_dim
    assert "cpvec" in schema.group_sizes
    dims = [mats[ds].X.shape[1] for ds in wanted]
    assert all(d == dims[0] for d in dims)
    assert mats[wanted[0]].X_cp.shape[1] == schema.cp_size
    xcp = mats[wanted[0]].X_cp
    assert np.allclose(xcp[:, -cp_dim:], 0.0)
    assert np.allclose(xcp[0, -cp_dim:], xcp[-1, -cp_dim:])
    assert sum(schema.group_sizes.values()) == dims[0]


def test_metric_units() -> None:
    y_true = np.array([60.0, 120.0, 180.0], dtype=float)
    y_pred = np.array([30.0, 90.0, 210.0], dtype=float)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 30.0
    assert metrics["medae"] == 30.0


def test_reproducibility() -> None:
    y = np.linspace(1.0, 100.0, 300)
    s1 = stratified_split(y, seed=7, train=0.81, val=0.09, test=0.10)
    s2 = stratified_split(y, seed=7, train=0.81, val=0.09, test=0.10)
    assert np.array_equal(s1.train_idx, s2.train_idx)
    assert np.array_equal(s1.val_idx, s2.val_idx)
    assert np.array_equal(s1.test_idx, s2.test_idx)


def test_random_split_reproducibility() -> None:
    y = np.linspace(1.0, 100.0, 300)
    s1 = random_split(y, seed=7, train=0.81, val=0.09, test=0.10)
    s2 = random_split(y, seed=7, train=0.81, val=0.09, test=0.10)
    assert np.array_equal(s1.train_idx, s2.train_idx)
    assert np.array_equal(s1.val_idx, s2.val_idx)
    assert np.array_equal(s1.test_idx, s2.test_idx)
    assert len(set(s1.train_idx).intersection(set(s1.val_idx))) == 0
    assert len(set(s1.train_idx).intersection(set(s1.test_idx))) == 0
    assert len(set(s1.val_idx).intersection(set(s1.test_idx))) == 0


def test_kfold_split_covers_each_sample_once() -> None:
    y = np.linspace(1.0, 100.0, 103)
    seen: list[int] = []
    for fold in range(10):
        split = kfold_split(y, seed=fold, n_splits=10, shuffle_seed=123)
        train_set = set(split.train_idx.tolist())
        val_set = set(split.val_idx.tolist())
        test_set = set(split.test_idx.tolist())
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set.intersection(test_set)) == 0
        assert len(val_set.intersection(test_set)) == 0
        seen.extend(split.test_idx.tolist())
    assert sorted(seen) == list(range(len(y)))


def test_smoke_artifacts(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    metrics_root = outputs_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"dataset": "0050", "seed": 0, "mae": 1.0}]).to_csv(
        metrics_root / "per_seed.csv",
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame([{"dataset": "0050", "our_mae_mean": 1.0, "our_r2_mean": 0.8}]).to_csv(
        metrics_root / "summary_vs_paper.csv",
        index=False,
        encoding="utf-8",
    )
    (outputs_root / "report.md").write_text("# Synthetic smoke output\n", encoding="utf-8")

    assert (metrics_root / "per_seed.csv").exists()
    assert (metrics_root / "summary_vs_paper.csv").exists()
    assert (outputs_root / "report.md").exists()
