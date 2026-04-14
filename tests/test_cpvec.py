from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from rtbench import cpvec


def _write_cp_study(
    processed_root: Path,
    dataset_id: str,
    *,
    column_name: str,
    usp_code: str,
    flowrate: float,
    temperature: float,
    t0: float,
) -> None:
    ds_root = processed_root / dataset_id
    ds_root.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        [
            {
                "column.name": column_name,
                "column.usp.code": usp_code,
                "column.temperature": temperature,
                "column.flowrate": flowrate,
                "column.length": 100.0 + int(dataset_id),
                "column.id": 2.1,
                "column.particle.size": 1.7,
                "column.t0": t0,
                "eluent.a": 60.0 - int(dataset_id[-1]),
                "eluent.a.unit": "",
                "eluent.b": 40.0 + int(dataset_id[-1]),
                "eluent.b.unit": "",
            }
        ]
    )
    meta.to_csv(ds_root / f"{dataset_id}_metadata.tsv", sep="\t", index=False, encoding="utf-8")
    grad = pd.DataFrame(
        {
            "time": [0.0, 2.0, 5.0],
            "A [%]": [95.0, 80.0 - int(dataset_id[-1]), 60.0 - int(dataset_id[-1])],
            "B [%]": [5.0, 20.0 + int(dataset_id[-1]), 40.0 + int(dataset_id[-1])],
            "flow [mL/min]": [flowrate, flowrate + 0.05, flowrate + 0.1],
        }
    )
    grad.to_csv(ds_root / f"{dataset_id}_gradient.tsv", sep="\t", index=False, encoding="utf-8")


def _make_cpvec_repo(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "repoRT"
    processed_root = data_root / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["0001", "0002", "0003", "0004"]}).to_csv(processed_root / "studies.tsv", sep="\t", index=False, encoding="utf-8")
    _write_cp_study(processed_root, "0001", column_name="Merck LiChrospher RP-18", usp_code="L1", flowrate=0.30, temperature=30.0, t0=1.0)
    _write_cp_study(processed_root, "0002", column_name="Waters BEH C18", usp_code="L1", flowrate=0.35, temperature=32.0, t0=1.2)
    _write_cp_study(processed_root, "0003", column_name="Phenomenex Luna C8", usp_code="L7", flowrate=0.40, temperature=28.0, t0=0.9)
    _write_cp_study(processed_root, "0004", column_name="Agilent Eclipse Plus C18", usp_code="L1", flowrate=0.45, temperature=35.0, t0=1.1)
    return data_root, processed_root


def _small_cpvec_cfg() -> dict[str, float | int]:
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


def test_train_ae_reconstructs_gradient_segments() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(96, 3)).astype(np.float32)
    basis = rng.normal(size=(3, 11)).astype(np.float32)
    X = latent @ basis + 0.01 * rng.normal(size=(96, 11)).astype(np.float32)

    state, mean, std = cpvec._train_ae(
        X=X,
        latent_dim=3,
        hidden=24,
        dropout=0.0,
        epochs=160,
        lr=0.02,
        seed=0,
        batch_size=32,
    )

    model = cpvec._AE(in_dim=11, latent_dim=3, hidden=24, dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    Xn = cpvec._norm(X, mean, std)
    with torch.no_grad():
        encoded = model.encode(torch.from_numpy(Xn)).numpy()
        reconstructed = model(torch.from_numpy(Xn)).numpy()

    assert encoded.shape == (96, 3)
    assert float(np.mean((reconstructed - Xn) ** 2)) < 0.02


def test_load_or_train_cpvec_builds_expected_component_dimensions(tmp_path: Path) -> None:
    data_root, processed_root = _make_cpvec_repo(tmp_path)
    cfg = _small_cpvec_cfg()

    encoder, cp_dim = cpvec.load_or_train_cpvec(
        data_root=data_root,
        processed_root=processed_root,
        repo_url="https://example.com/repo",
        commit="deadbeefcafebabe",
        cfg=cfg,
        download=False,
        dataset_ids=["0001", "0002", "0003"],
    )

    ds_root = processed_root / "0001"
    meta_row = pd.read_csv(ds_root / "0001_metadata.tsv", sep="\t", encoding="utf-8").iloc[0]
    grad_df = pd.read_csv(ds_root / "0001_gradient.tsv", sep="\t", encoding="utf-8")

    col_vec = encoder._encode_column_type(str(meta_row["column.name"]), str(meta_row["column.usp.code"]))
    mp_vec = encoder._encode_mobile_phase(meta_row)
    grad_vec = encoder._encode_gradient(grad_df)
    ccp_vec = encoder._encode_ccp(meta_row)
    full_vec = encoder.cp_vector_for_dataset(ds_root=ds_root, ds="0001")

    assert col_vec.shape == (5,)
    assert mp_vec.shape == (3,)
    assert grad_vec.shape == (4,)
    assert ccp_vec.shape == (6,)
    assert cp_dim == 18
    assert encoder.cp_dim == 18
    assert full_vec.shape == (18,)
    np.testing.assert_allclose(full_vec[-6:], ccp_vec, atol=1e-6)


def test_load_or_train_cpvec_writes_cache_and_hits_on_second_call(tmp_path: Path) -> None:
    data_root, processed_root = _make_cpvec_repo(tmp_path)
    cfg = _small_cpvec_cfg()

    with mock.patch.object(cpvec, "ensure_cp_inputs") as ensure_mock:
        encoder_1, cp_dim_1 = cpvec.load_or_train_cpvec(
            data_root=data_root,
            processed_root=processed_root,
            repo_url="https://example.com/repo",
            commit="deadbeefcafebabe",
            cfg=cfg,
            download=False,
            dataset_ids=["0001", "0002", "0003"],
        )

    cache_dir = cpvec._cache_dir(data_root=data_root, commit="deadbeefcafebabe", cfg=cfg, dataset_ids=["0001", "0002", "0003"])
    assert (cache_dir / "ae1.pt").exists()
    assert (cache_dir / "ae2.pt").exists()
    assert (cache_dir / "col_word_vecs.npy").exists()
    ensure_mock.assert_called_once()

    vec_1 = encoder_1.cp_vector_for_dataset(processed_root / "0002", "0002")
    with (
        mock.patch.object(cpvec, "ensure_cp_inputs", side_effect=AssertionError("cache should satisfy reload")),
        mock.patch.object(cpvec, "_train_ae", side_effect=AssertionError("AE retraining should not happen")),
        mock.patch.object(cpvec, "_train_word2vec", side_effect=AssertionError("Word2Vec retraining should not happen")),
    ):
        encoder_2, cp_dim_2 = cpvec.load_or_train_cpvec(
            data_root=data_root,
            processed_root=processed_root,
            repo_url="https://example.com/repo",
            commit="deadbeefcafebabe",
            cfg=cfg,
            download=False,
            dataset_ids=["0001", "0002", "0003"],
        )

    vec_2 = encoder_2.cp_vector_for_dataset(processed_root / "0002", "0002")
    assert cp_dim_1 == cp_dim_2
    np.testing.assert_allclose(vec_1, vec_2, atol=1e-6)


def test_load_or_train_cpvec_respects_use_all_studies_training_scope(tmp_path: Path) -> None:
    data_root, processed_root = _make_cpvec_repo(tmp_path)
    cfg = _small_cpvec_cfg()

    cpvec.load_or_train_cpvec(
        data_root=data_root,
        processed_root=processed_root,
        repo_url="https://example.com/repo",
        commit="deadbeefcafebabe",
        cfg=cfg,
        download=False,
        dataset_ids=["0001", "0002"],
    )
    subset_cache = cpvec._cache_dir(data_root=data_root, commit="deadbeefcafebabe", cfg=cfg, dataset_ids=["0001", "0002"])
    subset_meta = cpvec._load_json(subset_cache / "meta.json")

    all_ids = cpvec.list_all_study_ids(processed_root)
    cpvec.load_or_train_cpvec(
        data_root=data_root,
        processed_root=processed_root,
        repo_url="https://example.com/repo",
        commit="deadbeefcafebabe",
        cfg=cfg,
        download=False,
        dataset_ids=all_ids,
    )
    all_cache = cpvec._cache_dir(data_root=data_root, commit="deadbeefcafebabe", cfg=cfg, dataset_ids=all_ids)
    all_meta = cpvec._load_json(all_cache / "meta.json")

    assert subset_meta["dataset_ids"] == ["0001", "0002"]
    assert subset_meta["n_studies"] == 2
    assert all_meta["dataset_ids"] == ["0001", "0002", "0003", "0004"]
    assert all_meta["n_studies"] == 4
    assert subset_cache != all_cache
