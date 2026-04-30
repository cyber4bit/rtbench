from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data import _download_file
from .cache import _cache_dir, _load_artifacts, _load_json, _save_artifacts, _save_json
from .encoders import (
    CPEncoder,
    CPEncoderArtifacts,
    _AE,
    _build_vocab,
    _fit_norm,
    _make_pairs,
    _norm,
    _seqs_to_ids,
    _train_ae,
    _train_word2vec,
)
from .features import _as_float, _gradient_segments, _slug, _split_words, _unit_scale


def ensure_cp_inputs(
    repo_url: str,
    commit: str,
    data_root: Path,
    dataset_ids: list[str],
    download: bool,
) -> None:
    processed_root = data_root / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)
    if download:
        studies_url = f"{repo_url}/raw/{commit}/processed_data/studies.tsv"
        _download_file(studies_url, processed_root / "studies.tsv")
    for ds in dataset_ids:
        ds_root = processed_root / ds
        ds_root.mkdir(parents=True, exist_ok=True)
        for name in (f"{ds}_metadata.tsv", f"{ds}_gradient.tsv"):
            local_file = ds_root / name
            if local_file.exists() or not download:
                continue
            url = f"{repo_url}/raw/{commit}/processed_data/{ds}/{name}"
            try:
                _download_file(url, local_file)
            except RuntimeError:
                pass


def list_all_study_ids(processed_root: Path) -> list[str]:
    studies = processed_root / "studies.tsv"
    if not studies.exists():
        return sorted([p.name for p in processed_root.iterdir() if p.is_dir()])
    df = pd.read_csv(studies, sep="\t", dtype={"id": str}, encoding="utf-8")
    ids = [str(x).zfill(4) for x in df["id"].tolist() if isinstance(x, str) or np.isfinite(x)]
    return sorted(set(ids))


def load_or_train_cpvec(
    data_root: Path,
    processed_root: Path,
    repo_url: str,
    commit: str,
    cfg: dict[str, float | int],
    download: bool,
    dataset_ids: list[str] | None = None,
) -> tuple[CPEncoder, int]:
    train_ids = sorted(str(x).zfill(4) for x in (dataset_ids or list_all_study_ids(processed_root)))
    cache = _cache_dir(data_root=data_root, commit=commit, cfg=cfg, dataset_ids=train_ids)
    if (cache / "ae1.pt").exists() and (cache / "ae2.pt").exists() and (cache / "col_word_vecs.npy").exists():
        art = _load_artifacts(cache)
        enc = CPEncoder(art=art, device=torch.device("cpu"))
        return enc, int(enc.cp_dim)

    ensure_cp_inputs(repo_url=repo_url, commit=commit, data_root=data_root, dataset_ids=train_ids, download=download)

    meta_rows: list[tuple[str, pd.Series]] = []
    eluent_cols: set[str] = set()
    segs: list[np.ndarray] = []
    col_seqs: list[list[str]] = []
    ccp_rows: list[np.ndarray] = []

    for ds in train_ids:
        ds_root = processed_root / ds
        meta_path = ds_root / f"{ds}_metadata.tsv"
        if not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path, sep="\t", encoding="utf-8")
        if meta.empty:
            continue
        row = meta.iloc[0]
        meta_rows.append((ds, row))

        col_seqs.append(_split_words(str(row.get("column.name", "__missing__"))) + [_slug(str(row.get("column.usp.code", "__missing__")))])

        for c in meta.columns:
            if c.startswith("eluent.") and not c.endswith(".unit"):
                eluent_cols.add(c)

        ccp_rows.append(
            np.asarray(
                [
                    _as_float(row.get("column.temperature", np.nan)),
                    _as_float(row.get("column.flowrate", np.nan)),
                    _as_float(row.get("column.length", np.nan)),
                    _as_float(row.get("column.id", np.nan)),
                    _as_float(row.get("column.particle.size", np.nan)),
                    _as_float(row.get("column.t0", np.nan)),
                ],
                dtype=np.float32,
            )
        )

        grad_path = ds_root / f"{ds}_gradient.tsv"
        if grad_path.exists():
            g = pd.read_csv(grad_path, sep="\t", encoding="utf-8")
            segs.extend(_gradient_segments(g))

    if not meta_rows:
        raise RuntimeError("No metadata.tsv available to train CP encoder.")
    if not segs:
        raise RuntimeError("No gradient segments available to train AE-1.")

    vocab = _build_vocab(col_seqs, min_count=int(cfg.get("col_w2v_min_count", 2)))
    col_ids = _seqs_to_ids(col_seqs, vocab=vocab)
    col_dim = int(cfg.get("col_w2v_dim", 16))
    col_wv = _train_word2vec(
        docs_ids=col_ids,
        vocab_size=len(vocab),
        dim=col_dim,
        window=int(cfg.get("col_w2v_window", 2)),
        neg_k=int(cfg.get("col_w2v_neg", 6)),
        epochs=int(cfg.get("col_w2v_epochs", 50)),
        lr=float(cfg.get("col_w2v_lr", 0.01)),
        batch_size=int(cfg.get("col_w2v_batch_size", 512)),
        seed=int(cfg.get("seed", 0)),
    )

    seg_mat = np.vstack(segs).astype(np.float32)
    ae1_state, seg_mean, seg_std = _train_ae(
        X=seg_mat,
        latent_dim=int(cfg.get("ae1_latent", 16)),
        hidden=int(cfg.get("ae1_hidden", 64)),
        dropout=float(cfg.get("ae1_dropout", 0.10)),
        epochs=int(cfg.get("ae1_epochs", 600)),
        lr=float(cfg.get("ae1_lr", 0.01)),
        seed=int(cfg.get("seed", 0)) + 11,
        batch_size=int(cfg.get("ae1_batch_size", 256)),
    )

    eluent_cols_sorted = sorted(eluent_cols)
    mp_rows = []
    for _, row in meta_rows:
        vals = []
        for c in eluent_cols_sorted:
            v = _as_float(row.get(c, 0.0))
            if not np.isfinite(v):
                v = 0.0
            unit = row.get(f"{c}.unit", "")
            v = float(v) * _unit_scale(str(unit))
            vals.append(v)
        mp_rows.append(np.asarray(vals, dtype=np.float32))
    mp_mat = np.vstack(mp_rows).astype(np.float32)
    ae2_state, mp_mean, mp_std = _train_ae(
        X=mp_mat,
        latent_dim=int(cfg.get("ae2_latent", 32)),
        hidden=int(cfg.get("ae2_hidden", 256)),
        dropout=float(cfg.get("ae2_dropout", 0.10)),
        epochs=int(cfg.get("ae2_epochs", 800)),
        lr=float(cfg.get("ae2_lr", 0.01)),
        seed=int(cfg.get("seed", 0)) + 17,
        batch_size=int(cfg.get("ae2_batch_size", 128)),
    )

    ccp_mat = np.vstack(ccp_rows).astype(np.float32)
    ccp_mat = np.where(np.isfinite(ccp_mat), ccp_mat, 0.0).astype(np.float32)
    ccp_mean, ccp_std = _fit_norm(ccp_mat)

    art = CPEncoderArtifacts(
        col_vocab=vocab,
        col_word_vecs=col_wv,
        col_dim=col_dim,
        ae1_state=ae1_state,
        ae1_in_dim=int(seg_mat.shape[1]),
        ae1_latent=int(cfg.get("ae1_latent", 16)),
        ae1_hidden=int(cfg.get("ae1_hidden", 64)),
        ae1_dropout=float(cfg.get("ae1_dropout", 0.10)),
        seg_mean=seg_mean,
        seg_std=seg_std,
        ae2_state=ae2_state,
        ae2_in_dim=int(mp_mat.shape[1]),
        ae2_latent=int(cfg.get("ae2_latent", 32)),
        ae2_hidden=int(cfg.get("ae2_hidden", 256)),
        ae2_dropout=float(cfg.get("ae2_dropout", 0.10)),
        mp_mean=mp_mean,
        mp_std=mp_std,
        eluent_cols=eluent_cols_sorted,
        ccp_mean=ccp_mean,
        ccp_std=ccp_std,
    )
    meta = {
        "commit": commit,
        "cache_dir": str(cache),
        "dataset_ids": train_ids,
        "col_vocab_size": len(vocab),
        "col_dim": col_dim,
        "eluent_dim": int(mp_mat.shape[1]),
        "n_studies": int(len(meta_rows)),
        "n_segments": int(len(segs)),
        "ae1_latent": int(cfg.get("ae1_latent", 16)),
        "ae2_latent": int(cfg.get("ae2_latent", 32)),
    }
    _save_artifacts(cache, art, meta=meta)
    enc = CPEncoder(art=art, device=torch.device("cpu"))
    return enc, int(enc.cp_dim)


__all__ = [
    "CPEncoder",
    "CPEncoderArtifacts",
    "_AE",
    "_as_float",
    "_cache_dir",
    "_fit_norm",
    "_gradient_segments",
    "_load_artifacts",
    "_load_json",
    "_make_pairs",
    "_norm",
    "_save_artifacts",
    "_save_json",
    "_seqs_to_ids",
    "_slug",
    "_split_words",
    "_train_ae",
    "_train_word2vec",
    "_unit_scale",
    "ensure_cp_inputs",
    "list_all_study_ids",
    "load_or_train_cpvec",
]
