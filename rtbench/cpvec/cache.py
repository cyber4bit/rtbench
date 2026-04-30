from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .encoders import CPEncoderArtifacts


def _cache_dir(data_root: Path, commit: str, cfg: dict[str, Any], dataset_ids: list[str] | None = None) -> Path:
    keep = {
        "col_w2v_dim": int(cfg.get("col_w2v_dim", 16)),
        "col_w2v_window": int(cfg.get("col_w2v_window", 2)),
        "col_w2v_neg": int(cfg.get("col_w2v_neg", 6)),
        "col_w2v_epochs": int(cfg.get("col_w2v_epochs", 50)),
        "col_w2v_min_count": int(cfg.get("col_w2v_min_count", 2)),
        "ae1_latent": int(cfg.get("ae1_latent", 16)),
        "ae1_hidden": int(cfg.get("ae1_hidden", 64)),
        "ae1_epochs": int(cfg.get("ae1_epochs", 600)),
        "ae2_latent": int(cfg.get("ae2_latent", 32)),
        "ae2_hidden": int(cfg.get("ae2_hidden", 256)),
        "ae2_epochs": int(cfg.get("ae2_epochs", 800)),
        "dataset_ids": sorted(str(x).zfill(4) for x in (dataset_ids or [])),
    }
    blob = json.dumps(keep, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(blob).hexdigest()[:10]
    return data_root / "cpvec_cache" / f"{commit[:8]}_{h}"


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_artifacts(dir_path: Path, art: CPEncoderArtifacts, meta: dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    _save_json(dir_path / "meta.json", meta)
    _save_json(dir_path / "col_vocab.json", art.col_vocab)
    np.save(dir_path / "col_word_vecs.npy", art.col_word_vecs.astype(np.float32))
    _save_json(dir_path / "eluent_cols.json", art.eluent_cols)
    np.savez(dir_path / "seg_norm.npz", mean=art.seg_mean.astype(np.float32), std=art.seg_std.astype(np.float32))
    np.savez(dir_path / "mp_norm.npz", mean=art.mp_mean.astype(np.float32), std=art.mp_std.astype(np.float32))
    np.savez(dir_path / "ccp_norm.npz", mean=art.ccp_mean.astype(np.float32), std=art.ccp_std.astype(np.float32))
    torch.save(
        {
            "state": art.ae1_state,
            "in_dim": art.ae1_in_dim,
            "latent": art.ae1_latent,
            "hidden": art.ae1_hidden,
            "dropout": art.ae1_dropout,
        },
        dir_path / "ae1.pt",
    )
    torch.save(
        {
            "state": art.ae2_state,
            "in_dim": art.ae2_in_dim,
            "latent": art.ae2_latent,
            "hidden": art.ae2_hidden,
            "dropout": art.ae2_dropout,
        },
        dir_path / "ae2.pt",
    )


def _load_artifacts(dir_path: Path) -> CPEncoderArtifacts:
    col_vocab = {str(k): int(v) for k, v in _load_json(dir_path / "col_vocab.json").items()}
    col_word_vecs = np.load(dir_path / "col_word_vecs.npy").astype(np.float32)
    eluent_cols = [str(x) for x in _load_json(dir_path / "eluent_cols.json")]
    seg_norm = np.load(dir_path / "seg_norm.npz")
    mp_norm = np.load(dir_path / "mp_norm.npz")
    ccp_norm = np.load(dir_path / "ccp_norm.npz")
    ae1 = torch.load(dir_path / "ae1.pt", map_location="cpu")
    ae2 = torch.load(dir_path / "ae2.pt", map_location="cpu")
    return CPEncoderArtifacts(
        col_vocab=col_vocab,
        col_word_vecs=col_word_vecs,
        col_dim=int(col_word_vecs.shape[1]),
        ae1_state=ae1["state"],
        ae1_in_dim=int(ae1["in_dim"]),
        ae1_latent=int(ae1["latent"]),
        ae1_hidden=int(ae1["hidden"]),
        ae1_dropout=float(ae1["dropout"]),
        seg_mean=seg_norm["mean"].astype(np.float32),
        seg_std=seg_norm["std"].astype(np.float32),
        ae2_state=ae2["state"],
        ae2_in_dim=int(ae2["in_dim"]),
        ae2_latent=int(ae2["latent"]),
        ae2_hidden=int(ae2["hidden"]),
        ae2_dropout=float(ae2["dropout"]),
        mp_mean=mp_norm["mean"].astype(np.float32),
        mp_std=mp_norm["std"].astype(np.float32),
        eluent_cols=eluent_cols,
        ccp_mean=ccp_norm["mean"].astype(np.float32),
        ccp_std=ccp_norm["std"].astype(np.float32),
    )


__all__ = ["_cache_dir", "_load_artifacts", "_load_json", "_save_artifacts", "_save_json"]
