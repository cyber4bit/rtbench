from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import Config
from ..cpvec import ensure_cp_inputs, list_all_study_ids, load_or_train_cpvec
from ..data import build_all_matrices, ensure_repo_data, pretrain_count_14, validate_required_inputs
from ..hyper import HyperTLBundle


REQUIRED_CONFIG_SECTIONS = [
    "data",
    "datasets",
    "split",
    "models",
    "transfer_weights",
    "seeds",
    "metrics",
    "stats",
    "outputs",
]


def config_from_raw(raw: dict[str, Any]) -> Config:
    missing = [k for k in REQUIRED_CONFIG_SECTIONS if k not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return Config(**{k: raw[k] for k in REQUIRED_CONFIG_SECTIONS})


def raw_from_config(cfg: Config) -> dict[str, Any]:
    return {k: copy.deepcopy(getattr(cfg, k)) for k in REQUIRED_CONFIG_SECTIONS}


def config_sha1_from_raw(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def parse_list_expr(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def normalize_baseline_df(raw_df: pd.DataFrame, *, external_ids: list[str], baseline_method: str = "Uni-RT") -> pd.DataFrame:
    """Return a paper-baseline frame compatible with existing summary code.

    Older rtbench configs use a compact format with `paper_mae` and `paper_r2`.
    Uni-RT supplementary baselines are easier to maintain as long-format rows
    (`dataset, method, mae, medae, mre, r2`).  This adapter keeps the existing
    runner contract while preserving any extra columns for downstream reports.
    """
    if "dataset" not in raw_df.columns:
        raise ValueError("baseline_csv must contain a 'dataset' column")
    df = raw_df.copy()
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)

    if {"paper_mae", "paper_r2"}.issubset(df.columns):
        out = df.copy()
    elif {"method", "mae", "r2"}.issubset(df.columns):
        method_norm = str(baseline_method).strip().lower()
        selected = df.loc[df["method"].astype(str).str.strip().str.lower() == method_norm].copy()
        if selected.empty:
            raise ValueError(f"baseline_csv does not contain method={baseline_method!r}")
        out = selected.rename(
            columns={
                "mae": "paper_mae",
                "r2": "paper_r2",
                "medae": "paper_medae",
                "mre": "paper_mre",
            }
        )
    else:
        raise ValueError(
            "baseline_csv must contain either paper_mae/paper_r2 columns or long-format method/mae/r2 columns"
        )

    out = out.loc[out["dataset"].isin(external_ids)].copy()
    missing_base = sorted(set(external_ids) - set(out["dataset"].tolist()))
    if missing_base:
        raise ValueError(f"baseline_csv is missing datasets required for evaluation: {missing_base}")
    return out


@dataclass
class PreparedBenchmark:
    cfg: Config
    pretrain_ids: list[str]
    external_ids: list[str]
    all_ids: list[str]
    mats: dict[str, Any]
    schema: Any
    X_src: np.ndarray
    X_src_mol: np.ndarray
    X_src_cp: np.ndarray
    y_src_sec: np.ndarray
    source_row_dataset_ids: np.ndarray
    source_row_mol_keys: np.ndarray
    source_row_context_tokens: list[tuple[str, ...]]
    source_mol_key_set: set[str]
    baseline_df: pd.DataFrame
    hyper_cache: dict[str, list[HyperTLBundle]] = field(default_factory=dict)


def prepare(cfg: Config, *, external_ids: list[str] | None = None, no_download: bool = False) -> PreparedBenchmark:
    pretrain_ids = [str(x).zfill(4) for x in cfg.datasets["pretrain"]]
    external_ids_used = [str(x).zfill(4) for x in (external_ids if external_ids is not None else cfg.datasets["external"])]
    all_ids = sorted(set(pretrain_ids + external_ids_used))

    data_root = Path(cfg.data["local_root"])
    processed_root = data_root / "processed_data"
    ensure_dirs([data_root, processed_root])

    ensure_repo_data(
        repo_url=cfg.data["repo_url"],
        commit=str(cfg.data["commit"]),
        data_root=data_root,
        dataset_ids=all_ids,
        download=(not no_download),
    )
    validate_required_inputs(processed_root=processed_root, dataset_ids=all_ids)

    cpvec_cfg = dict(cfg.data.get("cpvec", {}) or {})
    cpvec_enabled = bool(cpvec_cfg.get("enabled", False))
    cpvec_map = None
    if cpvec_enabled:
        if bool(cpvec_cfg.get("use_all_studies", True)):
            cp_ids = list_all_study_ids(processed_root=processed_root)
        else:
            cp_ids = all_ids
        ensure_cp_inputs(
            repo_url=cfg.data["repo_url"],
            commit=str(cfg.data["commit"]),
            data_root=data_root,
            dataset_ids=cp_ids,
            download=(not no_download),
        )
        cp_model, _cp_dim = load_or_train_cpvec(
            data_root=data_root,
            processed_root=processed_root,
            repo_url=cfg.data["repo_url"],
            commit=str(cfg.data["commit"]),
            cfg=cpvec_cfg,
            download=(not no_download),
            dataset_ids=cp_ids,
        )
        cpvec_map = {ds: cp_model.cp_vector_for_dataset(ds_root=processed_root / ds, ds=ds) for ds in all_ids}

    schema, mats = build_all_matrices(
        processed_root=processed_root,
        dataset_ids=all_ids,
        gradient_points=int(cfg.data.get("gradient_points", 20)),
        cpvec_map=cpvec_map,
        include_molecule_context=bool(cfg.data.get("include_molecule_context", False)),
        molecule_text_ngram_dim=int(cfg.data.get("molecule_text_ngram_dim", 0) or 0),
        molecule_sequence_max_len=int(cfg.data.get("molecule_sequence_max_len", 0) or 0),
    )

    expected_pretrain_count = int(cfg.datasets.get("expected_pretrain_count", 4157))
    actual_pretrain_count = pretrain_count_14(mats=mats, pretrain_ids=pretrain_ids)
    if actual_pretrain_count != expected_pretrain_count:
        raise AssertionError(
            f"Pretrain sample count mismatch: expected={expected_pretrain_count}, actual={actual_pretrain_count}"
        )

    src_X_parts = []
    src_X_mol_parts = []
    src_X_cp_parts = []
    src_y_parts = []
    src_row_ds_parts = []
    src_row_key_parts = []
    src_row_context_tokens: list[tuple[str, ...]] = []
    for ds in pretrain_ids:
        src_X_parts.append(mats[ds].X)
        src_X_mol_parts.append(mats[ds].X_mol)
        src_X_cp_parts.append(mats[ds].X_cp)
        src_y_parts.append(mats[ds].y_sec)
        src_row_ds_parts.append(np.array([ds] * len(mats[ds].y_sec), dtype=object))
        src_row_key_parts.append(np.array(mats[ds].mol_keys, dtype=object))
        if getattr(mats[ds], "mol_context_tokens", None):
            src_row_context_tokens.extend(list(mats[ds].mol_context_tokens))
        else:
            src_row_context_tokens.extend([tuple()] * len(mats[ds].y_sec))
    X_src = np.concatenate(src_X_parts, axis=0).astype(np.float32)
    X_src_mol = np.concatenate(src_X_mol_parts, axis=0).astype(np.float32)
    X_src_cp = np.concatenate(src_X_cp_parts, axis=0).astype(np.float32)
    source_row_dataset_ids = np.concatenate(src_row_ds_parts, axis=0)
    source_row_mol_keys = np.concatenate(src_row_key_parts, axis=0)
    source_mol_key_set = {str(k) for k in source_row_mol_keys.tolist() if str(k).strip()}
    y_src_sec = np.concatenate(src_y_parts, axis=0).astype(np.float32)

    baseline_raw = pd.read_csv(cfg.data["baseline_csv"], dtype={"dataset": str}, encoding="utf-8")
    baseline_df = normalize_baseline_df(
        baseline_raw,
        external_ids=external_ids_used,
        baseline_method=str(cfg.metrics.get("baseline_method", "Uni-RT")),
    )

    return PreparedBenchmark(
        cfg=cfg,
        pretrain_ids=pretrain_ids,
        external_ids=external_ids_used,
        all_ids=all_ids,
        mats=mats,
        schema=schema,
        X_src=X_src,
        X_src_mol=X_src_mol,
        X_src_cp=X_src_cp,
        y_src_sec=y_src_sec,
        source_row_dataset_ids=source_row_dataset_ids,
        source_row_mol_keys=source_row_mol_keys,
        source_row_context_tokens=src_row_context_tokens,
        source_mol_key_set=source_mol_key_set,
        baseline_df=baseline_df,
    )
