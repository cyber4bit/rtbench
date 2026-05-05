from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data import FINGERPRINT_SIZES
from ..hyper import HyperTLBundle, pretrain_hyper_tl
from ..metrics import compute_metrics
from ..models import kfold_split, random_split, stratified_split, train_and_ensemble
from ..report import write_report
from ..report_vs_unirt import write_unirt_report
from ..stats import summarize_vs_paper
from .prepare import PreparedBenchmark, config_sha1_from_raw, ensure_dirs
from .weighting import build_adaptive_source_weights, expand_per_dataset_multiplier, normalize_target_transform


logger = logging.getLogger(__name__)


def aggregate_group_importance(entries: list[dict[str, float]]) -> dict[str, float]:
    if not entries:
        return {}
    keys = sorted({k for e in entries for k in e.keys()})
    out = {}
    for k in keys:
        vals = [e.get(k, 0.0) for e in entries]
        out[k] = float(np.mean(vals))
    return out


def load_previous_failed(summary_path: Path, external_ids: list[str]) -> set[str]:
    if not summary_path.exists():
        return set()
    try:
        df = pd.read_csv(summary_path, dtype={"dataset": str}, encoding="utf-8")
    except Exception:
        return set()
    if "dataset" not in df.columns or "win_both" not in df.columns:
        return set()
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    failed = set(df.loc[~df["win_both"].astype(bool), "dataset"].tolist())
    return {x for x in failed if x in set(external_ids)}


def write_per_seed_csv(per_seed_df: pd.DataFrame, out_path: Path, external_ids: list[str]) -> pd.DataFrame:
    """Persist per-seed metrics incrementally so long runs can be resumed safely."""
    if per_seed_df.empty:
        return per_seed_df
    per_seed_df = per_seed_df.copy()
    per_seed_df["dataset"] = per_seed_df["dataset"].astype(str).str.zfill(4)
    per_seed_df = per_seed_df.loc[per_seed_df["dataset"].isin(external_ids)].copy()
    per_seed_df = (
        per_seed_df.drop_duplicates(subset=["dataset", "seed"], keep="last")
        .sort_values(["dataset", "seed"])
        .reset_index(drop=True)
    )
    required_cols = ["dataset", "seed", "mae", "medae", "mre", "medre", "r2", "rmse"]
    extra_cols = [c for c in per_seed_df.columns if c not in required_cols]
    per_seed_df = per_seed_df[required_cols + extra_cols]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(out_path, index=False, encoding="utf-8")
    return per_seed_df


def append_candidate_diagnostics_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path, dtype={"dataset": str}, encoding="utf-8")
        except Exception:
            existing = pd.DataFrame()
        if "dataset" in existing.columns:
            existing["dataset"] = existing["dataset"].astype(str).str.zfill(4)
        columns = list(existing.columns)
        for column in df.columns:
            if column not in columns:
                columns.append(column)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True, sort=False)
        df = df.reindex(columns=columns)
    df.to_csv(out_path, index=False, encoding="utf-8")


def _diagnostic_scalar(value: Any) -> Any | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, str):
        return value
    return None


def _candidate_diagnostic_csv_row(
    *,
    dataset: str,
    seed: int,
    auto_policy_rule: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    diag_row = {
        "dataset": str(dataset).zfill(4),
        "seed": int(seed),
        "auto_policy_rule": str(auto_policy_rule),
        "candidate_rank": int(row.get("rank", 0)),
        "candidate_name": str(row.get("name", "")),
        "val_mae": float(row.get("val_mae", np.nan)),
        "val_r2": float(row.get("val_r2", np.nan)),
        "selected": bool(row.get("selected", False)),
        "weight": float(row.get("weight", 0.0)),
    }
    reserved = {
        "rank",
        "name",
        "val_mae",
        "val_r2",
        "selected",
        "weight",
    }
    for key, value in row.items():
        if key in reserved:
            continue
        scalar = _diagnostic_scalar(value)
        if scalar is not None:
            diag_row[str(key)] = scalar
    return diag_row


def _id_set(raw: Any) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, (str, bytes)):
        values = [part for part in str(raw).replace(";", ",").split(",")]
    else:
        try:
            values = list(raw)
        except TypeError:
            values = [raw]
    return {str(value).strip().zfill(4) for value in values if str(value).strip()}


@dataclass
class TrialResult:
    out_root: Path
    per_seed_df: pd.DataFrame
    summary_df: pd.DataFrame
    avg_mae: float
    avg_r2: float
    wins: int
    success: bool
    early_stop_reason: str = ""


def _hyper_cache_key(
    target_transform: str,
    hyper_cfg: dict[str, Any],
    balance_pretrain_by_dataset: bool,
) -> str:
    # Exclude ridge_lambdas/ridge_lambda_b (adaptation-only) so we can reuse the same pretrained model.
    key_fields = {
        "target_transform": str(target_transform),
        "balance_pretrain_by_dataset": bool(balance_pretrain_by_dataset),
        "n_models": int(hyper_cfg.get("n_models", 1)),
        "embed_dim": int(hyper_cfg.get("embed_dim", 128)),
        "mol_hidden": int(hyper_cfg.get("mol_hidden", 512)),
        "cp_hidden": int(hyper_cfg.get("cp_hidden", 256)),
        "use_mdl_subset_mol": bool(hyper_cfg.get("use_mdl_subset_mol", False)),
        "use_mol_context": bool(hyper_cfg.get("use_mol_context", True)),
        "use_mol_text_ngram": bool(hyper_cfg.get("use_mol_text_ngram", False)),
        "use_mol_sequence": bool(hyper_cfg.get("use_mol_sequence", False)),
        "mol_sequence_len": int(hyper_cfg.get("mol_sequence_len", 0) or 0),
        "mol_sequence_vocab_size": int(hyper_cfg.get("mol_sequence_vocab_size", 0) or 0),
        "mol_sequence_embed_dim": int(hyper_cfg.get("mol_sequence_embed_dim", 16)),
        "mol_sequence_channels": int(hyper_cfg.get("mol_sequence_channels", 32)),
        "mol_sequence_weight": float(hyper_cfg.get("mol_sequence_weight", 1.0)),
        "dropout": float(hyper_cfg.get("dropout", 0.10)),
        "use_film": bool(hyper_cfg.get("use_film", False)),
        "film_scale": float(hyper_cfg.get("film_scale", 0.25)),
        "use_conditioned_embeddings": bool(
            hyper_cfg.get("use_conditioned_embeddings", bool(hyper_cfg.get("use_film", False)))
        ),
        "use_task_adapters": bool(hyper_cfg.get("use_task_adapters", False)),
        "adapter_reduction": int(hyper_cfg.get("adapter_reduction", 4)),
        "use_cross_stitch": bool(hyper_cfg.get("use_cross_stitch", False)),
        "cross_stitch_init": float(hyper_cfg.get("cross_stitch_init", 0.05)),
        "epochs": int(hyper_cfg.get("epochs", 60)),
        "batch_size": int(hyper_cfg.get("batch_size", 256)),
        "lr": float(hyper_cfg.get("lr", 1e-3)),
        "weight_decay": float(hyper_cfg.get("weight_decay", 1e-4)),
        "val_frac": float(hyper_cfg.get("val_frac", 0.10)),
        "patience": int(hyper_cfg.get("patience", 8)),
    }
    return config_sha1_from_raw(key_fields)


def _hilic_hyper_cache_key(
    *,
    target_transform: str,
    hyper_cfg: dict[str, Any],
    balance_pretrain_by_dataset: bool,
    target_ds: str,
    seed: int,
    source_ids: list[str],
    source_rows: int,
) -> str:
    key_fields = {
        "kind": "hilic_external_train_val_hyper",
        "target_transform": str(target_transform),
        "balance_pretrain_by_dataset": bool(balance_pretrain_by_dataset),
        "target_ds": str(target_ds).zfill(4),
        "seed": int(seed),
        "source_ids": sorted(str(x).zfill(4) for x in source_ids),
        "source_rows": int(source_rows),
        "n_models": int(hyper_cfg.get("n_models", 1)),
        "embed_dim": int(hyper_cfg.get("embed_dim", 128)),
        "mol_hidden": int(hyper_cfg.get("mol_hidden", 512)),
        "cp_hidden": int(hyper_cfg.get("cp_hidden", 256)),
        "use_mdl_subset_mol": bool(hyper_cfg.get("use_mdl_subset_mol", False)),
        "use_mol_context": bool(hyper_cfg.get("use_mol_context", True)),
        "use_mol_text_ngram": bool(hyper_cfg.get("use_mol_text_ngram", False)),
        "use_mol_sequence": bool(hyper_cfg.get("use_mol_sequence", False)),
        "mol_sequence_len": int(hyper_cfg.get("mol_sequence_len", 0) or 0),
        "mol_sequence_vocab_size": int(hyper_cfg.get("mol_sequence_vocab_size", 0) or 0),
        "mol_sequence_embed_dim": int(hyper_cfg.get("mol_sequence_embed_dim", 16)),
        "mol_sequence_channels": int(hyper_cfg.get("mol_sequence_channels", 32)),
        "mol_sequence_weight": float(hyper_cfg.get("mol_sequence_weight", 1.0)),
        "dropout": float(hyper_cfg.get("dropout", 0.10)),
        "use_film": bool(hyper_cfg.get("use_film", False)),
        "film_scale": float(hyper_cfg.get("film_scale", 0.25)),
        "use_conditioned_embeddings": bool(
            hyper_cfg.get("use_conditioned_embeddings", bool(hyper_cfg.get("use_film", False)))
        ),
        "use_task_adapters": bool(hyper_cfg.get("use_task_adapters", False)),
        "adapter_reduction": int(hyper_cfg.get("adapter_reduction", 4)),
        "use_cross_stitch": bool(hyper_cfg.get("use_cross_stitch", False)),
        "cross_stitch_init": float(hyper_cfg.get("cross_stitch_init", 0.05)),
        "epochs": int(hyper_cfg.get("epochs", 60)),
        "batch_size": int(hyper_cfg.get("batch_size", 256)),
        "lr": float(hyper_cfg.get("lr", 1e-3)),
        "weight_decay": float(hyper_cfg.get("weight_decay", 1e-4)),
        "val_frac": float(hyper_cfg.get("val_frac", 0.10)),
        "patience": int(hyper_cfg.get("patience", 8)),
    }
    return "hilic:" + config_sha1_from_raw(key_fields)


def _sheet_unified_hyper_cache_key(
    *,
    target_transform: str,
    hyper_cfg: dict[str, Any],
    balance_pretrain_by_dataset: bool,
    seed: int,
    include_base_source: bool,
    external_ids: list[str],
    conditioning: str,
    task_ids: list[str],
    split_cfg: dict[str, Any],
    source_rows: int,
    external_rows: int,
) -> str:
    key_fields = {
        "kind": "sheet_unified_external_train_val_hyper",
        "target_transform": str(target_transform),
        "balance_pretrain_by_dataset": bool(balance_pretrain_by_dataset),
        "seed": int(seed),
        "include_base_source": bool(include_base_source),
        "external_ids": sorted(str(x).zfill(4) for x in external_ids),
        "conditioning": str(conditioning),
        "task_ids": sorted(str(x).zfill(4) for x in task_ids),
        "split": {
            "strategy": str(split_cfg.get("strategy", "")),
            "train": float(split_cfg.get("train", 0.0)),
            "val": float(split_cfg.get("val", 0.0)),
            "test": float(split_cfg.get("test", 0.0)),
        },
        "source_rows": int(source_rows),
        "external_train_val_rows": int(external_rows),
        "n_models": int(hyper_cfg.get("n_models", 1)),
        "embed_dim": int(hyper_cfg.get("embed_dim", 128)),
        "mol_hidden": int(hyper_cfg.get("mol_hidden", 512)),
        "cp_hidden": int(hyper_cfg.get("cp_hidden", 256)),
        "use_mdl_subset_mol": bool(hyper_cfg.get("use_mdl_subset_mol", False)),
        "use_mol_context": bool(hyper_cfg.get("use_mol_context", True)),
        "use_mol_text_ngram": bool(hyper_cfg.get("use_mol_text_ngram", False)),
        "use_mol_sequence": bool(hyper_cfg.get("use_mol_sequence", False)),
        "mol_sequence_len": int(hyper_cfg.get("mol_sequence_len", 0) or 0),
        "mol_sequence_vocab_size": int(hyper_cfg.get("mol_sequence_vocab_size", 0) or 0),
        "mol_sequence_embed_dim": int(hyper_cfg.get("mol_sequence_embed_dim", 16)),
        "mol_sequence_channels": int(hyper_cfg.get("mol_sequence_channels", 32)),
        "mol_sequence_weight": float(hyper_cfg.get("mol_sequence_weight", 1.0)),
        "dropout": float(hyper_cfg.get("dropout", 0.10)),
        "use_film": bool(hyper_cfg.get("use_film", False)),
        "film_scale": float(hyper_cfg.get("film_scale", 0.25)),
        "use_conditioned_embeddings": bool(
            hyper_cfg.get("use_conditioned_embeddings", bool(hyper_cfg.get("use_film", False)))
        ),
        "use_task_adapters": bool(hyper_cfg.get("use_task_adapters", False)),
        "adapter_reduction": int(hyper_cfg.get("adapter_reduction", 4)),
        "use_cross_stitch": bool(hyper_cfg.get("use_cross_stitch", False)),
        "cross_stitch_init": float(hyper_cfg.get("cross_stitch_init", 0.05)),
        "epochs": int(hyper_cfg.get("epochs", 60)),
        "batch_size": int(hyper_cfg.get("batch_size", 256)),
        "lr": float(hyper_cfg.get("lr", 1e-3)),
        "weight_decay": float(hyper_cfg.get("weight_decay", 1e-4)),
        "val_frac": float(hyper_cfg.get("val_frac", 0.10)),
        "patience": int(hyper_cfg.get("patience", 8)),
    }
    return "sheet_unified:" + config_sha1_from_raw(key_fields)


def _bundle_list(bundle: HyperTLBundle | list[HyperTLBundle] | None) -> list[HyperTLBundle]:
    if bundle is None:
        return []
    return list(bundle) if isinstance(bundle, list) else [bundle]


def _bundle_or_none(bundles: list[HyperTLBundle]) -> HyperTLBundle | list[HyperTLBundle] | None:
    if not bundles:
        return None
    return bundles[0] if len(bundles) == 1 else bundles


def _adapt_hyper_bundles(bundles: list[HyperTLBundle], hyper_cfg: dict[str, Any]) -> list[HyperTLBundle]:
    ridge_lambdas = [float(x) for x in hyper_cfg.get("ridge_lambdas", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])]
    ridge_lambda_b = float(hyper_cfg.get("ridge_lambda_b", 1e-2))
    return [
        HyperTLBundle(
            model=b.model,
            device=b.device,
            mol_mean=b.mol_mean,
            mol_std=b.mol_std,
            cp_mean=b.cp_mean,
            cp_std=b.cp_std,
            ridge_lambdas=ridge_lambdas,
            ridge_lambda_b=ridge_lambda_b,
            use_conditioned_embeddings=bool(getattr(b, "use_conditioned_embeddings", False)),
            task_to_index=getattr(b, "task_to_index", None),
            use_task_adapters=bool(getattr(b, "use_task_adapters", False)),
            aux_X_mol=getattr(b, "aux_X_mol", None),
            aux_X_cp=getattr(b, "aux_X_cp", None),
            aux_X_full=getattr(b, "aux_X_full", None),
            aux_y=getattr(b, "aux_y", None),
            aux_y_sec=getattr(b, "aux_y_sec", None),
            aux_dataset_ids=getattr(b, "aux_dataset_ids", None),
            aux_mol_keys=getattr(b, "aux_mol_keys", None),
        )
        for b in bundles
    ]


def _dataset_balanced_weights(dataset_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(dataset_ids, dtype=object).reshape(-1)
    counts: dict[str, int] = {}
    for raw in ids.tolist():
        key = str(raw).zfill(4)
        counts[key] = counts.get(key, 0) + 1
    base = np.asarray([1.0 / max(counts.get(str(raw).zfill(4), 1), 1) for raw in ids.tolist()], dtype=np.float32)
    return (base / max(float(np.mean(base)), 1e-12)).astype(np.float32)


def _normalize_sheet_conditioning(raw: Any) -> str:
    text = str(raw or "cp").strip().lower().replace("-", "_")
    if text in {"dataset_onehot", "task_onehot", "cp_dataset_onehot", "cp_task_onehot"}:
        return "dataset_onehot"
    return "cp"


def _dataset_onehot_index(dataset_ids: list[str]) -> dict[str, int]:
    ids = sorted({str(value).zfill(4) for value in dataset_ids})
    return {dataset_id: idx for idx, dataset_id in enumerate(ids)}


def _append_dataset_onehot(
    X_cp: np.ndarray,
    dataset_ids: np.ndarray,
    task_index: dict[str, int],
) -> np.ndarray:
    cp = np.asarray(X_cp, dtype=np.float32)
    if not task_index:
        return cp
    ids = np.asarray(dataset_ids, dtype=object).reshape(-1)
    if len(ids) != len(cp):
        raise ValueError(f"dataset_ids length mismatch for onehot CP: got={len(ids)}, expected={len(cp)}")
    onehot = np.zeros((len(cp), len(task_index)), dtype=np.float32)
    for row, raw_id in enumerate(ids.tolist()):
        col = task_index.get(str(raw_id).zfill(4))
        if col is not None:
            onehot[row, col] = 1.0
    return np.concatenate([cp, onehot], axis=1).astype(np.float32)


def _mean_cp_vector(mat: Any) -> np.ndarray:
    cp = np.asarray(getattr(mat, "X_cp", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    if cp.ndim != 2 or cp.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(np.mean(cp, axis=0), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _filter_pool_source_ids(
    *,
    mats: dict[str, Any],
    target_ds: str,
    candidate_ids: list[str],
    pool_cfg: dict[str, Any],
) -> list[str] | None:
    t0_tol = float(pool_cfg.get("source_t0_tolerance_sec", 0.0) or 0.0)
    min_cp_cosine = float(pool_cfg.get("source_min_cp_cosine", 0.0) or 0.0)
    top_k = int(pool_cfg.get("source_top_k_by_cp", 0) or 0)
    if t0_tol <= 0.0 and min_cp_cosine <= 0.0 and top_k <= 0:
        return None
    if target_ds not in mats:
        return None
    target_mat = mats[target_ds]
    target_t0 = float(getattr(target_mat, "t0_sec", 0.0) or 0.0)
    target_cp = _mean_cp_vector(target_mat)
    scored: list[tuple[str, float]] = []
    for raw_id in candidate_ids:
        source_id = str(raw_id).zfill(4)
        if source_id == target_ds or source_id not in mats:
            continue
        source_mat = mats[source_id]
        if t0_tol > 0.0:
            source_t0 = float(getattr(source_mat, "t0_sec", 0.0) or 0.0)
            if abs(source_t0 - target_t0) > t0_tol:
                continue
        cp_sim = _cosine_similarity(target_cp, _mean_cp_vector(source_mat))
        if min_cp_cosine > 0.0 and cp_sim < min_cp_cosine:
            continue
        scored.append((source_id, cp_sim))
    scored = sorted(scored, key=lambda item: (-item[1], item[0]))
    if top_k > 0:
        scored = scored[:top_k]
    if not scored and str(pool_cfg.get("source_filter_fallback", "all")).strip().lower() == "all":
        return None
    return [source_id for source_id, _ in scored]


def _merge_nested_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_per_dataset_model_override(
    model_cfg: dict[str, Any],
    resolved_cfg: dict[str, Any],
    mat: Any,
    stats: dict[str, float | int | str],
) -> dict[str, Any]:
    per_dataset = model_cfg.get("PER_DATASET_MODEL_OVERRIDES", {}) or {}
    if not isinstance(per_dataset, dict):
        return resolved_cfg
    dataset_id = str(getattr(mat, "dataset_id", "")).zfill(4)
    ds_override = per_dataset.get(dataset_id)
    if ds_override is None:
        for raw_key, value in per_dataset.items():
            if str(raw_key).zfill(4) == dataset_id:
                ds_override = value
                break
    if not isinstance(ds_override, dict):
        return resolved_cfg
    stats["dataset_override"] = dataset_id
    return _merge_nested_dict(resolved_cfg, ds_override)


def _dataset_outlier_rate(y_sec: np.ndarray) -> float:
    y = np.asarray(y_sec, dtype=np.float64).reshape(-1)
    if y.size < 4:
        return 0.0
    q1, q3 = np.percentile(y, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        return 0.0
    mask = (y < (q1 - 1.5 * iqr)) | (y > (q3 + 1.5 * iqr))
    return float(np.mean(mask))


def _duplicate_mol_key_rate(mol_keys: list[str] | np.ndarray | None) -> float:
    if mol_keys is None:
        return 0.0
    keys = [str(key).strip() for key in list(mol_keys)]
    keys = [key for key in keys if key and key not in {"nan", "NA", "None"}]
    if not keys:
        return 0.0
    return float(max(len(keys) - len(set(keys)), 0) / max(len(keys), 1))


def resolve_dataset_model_cfg(model_cfg: dict[str, Any], mat: Any) -> tuple[dict[str, Any], dict[str, float | int | str]]:
    cfg = copy.deepcopy(model_cfg)
    auto = dict(model_cfg.get("SINGLE_TASK_AUTO_POLICY", {}) or {})
    stats: dict[str, float | int | str] = {
        "selected_rule": "",
        "dataset_override": "",
        "n_rows": int(len(mat.y_sec)),
        "outlier_rate": _dataset_outlier_rate(mat.y_sec),
        "median_rt_sec": float(np.median(mat.y_sec)) if len(mat.y_sec) else 0.0,
        "t0_sec": float(getattr(mat, "t0_sec", 0.0) or 0.0),
        "duplicate_mol_key_rate": _duplicate_mol_key_rate(getattr(mat, "mol_keys", None)),
    }
    stats["median_to_t0_ratio"] = float(
        stats["median_rt_sec"] / max(float(stats["t0_sec"]), 1e-6) if float(stats["t0_sec"]) > 0.0 else 0.0
    )
    if not bool(auto.get("enabled", False)):
        cfg = _apply_per_dataset_model_override(model_cfg, cfg, mat, stats)
        if bool(cfg.get("ONLY_HYPER_TL", False)) and not bool(cfg.get("ENABLE_HYPER_TL", False)):
            cfg["ONLY_HYPER_TL"] = False
        return cfg, stats

    selected_rule = ""
    tiny_n_max = int(auto.get("tiny_n_max", 80))
    small_n_max = int(auto.get("small_n_max", 140))
    medium_n_max = int(auto.get("medium_n_max", 0))
    high_outlier_rate = float(auto.get("high_outlier_rate", 0.10))
    near_dead_time_ratio_max = float(auto.get("near_dead_time_ratio_max", 2.0))
    small_duplicate_mol_key_rate_min = float(auto.get("small_duplicate_mol_key_rate_min", 0.20))
    duplicate_mol_key_rate_min = float(auto.get("duplicate_mol_key_rate_min", 0.12))
    n_rows = int(stats["n_rows"])
    outlier_rate = float(stats["outlier_rate"])
    median_to_t0_ratio = float(stats["median_to_t0_ratio"])
    duplicate_mol_key_rate = float(stats["duplicate_mol_key_rate"])

    if n_rows <= tiny_n_max and median_to_t0_ratio <= near_dead_time_ratio_max and isinstance(auto.get("tiny_near_dead_time"), dict):
        selected_rule = "tiny_near_dead_time"
    elif (
        n_rows <= small_n_max
        and duplicate_mol_key_rate >= small_duplicate_mol_key_rate_min
        and isinstance(auto.get("small_duplicate_rich"), dict)
    ):
        selected_rule = "small_duplicate_rich"
    elif n_rows <= small_n_max and outlier_rate >= high_outlier_rate and isinstance(auto.get("small_outlier"), dict):
        selected_rule = "small_outlier"
    elif n_rows <= small_n_max and isinstance(auto.get("small"), dict):
        selected_rule = "small"
    elif (
        medium_n_max > 0
        and n_rows <= medium_n_max
        and duplicate_mol_key_rate >= duplicate_mol_key_rate_min
        and isinstance(auto.get("medium_duplicate_rich"), dict)
    ):
        selected_rule = "medium_duplicate_rich"
    elif medium_n_max > 0 and n_rows <= medium_n_max and isinstance(auto.get("medium"), dict):
        selected_rule = "medium"

    if selected_rule:
        cfg = _merge_nested_dict(cfg, dict(auto[selected_rule]))
    cfg = _apply_per_dataset_model_override(model_cfg, cfg, mat, stats)
    if bool(cfg.get("ONLY_HYPER_TL", False)) and not bool(cfg.get("ENABLE_HYPER_TL", False)):
        cfg["ONLY_HYPER_TL"] = False
    stats["selected_rule"] = selected_rule
    return cfg, stats


def _split_for_mat(mat: Any, seed: int, split_cfg: dict[str, Any]) -> Any:
    split_strategy = str(split_cfg.get("strategy", "stratified")).strip().lower()
    if split_strategy == "random":
        return random_split(
            y=mat.y_sec,
            seed=seed,
            train=float(split_cfg["train"]),
            val=float(split_cfg["val"]),
            test=float(split_cfg["test"]),
        )
    if split_strategy in ("kfold", "k-fold", "cv", "cross_validation", "cross-validation"):
        return kfold_split(
            y=mat.y_sec,
            seed=seed,
            n_splits=int(split_cfg.get("folds", split_cfg.get("n_splits", 10))),
            shuffle_seed=int(split_cfg.get("shuffle_seed", split_cfg.get("seed", 0))),
            val_fold_offset=int(split_cfg.get("val_fold_offset", 1)),
        )
    return stratified_split(
        y=mat.y_sec,
        seed=seed,
        train=float(split_cfg["train"]),
        val=float(split_cfg["val"]),
        test=float(split_cfg["test"]),
    )


def _transform_target_for_mat(mat: Any, target_transform: str) -> tuple[np.ndarray, float]:
    if target_transform == "gradient_norm":
        ds_scale = float(mat.y_scale_sec)
        y_used = (mat.y_sec / max(ds_scale, 1e-6)).astype(np.float32)
    elif target_transform == "logk":
        ds_scale = 1.0
        t0 = max(float(mat.t0_sec), 1e-6)
        y_used = np.log(np.clip((mat.y_sec - t0) / t0, 1e-6, None)).astype(np.float32)
    elif target_transform == "log1p":
        ds_scale = 1.0
        y_used = np.log1p(mat.y_sec).astype(np.float32)
    else:
        ds_scale = 1.0
        y_used = mat.y_sec.astype(np.float32, copy=False)
    return y_used, ds_scale


def _transform_source_targets(
    y_src_sec: np.ndarray,
    src_scales_per_row: np.ndarray,
    src_t0_per_row: np.ndarray,
    target_transform: str,
) -> np.ndarray:
    if target_transform == "gradient_norm":
        return (y_src_sec / np.maximum(src_scales_per_row, 1e-6)).astype(np.float32)
    if target_transform == "logk":
        t0 = np.maximum(src_t0_per_row, 1e-6)
        return np.log(np.clip((y_src_sec - t0) / t0, 1e-6, None)).astype(np.float32)
    if target_transform == "log1p":
        return np.log1p(y_src_sec).astype(np.float32)
    return y_src_sec.astype(np.float32, copy=False)


def run_trial(
    prep: PreparedBenchmark,
    cfg: Config,
    *,
    seeds: list[int],
    external_ids: list[str] | None = None,
    external_pool_ids: list[str] | None = None,
    config_sha1: str,
    resume_enabled: bool | None = None,
    write_predictions: bool = True,
    early_stop: bool = False,
) -> TrialResult:
    external_ids_used = [str(x).zfill(4) for x in (external_ids if external_ids is not None else prep.external_ids)]
    external_pool_ids_used = [
        str(x).zfill(4) for x in (external_pool_ids if external_pool_ids is not None else prep.external_ids)
    ]
    out_root = Path(cfg.outputs["root"])
    pred_root = out_root / "predictions"
    metrics_root = out_root / "metrics"
    ensure_dirs([out_root, pred_root, metrics_root])
    total_jobs = len(external_ids_used) * len(seeds)
    requested_keys = {(ds, int(seed)) for ds in external_ids_used for seed in seeds}

    # Resume safety: if user changes the config but reuses the same output root, don't silently mix results.
    sha_path = out_root / "config.sha1"
    resume_on = bool(cfg.outputs.get("resume", True)) if resume_enabled is None else bool(resume_enabled)
    if sha_path.exists():
        prev = sha_path.read_text(encoding="utf-8").strip()
        if prev and prev != config_sha1:
            resume_on = False
            sha_path.write_text(config_sha1, encoding="utf-8")
            logger.warning(
                "Config hash changed under an existing output root; disabling resume for this run.",
                extra={"run_dir": out_root.as_posix(), "previous_config_sha1": prev, "config_sha1": config_sha1},
            )
    else:
        sha_path.write_text(config_sha1, encoding="utf-8")

    existing_keys: set[tuple[str, int]] = set()
    existing_df = pd.DataFrame()
    per_seed_csv = metrics_root / "per_seed.csv"
    write_candidate_diagnostics = bool(cfg.models.get("WRITE_CANDIDATE_DIAGNOSTICS", False))
    candidate_diagnostics_csv = metrics_root / "candidate_diagnostics.csv"
    if write_candidate_diagnostics and not resume_on and candidate_diagnostics_csv.exists():
        candidate_diagnostics_csv.unlink()
    if resume_on and per_seed_csv.exists():
        try:
            existing_df = pd.read_csv(per_seed_csv, dtype={"dataset": str}, encoding="utf-8")
            existing_df["dataset"] = existing_df["dataset"].astype(str).str.zfill(4)
            if "seed" in existing_df.columns:
                existing_keys = set((r["dataset"], int(r["seed"])) for _, r in existing_df.iterrows())
        except Exception:
            existing_df = pd.DataFrame()
            existing_keys = set()
    per_seed_df = existing_df.copy() if not existing_df.empty else pd.DataFrame()
    resumed_jobs = len(existing_keys.intersection(requested_keys))
    completed_jobs = resumed_jobs
    logger.info(
        "Starting training loop.",
        extra={
            "run_dir": out_root.as_posix(),
            "dataset_count": len(external_ids_used),
            "seed_count": len(seeds),
            "job_count": total_jobs,
            "resume_enabled": resume_on,
            "resumed_job_count": resumed_jobs,
        },
    )
    if resumed_jobs:
        logger.info(
            "Loaded existing per-seed results for resume.",
            extra={"run_dir": out_root.as_posix(), "resumed_job_count": resumed_jobs, "per_seed_csv": per_seed_csv.as_posix()},
        )

    split_cfg = cfg.split
    source_weight = float(cfg.transfer_weights["source"])
    target_weight = float(cfg.transfer_weights["target"])
    per_dataset_multiplier = dict(cfg.transfer_weights.get("per_dataset_multiplier", {}) or {})
    adaptive_source = bool(cfg.transfer_weights.get("adaptive_source", True))
    similarity_power = float(cfg.transfer_weights.get("similarity_power", 2.0))
    min_scale = float(cfg.transfer_weights.get("min_scale", 0.25))
    max_scale = float(cfg.transfer_weights.get("max_scale", 2.0))
    source_weight_mode = str(cfg.transfer_weights.get("source_weight_mode", "per_sample")).strip().lower()
    if source_weight_mode not in ("per_sample", "per_dataset"):
        source_weight_mode = "per_sample"
    top_k_sources = cfg.transfer_weights.get("top_k_sources", None)
    try:
        top_k_sources = int(top_k_sources) if top_k_sources is not None else None
    except Exception:
        top_k_sources = None
    overlap_adaptive = bool(cfg.transfer_weights.get("overlap_adaptive_source", False))
    overlap_ref = float(cfg.transfer_weights.get("overlap_ref", 0.5))
    overlap_power = float(cfg.transfer_weights.get("overlap_power", 1.0))
    overlap_min_scale = float(cfg.transfer_weights.get("overlap_min_scale", 0.05))
    overlap_max_scale = float(cfg.transfer_weights.get("overlap_max_scale", 2.0))
    overlap_disable_threshold = float(cfg.transfer_weights.get("overlap_disable_threshold", 0.0))
    target_transform = normalize_target_transform(cfg.transfer_weights)

    # Determine which datasets to run in fail-tuning mode.
    fail_tuning_enabled = bool(cfg.models.get("ENABLE_FAIL_TUNING", True))
    previous_summary_path = metrics_root / "summary_vs_paper.csv"
    failed_override = [str(x).zfill(4) for x in cfg.datasets.get("failed_override", [])]
    if fail_tuning_enabled and failed_override:
        previous_failed = set([x for x in failed_override if x in set(external_ids_used)])
    elif fail_tuning_enabled and resume_on:
        previous_failed = load_previous_failed(previous_summary_path, external_ids_used)
    else:
        previous_failed = set()

    mats = prep.mats
    X_src = prep.X_src
    X_src_mol = prep.X_src_mol
    X_src_cp = prep.X_src_cp
    y_src_sec = prep.y_src_sec
    source_row_dataset_ids = prep.source_row_dataset_ids
    source_row_mol_keys = prep.source_row_mol_keys
    source_row_context_tokens = list(getattr(prep, "source_row_context_tokens", []) or [])
    if len(source_row_context_tokens) != len(source_row_mol_keys):
        source_row_context_tokens = [tuple()] * int(len(source_row_mol_keys))
    pretrain_ids = prep.pretrain_ids
    desc_dim = int(prep.schema.group_sizes.get("descriptor", 0))
    base_desc_dim = int(len(getattr(prep.schema, "descriptor_cols", []) or []))
    base_desc_dim = min(max(base_desc_dim, 0), desc_dim)
    maccs_dim = int(FINGERPRINT_SIZES.get("maccs", 166))
    fingerprint_dim = int(sum(FINGERPRINT_SIZES.values()))
    mol_text_dim = int(prep.schema.group_sizes.get("mol_text", 0))
    mol_text_start = int(desc_dim + fingerprint_dim)
    mol_seq_dim = int(prep.schema.group_sizes.get("mol_seq", 0))
    mol_seq_start = int(mol_text_start + mol_text_dim)

    def resolve_hyper_feature_cfg(raw_cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(raw_cfg)
        if bool(out.get("use_mol_sequence", False)) and mol_seq_dim > 0:
            out["mol_sequence_len"] = mol_seq_dim
            out["mol_sequence_vocab_size"] = int(getattr(prep.schema, "molecule_sequence_vocab_size", 0))
        else:
            out["mol_sequence_len"] = 0
            out["mol_sequence_vocab_size"] = 0
        return out

    def select_hyper_mol_features(X_mol_raw: np.ndarray, hyper_cfg: dict[str, Any]) -> np.ndarray:
        X_arr = np.asarray(X_mol_raw, dtype=np.float32)
        use_context = bool(hyper_cfg.get("use_mol_context", True))
        if not bool(hyper_cfg.get("use_mdl_subset_mol", False)):
            if not use_context and base_desc_dim < desc_dim:
                return np.concatenate([X_arr[:, :base_desc_dim], X_arr[:, desc_dim:]], axis=1).astype(np.float32)
            return X_arr
        desc_keep = desc_dim if use_context else base_desc_dim
        pieces = [X_arr[:, :desc_keep]]
        fp_stop = min(int(X_arr.shape[1]), int(desc_dim + maccs_dim))
        if fp_stop > desc_dim:
            pieces.append(X_arr[:, desc_dim:fp_stop])
        base = np.concatenate(pieces, axis=1).astype(np.float32)
        pieces = [base]
        if bool(hyper_cfg.get("use_mol_text_ngram", False)) and mol_text_dim > 0:
            stop = mol_text_start + mol_text_dim
            if X_arr.shape[1] >= stop:
                pieces.append(X_arr[:, mol_text_start:stop])
        if bool(hyper_cfg.get("use_mol_sequence", False)) and mol_seq_dim > 0:
            stop = mol_seq_start + mol_seq_dim
            if X_arr.shape[1] >= stop:
                pieces.append(X_arr[:, mol_seq_start:stop])
        if len(pieces) == 1:
            return base
        return np.concatenate(pieces, axis=1).astype(np.float32)

    exclude_target_from_source = bool(cfg.datasets.get("exclude_target_from_source", False))
    overlapping_eval_ids = set(external_ids_used).intersection(set(pretrain_ids))
    if exclude_target_from_source and bool(cfg.models.get("ENABLE_HYPER_TL", False)) and overlapping_eval_ids:
        raise ValueError(
            "datasets.exclude_target_from_source is incompatible with shared ENABLE_HYPER_TL when external datasets "
            "overlap pretrain datasets; disable ENABLE_HYPER_TL or use non-overlapping evaluation datasets."
        )

    # Precompute per-source-row scaling terms for target transforms.
    src_scales_per_row = np.array([float(mats[str(d)].y_scale_sec) for d in source_row_dataset_ids], dtype=np.float32)
    src_t0_per_row = np.array([float(mats[str(d)].t0_sec) for d in source_row_dataset_ids], dtype=np.float32)
    y_src_used_cache: dict[str, np.ndarray] = {}

    def source_y_for_transform(transform_name: str) -> np.ndarray:
        normalized = normalize_target_transform({"target_transform": transform_name})
        if normalized not in y_src_used_cache:
            y_src_used_cache[normalized] = _transform_source_targets(
                y_src_sec,
                src_scales_per_row,
                src_t0_per_row,
                normalized,
            )
        return y_src_used_cache[normalized]

    global_hyper_cfg = resolve_hyper_feature_cfg(dict(cfg.models.get("HYPER_TL", {}) or {}))
    hilic_hyper_runtime_cfg = dict(cfg.models.get("HILIC_HYPER_TL", {}) or {})
    hilic_hyper_enabled = bool(cfg.models.get("ENABLE_HILIC_HYPER_TL", False)) or bool(
        hilic_hyper_runtime_cfg.get("enabled", False)
    )
    hilic_hyper_cfg = resolve_hyper_feature_cfg(_merge_nested_dict(global_hyper_cfg, hilic_hyper_runtime_cfg))
    for runtime_key in ("enabled", "include_global_bundles", "replace_global_bundles", "max_target_n", "force_enable"):
        hilic_hyper_cfg.pop(runtime_key, None)

    hyper_use_mdl_subset = bool(global_hyper_cfg.get("use_mdl_subset_mol", False))
    if hilic_hyper_enabled and not bool(cfg.models.get("ENABLE_HYPER_TL", False)):
        hyper_use_mdl_subset = bool(hilic_hyper_cfg.get("use_mdl_subset_mol", False))
    if hilic_hyper_enabled:
        hilic_hyper_cfg["use_mdl_subset_mol"] = hyper_use_mdl_subset

    sheet_hyper_runtime_cfg = dict(cfg.models.get("SHEET_UNIFIED_HYPER_TL", {}) or {})
    sheet_hyper_enabled = bool(cfg.models.get("ENABLE_SHEET_UNIFIED_HYPER_TL", False)) or bool(
        sheet_hyper_runtime_cfg.get("enabled", False)
    )
    sheet_hyper_replace_source = bool(sheet_hyper_runtime_cfg.get("replace_source_bundles", True))
    sheet_hyper_include_source = bool(sheet_hyper_runtime_cfg.get("include_base_source", True))
    sheet_hyper_include_source_bundles = bool(sheet_hyper_runtime_cfg.get("include_source_bundles", False))
    sheet_hyper_conditioning = _normalize_sheet_conditioning(sheet_hyper_runtime_cfg.get("conditioning", "cp"))
    sheet_hyper_task_ids = (
        sorted(set((pretrain_ids if sheet_hyper_include_source else []) + external_pool_ids_used))
        if sheet_hyper_conditioning == "dataset_onehot"
        else []
    )
    sheet_hyper_task_index = _dataset_onehot_index(sheet_hyper_task_ids)
    sheet_hyper_cfg = resolve_hyper_feature_cfg(_merge_nested_dict(global_hyper_cfg, sheet_hyper_runtime_cfg))
    for runtime_key in (
        "enabled",
        "include_base_source",
        "include_source_bundles",
        "replace_source_bundles",
        "conditioning",
    ):
        sheet_hyper_cfg.pop(runtime_key, None)

    def global_hyper_bundle_for_transform(transform_name: str) -> HyperTLBundle | list[HyperTLBundle] | None:
        if not bool(cfg.models.get("ENABLE_HYPER_TL", False)):
            return None
        normalized = normalize_target_transform({"target_transform": transform_name})
        hyper_cfg = dict(global_hyper_cfg)
        X_src_mol_h = select_hyper_mol_features(X_src_mol, hyper_cfg)
        balance = bool(hyper_cfg.get("balance_pretrain_by_dataset", False))
        key = _hyper_cache_key(target_transform=normalized, hyper_cfg=hyper_cfg, balance_pretrain_by_dataset=balance)
        if key not in prep.hyper_cache:
            logger.info(
                "Pretraining Hyper TL source model(s).",
                extra={
                    "run_dir": out_root.as_posix(),
                    "target_transform": normalized,
                    "hyper_model_count": int(hyper_cfg.get("n_models", 1)),
                    "balance_pretrain_by_dataset": balance,
                },
            )
            # Dataset-balanced weights: each source dataset contributes equal total mass.
            sw = None
            if balance:
                ds_to_n = {ds: int(len(mats[ds].y_sec)) for ds in pretrain_ids}
                base = np.array(
                    [1.0 / max(ds_to_n.get(str(d), 1), 1) for d in source_row_dataset_ids],
                    dtype=np.float32,
                )
                sw = base / max(float(np.mean(base)), 1e-12)
            n_models = int(hyper_cfg.get("n_models", 1))
            bundles: list[HyperTLBundle] = []
            for i in range(max(1, n_models)):
                bundles.append(
                    pretrain_hyper_tl(
                        X_src_mol=X_src_mol_h,
                        X_src_cp=X_src_cp,
                        y_src=source_y_for_transform(normalized),
                        cfg=hyper_cfg,
                        seed=i,
                        sample_weights=sw,
                        dataset_ids=source_row_dataset_ids,
                    )
                )
            prep.hyper_cache[key] = bundles
        else:
            logger.info(
                "Reusing cached Hyper TL source model(s).",
                extra={"run_dir": out_root.as_posix(), "hyper_cache_key": key},
            )
        # Apply adaptation-only hyperparameters per trial.
        bundles_out = _adapt_hyper_bundles(prep.hyper_cache[key], hyper_cfg)
        return _bundle_or_none(bundles_out)

    hyper_bundle = None
    if not (sheet_hyper_enabled and sheet_hyper_replace_source):
        hyper_bundle = global_hyper_bundle_for_transform(target_transform)

    def sheet_unified_hyper_bundle_for_seed(
        transform_name: str,
        seed: int,
        source_bundle: HyperTLBundle | list[HyperTLBundle] | None,
    ) -> tuple[HyperTLBundle | list[HyperTLBundle] | None, int]:
        if not sheet_hyper_enabled:
            return source_bundle, 0
        normalized = normalize_target_transform({"target_transform": transform_name})
        hyper_cfg = dict(sheet_hyper_cfg)
        X_mol_parts: list[np.ndarray] = []
        X_cp_parts: list[np.ndarray] = []
        X_full_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        y_sec_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        key_parts: list[np.ndarray] = []

        if sheet_hyper_include_source:
            X_mol_parts.append(select_hyper_mol_features(X_src_mol, hyper_cfg))
            X_cp_parts.append(X_src_cp)
            X_full_parts.append(X_src)
            y_parts.append(source_y_for_transform(normalized))
            y_sec_parts.append(y_src_sec)
            id_parts.append(np.asarray(source_row_dataset_ids, dtype=object))
            key_parts.append(np.asarray(source_row_mol_keys, dtype=object))

        external_rows = 0
        for other_ds in external_pool_ids_used:
            other_ds = str(other_ds).zfill(4)
            if other_ds not in mats:
                continue
            other_mat = mats[other_ds]
            other_split = _split_for_mat(other_mat, int(seed), split_cfg)
            other_train_val_idx = np.concatenate(
                [np.asarray(other_split.train_idx, dtype=int), np.asarray(other_split.val_idx, dtype=int)],
                axis=0,
            )
            if other_train_val_idx.size == 0:
                continue
            other_y_used, _ = _transform_target_for_mat(other_mat, normalized)
            other_x_mol = other_mat.X_mol[other_train_val_idx]
            X_mol_parts.append(select_hyper_mol_features(other_x_mol, hyper_cfg))
            X_cp_parts.append(other_mat.X_cp[other_train_val_idx])
            X_full_parts.append(other_mat.X[other_train_val_idx])
            y_parts.append(other_y_used[other_train_val_idx])
            y_sec_parts.append(other_mat.y_sec[other_train_val_idx])
            id_parts.append(np.array([other_ds] * int(other_train_val_idx.size), dtype=object))
            key_parts.append(np.asarray(other_mat.mol_keys, dtype=object)[other_train_val_idx])
            external_rows += int(other_train_val_idx.size)

        if not y_parts or external_rows <= 0:
            return source_bundle, 0

        X_mol_all = np.concatenate(X_mol_parts, axis=0).astype(np.float32)
        X_cp_all = np.concatenate(X_cp_parts, axis=0).astype(np.float32)
        X_full_all = np.concatenate(X_full_parts, axis=0).astype(np.float32) if X_full_parts else np.empty((0, 0), dtype=np.float32)
        y_all = np.concatenate(y_parts, axis=0).astype(np.float32)
        y_sec_all = np.concatenate(y_sec_parts, axis=0).astype(np.float32)
        ids_all = np.concatenate(id_parts, axis=0)
        keys_all = np.concatenate(key_parts, axis=0) if key_parts else np.array([], dtype=object)
        if sheet_hyper_conditioning == "dataset_onehot":
            X_cp_all = _append_dataset_onehot(X_cp_all, ids_all, sheet_hyper_task_index)
        balance = bool(hyper_cfg.get("balance_pretrain_by_dataset", False))
        key = _sheet_unified_hyper_cache_key(
            target_transform=normalized,
            hyper_cfg=hyper_cfg,
            balance_pretrain_by_dataset=balance,
            seed=int(seed),
            include_base_source=sheet_hyper_include_source,
            external_ids=external_pool_ids_used,
            conditioning=sheet_hyper_conditioning,
            task_ids=sheet_hyper_task_ids,
            split_cfg=split_cfg,
            source_rows=int(len(source_row_dataset_ids)) if sheet_hyper_include_source else 0,
            external_rows=external_rows,
        )
        if key not in prep.hyper_cache:
            logger.info(
                "Pretraining sheet-unified Hyper TL model(s).",
                extra={
                    "run_dir": out_root.as_posix(),
                    "target_transform": normalized,
                    "seed": int(seed),
                    "source_row_count": int(len(source_row_dataset_ids)) if sheet_hyper_include_source else 0,
                    "external_train_val_row_count": int(external_rows),
                    "hyper_model_count": int(hyper_cfg.get("n_models", 1)),
                    "balance_pretrain_by_dataset": balance,
                },
            )
            sample_weights = _dataset_balanced_weights(ids_all) if balance else None
            n_models = int(hyper_cfg.get("n_models", 1))
            bundles: list[HyperTLBundle] = []
            for i in range(max(1, n_models)):
                bundle = pretrain_hyper_tl(
                    X_src_mol=X_mol_all,
                    X_src_cp=X_cp_all,
                    y_src=y_all,
                    cfg=hyper_cfg,
                    seed=int(seed) * 1000 + i,
                    sample_weights=sample_weights,
                    dataset_ids=ids_all,
                )
                bundle.aux_X_mol = X_mol_all
                bundle.aux_X_cp = X_cp_all
                bundle.aux_X_full = X_full_all
                bundle.aux_y = y_all
                bundle.aux_y_sec = y_sec_all
                bundle.aux_dataset_ids = ids_all
                bundle.aux_mol_keys = keys_all
                bundles.append(bundle)
            prep.hyper_cache[key] = bundles
        else:
            logger.info(
                "Reusing cached sheet-unified Hyper TL model(s).",
                extra={"run_dir": out_root.as_posix(), "hyper_cache_key": key, "seed": int(seed)},
            )
        combined_bundles: list[HyperTLBundle] = []
        if (
            sheet_hyper_conditioning != "dataset_onehot"
            and (not sheet_hyper_replace_source or sheet_hyper_include_source_bundles)
            and source_bundle is not None
        ):
            combined_bundles.extend(_bundle_list(source_bundle))
        combined_bundles.extend(_adapt_hyper_bundles(prep.hyper_cache[key], hyper_cfg))
        return _bundle_or_none(combined_bundles), int(external_rows)

    feature_group_store: dict[str, list[dict[str, float]]] = {ds: [] for ds in external_ids_used}
    auto_policy = dict(cfg.models.get("SINGLE_TASK_AUTO_POLICY", {}) or {})
    aux_small_n_max = int(auto_policy.get("small_n_max", 140))
    aux_max_n = int(cfg.models.get("EXTERNAL_AUX_MAX_N", aux_small_n_max))
    use_external_aux_source = bool(cfg.models.get("UNIFIED_MULTITASK", False)) and len(external_pool_ids_used) > 1
    hilic_pool_enabled = bool(cfg.models.get("ENABLE_HILIC_POOL_CANDIDATES", False))
    hilic_hyper_max_n = int(hilic_hyper_runtime_cfg.get("max_target_n", 0) or 0)
    if hilic_hyper_enabled:
        hilic_hyper_cfg["use_mdl_subset_mol"] = hyper_use_mdl_subset
    # Track which datasets have completed all requested seeds (for early stop bounds).
    done_ds: set[str] = set()
    early_stop_reason = ""

    for ds_index, ds in enumerate(external_ids_used, start=1):
        mat = mats[ds]
        target_loss_multiplier = expand_per_dataset_multiplier(ds, per_dataset_multiplier, default=1.0)
        target_weight_eff = float(target_weight * target_loss_multiplier)
        ds_model_cfg, ds_policy_stats = resolve_dataset_model_cfg(cfg.models, mat)
        ds_target_transform = normalize_target_transform(
            {"target_transform": ds_model_cfg.get("TARGET_TRANSFORM", target_transform)}
        )
        ds_hyper_bundle = (
            hyper_bundle
            if ds_target_transform == target_transform
            else global_hyper_bundle_for_transform(ds_target_transform)
        )
        ds_pred_dir = pred_root / ds
        if write_predictions:
            ds_pred_dir.mkdir(parents=True, exist_ok=True)
        fail_tune = fail_tuning_enabled and (ds in previous_failed)
        pending_seeds = [int(seed) for seed in seeds if (ds, int(seed)) not in existing_keys]
        logger.info(
            "[dataset %d/%d] %s pending_seeds=%d resumed_seeds=%d fail_tune=%s",
            ds_index,
            len(external_ids_used),
            ds,
            len(pending_seeds),
            len(seeds) - len(pending_seeds),
            "true" if fail_tune else "false",
            extra={
                "run_dir": out_root.as_posix(),
                "dataset": ds,
                "dataset_index": ds_index,
                "dataset_count": len(external_ids_used),
                "pending_seed_count": len(pending_seeds),
                "resumed_seed_count": len(seeds) - len(pending_seeds),
                "fail_tune": bool(fail_tune),
                "auto_policy_rule": str(ds_policy_stats.get("selected_rule", "")),
                "dataset_override": str(ds_policy_stats.get("dataset_override", "")),
                "n_rows": int(ds_policy_stats.get("n_rows", 0)),
                "outlier_rate": float(ds_policy_stats.get("outlier_rate", 0.0)),
                "duplicate_mol_key_rate": float(ds_policy_stats.get("duplicate_mol_key_rate", 0.0)),
                "median_to_t0_ratio": float(ds_policy_stats.get("median_to_t0_ratio", 0.0)),
                "target_loss_multiplier": float(target_loss_multiplier),
                "target_transform": ds_target_transform,
            },
        )
        if adaptive_source:
            src_sample_w = build_adaptive_source_weights(
                pretrain_ids=pretrain_ids,
                mats=mats,
                source_row_dataset_ids=source_row_dataset_ids,
                target_ds=ds,
                base_source_weight=source_weight,
                similarity_power=similarity_power,
                min_scale=min_scale,
                max_scale=max_scale,
                mode=source_weight_mode,
                top_k_sources=top_k_sources,
            )
        else:
            if source_weight_mode == "per_dataset":
                ds_to_n = {sid: int(len(mats[sid].y_sec)) for sid in pretrain_ids}
                src_sample_w = np.array(
                    [source_weight / max(ds_to_n.get(str(sid), 1), 1) for sid in source_row_dataset_ids],
                    dtype=np.float32,
                )
            else:
                src_sample_w = np.full(len(X_src), source_weight, dtype=np.float32)
        for seed in seeds:
            if (ds, int(seed)) in existing_keys:
                continue
            job_index = completed_jobs + 1
            logger.info(
                "[job %d/%d] dataset=%s seed=%d",
                job_index,
                total_jobs,
                ds,
                int(seed),
                extra={
                    "run_dir": out_root.as_posix(),
                    "dataset": ds,
                    "seed": int(seed),
                    "job_index": job_index,
                    "job_count": total_jobs,
                },
            )
            split = _split_for_mat(mat, int(seed), split_cfg)
            y_t_used, ds_scale = _transform_target_for_mat(mat, ds_target_transform)
            y_src_used = source_y_for_transform(ds_target_transform)

            source_keep = np.ones(len(source_row_dataset_ids), dtype=bool)
            if exclude_target_from_source:
                source_keep = np.asarray([str(source_ds) != ds for source_ds in source_row_dataset_ids], dtype=bool)

            overlap_rate = 0.0
            overlap_scale = 1.0
            src_sample_w_eff = src_sample_w
            if overlap_adaptive:
                train_keys = [str(mat.mol_keys[i]).strip() for i in split.train_idx]
                train_keys = [k for k in train_keys if k]
                if train_keys:
                    train_key_set = set(train_keys)
                    source_keys = {str(k) for k in source_row_mol_keys[source_keep].tolist() if str(k).strip()}
                    overlap_rate = float(len(train_key_set.intersection(source_keys)) / max(len(train_key_set), 1))
                ratio = overlap_rate / max(overlap_ref, 1e-6)
                overlap_scale = float(np.clip(np.power(max(ratio, 0.0), overlap_power), overlap_min_scale, overlap_max_scale))
                if overlap_rate < overlap_disable_threshold:
                    overlap_scale = 0.0
                src_sample_w_eff = (src_sample_w * overlap_scale).astype(np.float32)

            src_sample_w_train = src_sample_w_eff[source_keep] if src_sample_w_eff is not None else None
            X_src_train = X_src[source_keep]
            X_src_mol_train = X_src_mol[source_keep]
            X_src_cp_train = X_src_cp[source_keep]
            y_src_train = y_src_used[source_keep]
            y_src_sec_train = y_src_sec[source_keep]
            source_row_dataset_ids_train = source_row_dataset_ids[source_keep]
            source_row_mol_keys_train = source_row_mol_keys[source_keep]
            source_row_context_tokens_train = [
                source_row_context_tokens[i] for i, keep in enumerate(source_keep.tolist()) if bool(keep)
            ]
            external_aux_rows = 0
            hilic_hyper_rows = 0
            sheet_unified_hyper_rows = 0
            aux_X: list[np.ndarray] = []
            aux_X_mol: list[np.ndarray] = []
            aux_X_cp: list[np.ndarray] = []
            aux_y: list[np.ndarray] = []
            aux_y_sec: list[np.ndarray] = []
            aux_ids: list[np.ndarray] = []
            aux_keys: list[np.ndarray] = []
            aux_context_tokens: list[list[tuple[str, ...]]] = []
            exact_allowed_source_ids: list[str] = []
            pool_source_filter_ids = _filter_pool_source_ids(
                mats=mats,
                target_ds=ds,
                candidate_ids=[x for x in external_pool_ids_used if str(x).zfill(4) != ds],
                pool_cfg=dict(ds_model_cfg.get("HILIC_POOL", {}) or {}),
            )
            n_rows_for_policy = int(ds_policy_stats.get("n_rows", 0))
            attach_aux_to_source = (use_external_aux_source and n_rows_for_policy <= aux_max_n) or hilic_pool_enabled
            force_hilic_hyper = bool(hilic_hyper_runtime_cfg.get("force_enable", False))
            allow_hilic_hyper_for_dataset = hilic_hyper_enabled and (
                force_hilic_hyper or bool(ds_model_cfg.get("ENABLE_HYPER_TL", False))
            )
            build_aux_for_hilic = (
                allow_hilic_hyper_for_dataset
                and len(external_pool_ids_used) > 1
                and (hilic_hyper_max_n <= 0 or n_rows_for_policy <= hilic_hyper_max_n)
            )
            if attach_aux_to_source or build_aux_for_hilic:
                for other_ds in external_pool_ids_used:
                    other_ds = str(other_ds).zfill(4)
                    if other_ds == ds:
                        continue
                    if other_ds not in mats:
                        continue
                    other_mat = mats[other_ds]
                    other_split = _split_for_mat(other_mat, int(seed), split_cfg)
                    other_train_idx = np.concatenate(
                        [np.asarray(other_split.train_idx, dtype=int), np.asarray(other_split.val_idx, dtype=int)],
                        axis=0,
                    )
                    if other_train_idx.size == 0:
                        continue
                    other_y_used, _ = _transform_target_for_mat(other_mat, ds_target_transform)
                    aux_X.append(other_mat.X[other_train_idx])
                    aux_X_mol.append(other_mat.X_mol[other_train_idx])
                    aux_X_cp.append(other_mat.X_cp[other_train_idx])
                    aux_y.append(other_y_used[other_train_idx])
                    aux_y_sec.append(other_mat.y_sec[other_train_idx])
                    aux_ids.append(np.array([other_ds] * len(other_train_idx), dtype=object))
                    aux_keys.append(np.asarray(other_mat.mol_keys, dtype=object)[other_train_idx])
                    other_tokens = list(getattr(other_mat, "mol_context_tokens", []) or [])
                    if len(other_tokens) == len(other_mat.y_sec):
                        aux_context_tokens.append([other_tokens[int(i)] for i in other_train_idx])
                    else:
                        aux_context_tokens.append([tuple()] * int(len(other_train_idx)))
                    exact_allowed_source_ids.append(other_ds)
                if aux_X and attach_aux_to_source:
                    X_src_train = np.concatenate([X_src_train, *aux_X], axis=0)
                    X_src_mol_train = np.concatenate([X_src_mol_train, *aux_X_mol], axis=0)
                    X_src_cp_train = np.concatenate([X_src_cp_train, *aux_X_cp], axis=0)
                    y_src_train = np.concatenate([y_src_train, *aux_y], axis=0)
                    y_src_sec_train = np.concatenate([y_src_sec_train, *aux_y_sec], axis=0)
                    source_row_dataset_ids_train = np.concatenate([source_row_dataset_ids_train, *aux_ids], axis=0)
                    source_row_mol_keys_train = np.concatenate([source_row_mol_keys_train, *aux_keys], axis=0)
                    for token_part in aux_context_tokens:
                        source_row_context_tokens_train.extend(token_part)
                    aux_weight = np.full(sum(len(x) for x in aux_y), target_weight_eff, dtype=np.float32)
                    if src_sample_w_train is None:
                        base_weight = np.full(len(y_src_sec_train) - len(aux_weight), source_weight, dtype=np.float32)
                        src_sample_w_train = np.concatenate([base_weight, aux_weight], axis=0)
                    else:
                        src_sample_w_train = np.concatenate([src_sample_w_train, aux_weight], axis=0)
                    external_aux_rows = int(sum(len(x) for x in aux_y))
                    ds_model_cfg = copy.deepcopy(ds_model_cfg)
                    configured_exact_allowed = _id_set(ds_model_cfg.get("EXACT_MOL_ALLOWED_SOURCE_IDS"))
                    if configured_exact_allowed:
                        exact_allowed_source_ids = sorted(
                            set(exact_allowed_source_ids).intersection(configured_exact_allowed)
                        )
                    ds_model_cfg["ENABLE_EXACT_MOL_LOOKUP"] = True
                    ds_model_cfg["ENABLE_CALIBRATED_MOL_LOOKUP"] = True
                    ds_model_cfg["EXACT_MOL_ALLOWED_SOURCE_IDS"] = sorted(set(exact_allowed_source_ids))
                    ds_model_cfg["EXACT_MOL_EXCLUDED_SOURCE_IDS"] = [ds]
                    if hilic_pool_enabled:
                        if pool_source_filter_ids is None:
                            pool_allowed_ids = sorted(set(exact_allowed_source_ids))
                        else:
                            pool_allowed_ids = sorted(set(exact_allowed_source_ids).intersection(set(pool_source_filter_ids)))
                        if pool_allowed_ids:
                            ds_model_cfg["HILIC_POOL_ALLOWED_SOURCE_IDS"] = pool_allowed_ids
                            ds_model_cfg["HILIC_POOL_EXCLUDED_SOURCE_IDS"] = [ds]
                        else:
                            ds_model_cfg["ENABLE_HILIC_POOL_CANDIDATES"] = False

            job_hyper_bundle, sheet_unified_hyper_rows = sheet_unified_hyper_bundle_for_seed(
                ds_target_transform,
                int(seed),
                ds_hyper_bundle,
            )
            if build_aux_for_hilic and aux_X_mol:
                aux_X_mol_all = np.concatenate(aux_X_mol, axis=0).astype(np.float32)
                aux_X_cp_all = np.concatenate(aux_X_cp, axis=0).astype(np.float32)
                aux_y_all = np.concatenate(aux_y, axis=0).astype(np.float32)
                aux_ids_all = np.concatenate(aux_ids, axis=0)
                aux_source_ids = sorted(set(str(value).zfill(4) for value in aux_ids_all.tolist()))
                aux_X_mol_h = select_hyper_mol_features(aux_X_mol_all, hilic_hyper_cfg)
                balance_hilic = bool(hilic_hyper_cfg.get("balance_pretrain_by_dataset", True))
                hilic_key = _hilic_hyper_cache_key(
                    target_transform=ds_target_transform,
                    hyper_cfg=hilic_hyper_cfg,
                    balance_pretrain_by_dataset=balance_hilic,
                    target_ds=ds,
                    seed=int(seed),
                    source_ids=aux_source_ids,
                    source_rows=len(aux_y_all),
                )
                if hilic_key not in prep.hyper_cache:
                    logger.info(
                        "Pretraining HILIC external-source Hyper TL model(s).",
                        extra={
                            "run_dir": out_root.as_posix(),
                            "target_dataset": ds,
                            "seed": int(seed),
                            "source_dataset_count": len(aux_source_ids),
                            "source_row_count": int(len(aux_y_all)),
                            "balance_pretrain_by_dataset": balance_hilic,
                        },
                    )
                    sw_hilic = _dataset_balanced_weights(aux_ids_all) if balance_hilic else None
                    n_models_hilic = int(hilic_hyper_cfg.get("n_models", 1))
                    prep.hyper_cache[hilic_key] = [
                        pretrain_hyper_tl(
                            X_src_mol=aux_X_mol_h,
                            X_src_cp=aux_X_cp_all,
                            y_src=aux_y_all,
                            cfg=hilic_hyper_cfg,
                            seed=int(seed) * 100 + i,
                            sample_weights=sw_hilic,
                            dataset_ids=aux_ids_all,
                        )
                        for i in range(max(1, n_models_hilic))
                    ]
                include_global = bool(hilic_hyper_runtime_cfg.get("include_global_bundles", True))
                replace_global = bool(hilic_hyper_runtime_cfg.get("replace_global_bundles", False))
                combined_bundles: list[HyperTLBundle] = []
                if include_global and not replace_global:
                    combined_bundles.extend(_bundle_list(ds_hyper_bundle))
                combined_bundles.extend(_adapt_hyper_bundles(prep.hyper_cache[hilic_key], hilic_hyper_cfg))
                if combined_bundles:
                    job_hyper_bundle = _bundle_or_none(combined_bundles)
                    ds_model_cfg = copy.deepcopy(ds_model_cfg)
                    if force_hilic_hyper:
                        ds_model_cfg["ENABLE_HYPER_TL"] = True
                    if "ensemble_lambdas" in hilic_hyper_cfg:
                        ds_model_cfg.setdefault("HYPER_TL", {})
                        ds_model_cfg["HYPER_TL"]["ensemble_lambdas"] = bool(hilic_hyper_cfg.get("ensemble_lambdas", False))
                    if bool(hilic_hyper_runtime_cfg.get("enable_prior_calibration", True)):
                        ds_model_cfg["ENABLE_HYPER_PRIOR_CAL"] = True
                    hilic_hyper_rows = int(len(aux_y_all))

            X_src_cp_job = X_src_cp_train
            X_target_cp_job = mat.X_cp
            if sheet_hyper_enabled and sheet_hyper_conditioning == "dataset_onehot" and sheet_hyper_task_index:
                X_src_cp_job = _append_dataset_onehot(
                    X_src_cp_train,
                    source_row_dataset_ids_train,
                    sheet_hyper_task_index,
                )
                X_target_cp_job = _append_dataset_onehot(
                    mat.X_cp,
                    np.array([ds] * int(len(mat.y_sec)), dtype=object),
                    sheet_hyper_task_index,
                )

            out = train_and_ensemble(
                model_cfg=ds_model_cfg,
                X_src=X_src_train,
                X_src_mol=select_hyper_mol_features(X_src_mol_train, sheet_hyper_cfg if sheet_hyper_enabled else global_hyper_cfg),
                X_src_cp=X_src_cp_job,
                y_src=y_src_train,
                y_src_sec_raw=y_src_sec_train,
                X_target=mat.X,
                X_target_mol=select_hyper_mol_features(mat.X_mol, sheet_hyper_cfg if sheet_hyper_enabled else global_hyper_cfg),
                X_target_cp=X_target_cp_job,
                y_target=y_t_used,
                y_target_sec=mat.y_sec,
                split=split,
                seed=seed,
                source_weight=source_weight,
                target_weight=target_weight_eff,
                group_sizes=prep.schema.group_sizes,
                fail_tune=fail_tune,
                source_sample_weights=src_sample_w_train,
                target_transform=ds_target_transform,
                target_inv_scale=ds_scale,
                target_t0_sec=float(mat.t0_sec),
                hyper_bundle=job_hyper_bundle,
                source_row_dataset_ids=source_row_dataset_ids_train,
                source_mol_keys=source_row_mol_keys_train,
                source_context_tokens=source_row_context_tokens_train,
                target_mol_keys=np.asarray(mat.mol_keys, dtype=object),
                target_context_tokens=list(getattr(mat, "mol_context_tokens", []) or []),
                target_dataset_id=ds,
            )
            y_true = mat.y_sec[split.test_idx]
            y_pred = np.maximum(out.pred_test, 0.0)
            m = compute_metrics(y_true=y_true, y_pred=y_pred)
            feature_group_store[ds].append(out.feature_group_importance)
            if write_candidate_diagnostics and out.candidate_diagnostics:
                diag_rows = []
                for row in out.candidate_diagnostics:
                    diag_row = _candidate_diagnostic_csv_row(
                        dataset=ds,
                        seed=int(seed),
                        auto_policy_rule=str(ds_policy_stats.get("selected_rule", "")),
                        row=row,
                    )
                    diag_rows.append(diag_row)
                append_candidate_diagnostics_csv(diag_rows, candidate_diagnostics_csv)
            if write_predictions:
                pred_df = pd.DataFrame(
                    {
                        "id": [mat.ids[i] for i in split.test_idx],
                        "y_true_sec": y_true,
                        "y_pred_sec": y_pred,
                    }
                )
                pred_df.to_csv(ds_pred_dir / f"seed_{seed}.csv", index=False, encoding="utf-8")
            rec = {
                "dataset": ds,
                "seed": int(seed),
                "mae": m["mae"],
                "medae": m["medae"],
                "mre": m["mre"],
                "medre": m["medre"],
                "r2": m["r2"],
                "rmse": m["rmse"],
                "top_models": ",".join(out.top_models),
                "ensemble_weights": ",".join([f"{w:.6f}" for w in out.weights]),
                "fail_tuned": fail_tune,
                "overlap_rate": overlap_rate,
                "overlap_source_scale": overlap_scale,
                "target_loss_multiplier": target_loss_multiplier,
                "external_aux_rows": external_aux_rows,
                "hilic_hyper_rows": hilic_hyper_rows,
                "sheet_unified_hyper_rows": sheet_unified_hyper_rows,
            }
            per_seed_df = pd.concat([per_seed_df, pd.DataFrame([rec])], ignore_index=True)
            per_seed_df = write_per_seed_csv(per_seed_df=per_seed_df, out_path=per_seed_csv, external_ids=external_ids_used)
            existing_keys.add((ds, int(seed)))
            completed_jobs = job_index
            logger.info(
                "Completed job [%d/%d] dataset=%s seed=%d mae=%.4f r2=%.4f",
                completed_jobs,
                total_jobs,
                ds,
                int(seed),
                float(m["mae"]),
                float(m["r2"]),
                extra={
                    "run_dir": out_root.as_posix(),
                    "dataset": ds,
                    "seed": int(seed),
                    "job_index": completed_jobs,
                    "job_count": total_jobs,
                    "mae": float(m["mae"]),
                    "r2": float(m["r2"]),
                    "overlap_rate": float(overlap_rate),
                    "overlap_source_scale": float(overlap_scale),
                },
            )

        # Dataset completion bookkeeping (all requested seeds present).
        have = per_seed_df.loc[per_seed_df["dataset"].astype(str).str.zfill(4) == ds, "seed"].tolist()
        if set(int(s) for s in have) >= set(int(s) for s in seeds):
            done_ds.add(ds)
            ds_rows = per_seed_df.loc[per_seed_df["dataset"].astype(str).str.zfill(4) == ds].copy()
            logger.info(
                "Finished dataset %s [%d/%d] avg_mae=%.4f avg_r2=%.4f",
                ds,
                ds_index,
                len(external_ids_used),
                float(ds_rows["mae"].mean()),
                float(ds_rows["r2"].mean()),
                extra={
                    "run_dir": out_root.as_posix(),
                    "dataset": ds,
                    "dataset_index": ds_index,
                    "dataset_count": len(external_ids_used),
                    "dataset_avg_mae": float(ds_rows["mae"].mean()),
                    "dataset_avg_r2": float(ds_rows["r2"].mean()),
                },
            )

        if early_stop and len(done_ds) > 0:
            # Upper/lower bound check on avg metrics using best-case remaining datasets.
            ds_means = per_seed_df.groupby("dataset")[["mae", "r2"]].mean()
            ds_means = ds_means.loc[[d for d in ds_means.index.tolist() if d in set(done_ds)]]
            n_total = len(external_ids_used)
            n_done = int(ds_means.shape[0])
            if n_done > 0 and n_total > 0:
                sum_mae = float(ds_means["mae"].sum())
                sum_r2 = float(ds_means["r2"].sum())
                best_avg_mae = (sum_mae + 0.0 * (n_total - n_done)) / n_total
                best_avg_r2 = (sum_r2 + 1.0 * (n_total - n_done)) / n_total
                if best_avg_mae >= float(cfg.metrics["paper_avg_mae"]):
                    early_stop_reason = f"early_stop: avg_mae cannot beat paper (best_case={best_avg_mae:.4f})"
                if not early_stop_reason and best_avg_r2 <= float(cfg.metrics["paper_avg_r2"]):
                    early_stop_reason = f"early_stop: avg_r2 cannot beat paper (best_case={best_avg_r2:.4f})"
                if not early_stop_reason:
                    # NOTE: We intentionally do not early-stop based on win_both because the statistical
                    # tests are low-power for small seed counts (screening runs).
                    pass
                if early_stop_reason:
                    logger.warning(
                        "Early stopping triggered: %s",
                        early_stop_reason,
                        extra={
                            "run_dir": out_root.as_posix(),
                            "reason": early_stop_reason,
                            "completed_dataset_count": n_done,
                            "dataset_count": n_total,
                        },
                    )
                    break
    per_seed_df = write_per_seed_csv(per_seed_df=per_seed_df, out_path=per_seed_csv, external_ids=external_ids_used)

    baseline_df = prep.baseline_df.loc[prep.baseline_df["dataset"].isin(external_ids_used)].copy()
    summary_df = summarize_vs_paper(
        per_seed_df=per_seed_df,
        baseline_df=baseline_df,
        fdr_q=float(cfg.stats.get("fdr_q", 0.05)),
    )
    summary_df = summary_df[
        [
            "dataset",
            "paper_mae",
            "paper_r2",
            "our_mae_mean",
            "our_r2_mean",
            "delta_mae",
            "delta_r2",
            "p_mae",
            "p_r2",
            "p_adj_mae",
            "p_adj_r2",
            "win_both",
        ]
    ]
    summary_csv = metrics_root / "summary_vs_paper.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    ds_means_all = per_seed_df.groupby("dataset")[["mae", "r2"]].mean()
    avg_mae = float(ds_means_all["mae"].mean()) if not ds_means_all.empty else float("inf")
    avg_r2 = float(ds_means_all["r2"].mean()) if not ds_means_all.empty else float("-inf")
    wins = int(summary_df["win_both"].sum()) if not summary_df.empty else 0
    success = (
        (avg_mae < float(cfg.metrics["paper_avg_mae"]))
        and (avg_r2 > float(cfg.metrics["paper_avg_r2"]))
        and (wins >= int(cfg.metrics["required_win_both"]))
    )

    failed_datasets = summary_df.loc[~summary_df["win_both"], "dataset"].tolist() if not summary_df.empty else []
    fi_failed = {ds: aggregate_group_importance(feature_group_store.get(ds, [])) for ds in failed_datasets}
    write_report(
        out_path=out_root / "report.md",
        per_seed=per_seed_df,
        summary=summary_df,
        baseline_avg_mae=float(cfg.metrics["paper_avg_mae"]),
        baseline_avg_r2=float(cfg.metrics["paper_avg_r2"]),
        success_win_required=int(cfg.metrics["required_win_both"]),
        feature_importance_failed=fi_failed,
    )
    if str(cfg.metrics.get("report_style", "")).strip().lower() == "unirt" or cfg.metrics.get("unirt_baseline_csv"):
        unirt_baseline_csv = Path(str(cfg.metrics.get("unirt_baseline_csv", cfg.data["baseline_csv"])))
        write_unirt_report(
            out_path=out_root / "report_vs_unirt.md",
            per_seed=per_seed_df,
            baseline_csv=unirt_baseline_csv,
            mode=str(cfg.metrics.get("unirt_mode", "RPLC")),
            ours_label=str(cfg.metrics.get("ours_label", "Ours")),
            n_model=int(cfg.metrics.get("ours_n_model", len(external_ids_used))),
            expected_seeds=[int(seed) for seed in seeds],
            split_cfg=split_cfg,
            stats_test=str(cfg.stats.get("test", "wilcoxon_signed_rank")),
            fdr_correction=str(cfg.stats.get("correction", "bh_fdr")),
            output_dir=metrics_root,
        )
    logger.info(
        "Trial summary computed.",
        extra={
            "run_dir": out_root.as_posix(),
            "summary_csv": summary_csv.as_posix(),
            "report_path": (out_root / "report.md").as_posix(),
            "avg_mae": float(avg_mae),
            "avg_r2": float(avg_r2),
            "win_both": int(wins),
            "success": bool(success),
            "early_stop_reason": early_stop_reason,
        },
    )

    return TrialResult(
        out_root=out_root,
        per_seed_df=per_seed_df,
        summary_df=summary_df,
        avg_mae=avg_mae,
        avg_r2=avg_r2,
        wins=wins,
        success=success,
        early_stop_reason=early_stop_reason,
    )
