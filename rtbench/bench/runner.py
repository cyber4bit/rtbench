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
from ..models import random_split, stratified_split, train_and_ensemble
from ..report import write_report
from ..stats import summarize_vs_paper
from .prepare import PreparedBenchmark, config_sha1_from_raw, ensure_dirs
from .weighting import build_adaptive_source_weights, normalize_target_transform


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
        "dropout": float(hyper_cfg.get("dropout", 0.10)),
        "epochs": int(hyper_cfg.get("epochs", 60)),
        "batch_size": int(hyper_cfg.get("batch_size", 256)),
        "lr": float(hyper_cfg.get("lr", 1e-3)),
        "weight_decay": float(hyper_cfg.get("weight_decay", 1e-4)),
        "val_frac": float(hyper_cfg.get("val_frac", 0.10)),
        "patience": int(hyper_cfg.get("patience", 8)),
    }
    return config_sha1_from_raw(key_fields)


def _merge_nested_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
    stats["selected_rule"] = selected_rule
    return cfg, stats


def run_trial(
    prep: PreparedBenchmark,
    cfg: Config,
    *,
    seeds: list[int],
    external_ids: list[str] | None = None,
    config_sha1: str,
    resume_enabled: bool | None = None,
    write_predictions: bool = True,
    early_stop: bool = False,
) -> TrialResult:
    external_ids_used = [str(x).zfill(4) for x in (external_ids if external_ids is not None else prep.external_ids)]
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
    else:
        previous_failed = load_previous_failed(previous_summary_path, external_ids_used) if fail_tuning_enabled else set()

    mats = prep.mats
    X_src = prep.X_src
    X_src_mol = prep.X_src_mol
    X_src_cp = prep.X_src_cp
    y_src_sec = prep.y_src_sec
    source_row_dataset_ids = prep.source_row_dataset_ids
    source_row_mol_keys = prep.source_row_mol_keys
    pretrain_ids = prep.pretrain_ids
    desc_dim = int(prep.schema.group_sizes.get("descriptor", 0))
    maccs_dim = int(FINGERPRINT_SIZES.get("maccs", 166))
    mol_mdl_keep = min(int(X_src_mol.shape[1]), int(desc_dim + maccs_dim))

    # Precompute per-source-row scaling terms for target transforms.
    src_scales_per_row = np.array([float(mats[str(d)].y_scale_sec) for d in source_row_dataset_ids], dtype=np.float32)
    src_t0_per_row = np.array([float(mats[str(d)].t0_sec) for d in source_row_dataset_ids], dtype=np.float32)
    if target_transform == "gradient_norm":
        y_src_used_base = (y_src_sec / np.maximum(src_scales_per_row, 1e-6)).astype(np.float32)
    elif target_transform == "logk":
        t0 = np.maximum(src_t0_per_row, 1e-6)
        y_src_used_base = np.log(np.clip((y_src_sec - t0) / t0, 1e-6, None)).astype(np.float32)
    elif target_transform == "log1p":
        y_src_used_base = np.log1p(y_src_sec).astype(np.float32)
    else:
        y_src_used_base = y_src_sec.astype(np.float32, copy=False)

    hyper_bundle: HyperTLBundle | list[HyperTLBundle] | None = None
    hyper_use_mdl_subset = False
    if bool(cfg.models.get("ENABLE_HYPER_TL", False)):
        hyper_cfg = dict(cfg.models.get("HYPER_TL", {}))
        hyper_use_mdl_subset = bool(hyper_cfg.get("use_mdl_subset_mol", False))
        X_src_mol_h = X_src_mol[:, :mol_mdl_keep] if hyper_use_mdl_subset else X_src_mol
        balance = bool(hyper_cfg.get("balance_pretrain_by_dataset", False))
        key = _hyper_cache_key(target_transform=target_transform, hyper_cfg=hyper_cfg, balance_pretrain_by_dataset=balance)
        if key not in prep.hyper_cache:
            logger.info(
                "Pretraining Hyper TL source model(s).",
                extra={
                    "run_dir": out_root.as_posix(),
                    "target_transform": target_transform,
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
                        y_src=y_src_used_base,
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
        ridge_lambdas = [float(x) for x in hyper_cfg.get("ridge_lambdas", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])]
        ridge_lambda_b = float(hyper_cfg.get("ridge_lambda_b", 1e-2))
        bundles_out: list[HyperTLBundle] = []
        for b in prep.hyper_cache[key]:
            bundles_out.append(
                HyperTLBundle(
                    model=b.model,
                    device=b.device,
                    mol_mean=b.mol_mean,
                    mol_std=b.mol_std,
                    cp_mean=b.cp_mean,
                    cp_std=b.cp_std,
                    ridge_lambdas=ridge_lambdas,
                    ridge_lambda_b=ridge_lambda_b,
                )
            )
        hyper_bundle = bundles_out[0] if len(bundles_out) == 1 else bundles_out

    feature_group_store: dict[str, list[dict[str, float]]] = {ds: [] for ds in external_ids_used}
    # Track which datasets have completed all requested seeds (for early stop bounds).
    done_ds: set[str] = set()
    early_stop_reason = ""

    for ds_index, ds in enumerate(external_ids_used, start=1):
        mat = mats[ds]
        ds_model_cfg, ds_policy_stats = resolve_dataset_model_cfg(cfg.models, mat)
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
                "n_rows": int(ds_policy_stats.get("n_rows", 0)),
                "outlier_rate": float(ds_policy_stats.get("outlier_rate", 0.0)),
                "duplicate_mol_key_rate": float(ds_policy_stats.get("duplicate_mol_key_rate", 0.0)),
                "median_to_t0_ratio": float(ds_policy_stats.get("median_to_t0_ratio", 0.0)),
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
            split_strategy = str(split_cfg.get("strategy", "stratified")).strip().lower()
            if split_strategy == "random":
                split = random_split(
                    y=mat.y_sec,
                    seed=seed,
                    train=float(split_cfg["train"]),
                    val=float(split_cfg["val"]),
                    test=float(split_cfg["test"]),
                )
            else:
                split = stratified_split(
                    y=mat.y_sec,
                    seed=seed,
                    train=float(split_cfg["train"]),
                    val=float(split_cfg["val"]),
                    test=float(split_cfg["test"]),
                )
            if target_transform == "gradient_norm":
                ds_scale = float(mat.y_scale_sec)
                y_src_used = y_src_used_base
                y_t_used = (mat.y_sec / max(ds_scale, 1e-6)).astype(np.float32)
            elif target_transform == "logk":
                ds_scale = 1.0
                t0 = max(float(mat.t0_sec), 1e-6)
                y_src_used = y_src_used_base
                y_t_used = np.log(np.clip((mat.y_sec - t0) / t0, 1e-6, None)).astype(np.float32)
            elif target_transform == "log1p":
                ds_scale = 1.0
                y_src_used = y_src_used_base
                y_t_used = np.log1p(mat.y_sec).astype(np.float32)
            else:
                ds_scale = 1.0
                y_src_used = y_src_used_base
                y_t_used = mat.y_sec.astype(np.float32, copy=False)

            overlap_rate = 0.0
            overlap_scale = 1.0
            src_sample_w_eff = src_sample_w
            if overlap_adaptive:
                train_keys = [str(mat.mol_keys[i]).strip() for i in split.train_idx]
                train_keys = [k for k in train_keys if k]
                if train_keys:
                    train_key_set = set(train_keys)
                    overlap_rate = float(len(train_key_set.intersection(prep.source_mol_key_set)) / max(len(train_key_set), 1))
                ratio = overlap_rate / max(overlap_ref, 1e-6)
                overlap_scale = float(np.clip(np.power(max(ratio, 0.0), overlap_power), overlap_min_scale, overlap_max_scale))
                if overlap_rate < overlap_disable_threshold:
                    overlap_scale = 0.0
                src_sample_w_eff = (src_sample_w * overlap_scale).astype(np.float32)

            out = train_and_ensemble(
                model_cfg=ds_model_cfg,
                X_src=X_src,
                X_src_mol=(X_src_mol[:, :mol_mdl_keep] if hyper_use_mdl_subset else X_src_mol),
                X_src_cp=X_src_cp,
                y_src=y_src_used,
                y_src_sec_raw=y_src_sec,
                X_target=mat.X,
                X_target_mol=(mat.X_mol[:, :mol_mdl_keep] if hyper_use_mdl_subset else mat.X_mol),
                X_target_cp=mat.X_cp,
                y_target=y_t_used,
                y_target_sec=mat.y_sec,
                split=split,
                seed=seed,
                source_weight=source_weight,
                target_weight=target_weight,
                group_sizes=prep.schema.group_sizes,
                fail_tune=fail_tune,
                source_sample_weights=src_sample_w_eff,
                target_transform=target_transform,
                target_inv_scale=ds_scale,
                target_t0_sec=float(mat.t0_sec),
                hyper_bundle=hyper_bundle,
                source_row_dataset_ids=source_row_dataset_ids,
                source_mol_keys=source_row_mol_keys,
                target_mol_keys=np.asarray(mat.mol_keys, dtype=object),
            )
            y_true = mat.y_sec[split.test_idx]
            y_pred = np.maximum(out.pred_test, 0.0)
            m = compute_metrics(y_true=y_true, y_pred=y_pred)
            feature_group_store[ds].append(out.feature_group_importance)
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
