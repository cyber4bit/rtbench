from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from ..hyper import HyperTLBundle, _standardize, pretrain_hyper_tl
from .unified_cv import FitPredictFn, UnifiedCVFold, UnifiedCVFoldPrediction


DEFAULT_UNIFIED_HYPER_TL_CFG: dict[str, Any] = {
    "embed_dim": 64,
    "mol_hidden": 256,
    "cp_hidden": 128,
    "dropout": 0.10,
    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 8,
    "use_film": True,
    "film_scale": 0.25,
    "use_conditioned_embeddings": True,
    "use_task_adapters": False,
    "use_cross_stitch": False,
    "ranking_loss_weight": 0.0,
}


def build_unified_hypertl_fit_predict(
    model_cfg: Mapping[str, Any] | None = None,
    *,
    seed: int = 0,
) -> FitPredictFn:
    """Return a strict nModel=1 CP-conditioned HyperTL pooled learner.

    The returned callback trains exactly one HyperTL bundle for each
    `UnifiedCVFold`. It passes explicit train/validation indices into
    `pretrain_hyper_tl`, so early stopping and model selection are restricted to
    the pooled train/validation rows assembled by strict unified CV.
    """

    cfg = unified_hypertl_config(model_cfg)
    base_seed = int(seed)

    def fit_predict(fold: UnifiedCVFold) -> UnifiedCVFoldPrediction:
        train_val = _train_val_pool(fold)
        train_idx = np.arange(len(fold.y_train_sec), dtype=int)
        val_idx = np.arange(len(fold.y_train_sec), len(train_val["y"]), dtype=int)

        bundle = pretrain_hyper_tl(
            train_val["X_mol"],
            train_val["X_cp"],
            train_val["y"],
            cfg=cfg,
            seed=base_seed + int(fold.fold_id),
            train_idx=train_idx,
            val_idx=val_idx,
        )
        val_pred = predict_hypertl_bundle(bundle, fold.X_val_mol, fold.X_val_cp, batch_size=int(cfg["batch_size"]))
        test_pred = predict_hypertl_bundle(bundle, fold.X_test_mol, fold.X_test_cp, batch_size=int(cfg["batch_size"]))
        return UnifiedCVFoldPrediction(
            fold_id=fold.fold_id,
            val_pred_sec=val_pred,
            test_pred_sec=test_pred,
            val_meta=fold.val_meta,
            test_meta=fold.test_meta,
            model_info={
                "learner": "unified_hypertl",
                "n_model": 1,
                "train_rows": int(len(fold.y_train_sec)),
                "validation_rows": int(len(fold.y_val_sec)),
                "test_rows": int(len(fold.y_test_sec)),
                "train_val_rows": int(len(train_val["y"])),
                "use_film": bool(cfg.get("use_film", False)),
                "use_task_adapters": bool(cfg.get("use_task_adapters", False)),
            },
        )

    return fit_predict


def unified_hypertl_config(model_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_UNIFIED_HYPER_TL_CFG)
    model_cfg = dict(model_cfg or {})
    for key in ("UNIFIED_CV_HYPER_TL", "STRICT_UNIFIED_HYPER_TL", "SHEET_UNIFIED_HYPER_TL", "HYPER_TL"):
        raw = model_cfg.get(key)
        if isinstance(raw, Mapping):
            cfg.update(copy.deepcopy(dict(raw)))

    # Strict unified CV reports nModel=1. Ignore ensemble counts carried over
    # from non-unified runner configs and keep conditioning CP-only.
    cfg["n_models"] = 1
    cfg["use_task_adapters"] = False
    cfg["use_cross_stitch"] = False
    cfg["use_film"] = bool(cfg.get("use_film", True))
    cfg["use_conditioned_embeddings"] = bool(cfg.get("use_conditioned_embeddings", cfg["use_film"]))
    cfg["batch_size"] = int(max(1, int(cfg.get("batch_size", 256))))
    return cfg


def predict_hypertl_bundle(
    bundle: HyperTLBundle,
    X_mol: np.ndarray,
    X_cp: np.ndarray,
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    X_mol = np.asarray(X_mol, dtype=np.float32)
    X_cp = np.asarray(X_cp, dtype=np.float32)
    if X_mol.shape[0] != X_cp.shape[0]:
        raise ValueError(f"prediction row count mismatch: X_mol={X_mol.shape[0]}, X_cp={X_cp.shape[0]}")
    if len(X_mol) == 0:
        return np.asarray([], dtype=np.float32)

    Xm = _standardize(X_mol, bundle.mol_mean, bundle.mol_std)
    Xc = _standardize(X_cp, bundle.cp_mean, bundle.cp_std)
    batch_size = int(max(1, batch_size))
    out: list[np.ndarray] = []
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(Xm), batch_size):
            end = min(start + batch_size, len(Xm))
            mol = torch.from_numpy(Xm[start:end]).to(bundle.device)
            cp = torch.from_numpy(Xc[start:end]).to(bundle.device)
            pred = bundle.model(mol, cp)
            out.append(pred.detach().cpu().numpy().astype(np.float32).reshape(-1))
    return np.concatenate(out).astype(np.float32, copy=False)


def _train_val_pool(fold: UnifiedCVFold) -> dict[str, np.ndarray]:
    return {
        "X_mol": np.concatenate(
            [np.asarray(fold.X_train_mol, dtype=np.float32), np.asarray(fold.X_val_mol, dtype=np.float32)],
            axis=0,
        ),
        "X_cp": np.concatenate(
            [np.asarray(fold.X_train_cp, dtype=np.float32), np.asarray(fold.X_val_cp, dtype=np.float32)],
            axis=0,
        ),
        "y": np.concatenate(
            [np.asarray(fold.y_train_sec, dtype=np.float32), np.asarray(fold.y_val_sec, dtype=np.float32)],
            axis=0,
        ),
    }
