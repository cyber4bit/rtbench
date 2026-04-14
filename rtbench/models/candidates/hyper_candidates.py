from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from ...hyper import head_prior, mol_embeddings, ridge_prior_fit_predict
from ...metrics import compute_metrics
from ..calibration import optimize_weights
from ..trees import CandidateOutput, _inverse_target
from .common import CandidateBuildContext


def build_hyper_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if ctx.hyper_bundle is None or not bool(ctx.model_cfg.get("ENABLE_HYPER_TL", False)):
        return []

    bundles = ctx.hyper_bundle if isinstance(ctx.hyper_bundle, list) else [ctx.hyper_bundle]
    hyper_cfg = dict(ctx.model_cfg.get("HYPER_TL", {}) or {})
    ensemble_lambdas = bool(hyper_cfg.get("ensemble_lambdas", False))
    candidates: list[CandidateOutput] = []
    ensemble_val: list[np.ndarray] = []
    ensemble_test: list[np.ndarray] = []

    for bundle in bundles:
        Z_tr = mol_embeddings(bundle, ctx.X_train_mol)
        Z_va = mol_embeddings(bundle, ctx.X_val_mol)
        Z_te = mol_embeddings(bundle, ctx.X_test_mol)
        w0, b0 = head_prior(bundle, ctx.X_train_cp[0])
        Z_tv = np.concatenate([Z_tr, Z_va], axis=0)
        y_tv = np.concatenate([ctx.y_train, ctx.y_val_used], axis=0)

        if ensemble_lambdas and len(bundle.ridge_lambdas) > 1:
            val_preds = [
                _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tr,
                        y_train=ctx.y_train,
                        Z_eval=Z_va,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                for lam in bundle.ridge_lambdas
            ]
            val_mat = np.column_stack(val_preds)
            w_lam = optimize_weights(ctx.y_val_sec, val_mat)
            val_pred = (val_mat @ w_lam).astype(np.float32)

            test_preds = [
                _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tv,
                        y_train=y_tv,
                        Z_eval=Z_te,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                for lam in bundle.ridge_lambdas
            ]
            test_pred = (np.column_stack(test_preds) @ w_lam).astype(np.float32)
        else:
            best_lambda = None
            best_mae = float("inf")
            best_val = None
            for lam in bundle.ridge_lambdas:
                val_pred = _inverse_target(
                    ridge_prior_fit_predict(
                        Z_train=Z_tr,
                        y_train=ctx.y_train,
                        Z_eval=Z_va,
                        w0=w0,
                        b0=b0,
                        lam=float(lam),
                        lam_b=float(bundle.ridge_lambda_b),
                    ),
                    ctx.target_transform,
                    ctx.target_inv_scale,
                    ctx.target_t0_sec,
                )
                mae = float(compute_metrics(ctx.y_val_sec, val_pred)["mae"])
                if mae < best_mae:
                    best_mae = mae
                    best_lambda = float(lam)
                    best_val = val_pred
            if best_lambda is None or best_val is None:
                continue
            val_pred = best_val
            test_pred = _inverse_target(
                ridge_prior_fit_predict(
                    Z_train=Z_tv,
                    y_train=y_tv,
                    Z_eval=Z_te,
                    w0=w0,
                    b0=b0,
                    lam=best_lambda,
                    lam_b=float(bundle.ridge_lambda_b),
                ),
                ctx.target_transform,
                ctx.target_inv_scale,
                ctx.target_t0_sec,
            )

        ensemble_val.append(val_pred)
        ensemble_test.append(test_pred)

    if ensemble_val and ensemble_test:
        val_pred = np.median(np.stack(ensemble_val, axis=0), axis=0).astype(np.float32)
        test_pred = np.median(np.stack(ensemble_test, axis=0), axis=0).astype(np.float32)
        candidates.append(
            CandidateOutput(
                name=f"HYPER_TL_ENS(n={len(ensemble_val)},lam_ens={ensemble_lambdas})",
                val_pred=val_pred,
                test_pred=test_pred,
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model={"n": len(ensemble_val)},
            )
        )

    if bool(ctx.model_cfg.get("ENABLE_HYPER_EMB_LGBM", False)) and bundles:
        candidates.append(_build_hyper_emb_lgbm(ctx, bundles[0]))
    return candidates


def _build_hyper_emb_lgbm(ctx: CandidateBuildContext, bundle: Any) -> CandidateOutput:
    Z_tr = mol_embeddings(bundle, ctx.X_train_mol)
    Z_va = mol_embeddings(bundle, ctx.X_val_mol)
    Z_te = mol_embeddings(bundle, ctx.X_test_mol)
    lgbm_params = ctx.model_cfg.get(
        "HYPER_EMB_LGBM",
        {
            "n_estimators": 2000,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "objective": "mae",
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
        },
    )
    stop_rounds = int(ctx.model_cfg.get("HYPER_EMB_EARLY_STOPPING_ROUNDS", 50))
    z_cols = [f"z{i}" for i in range(Z_tr.shape[1])]
    Z_tr_df = pd.DataFrame(Z_tr, columns=z_cols)
    Z_va_df = pd.DataFrame(Z_va, columns=z_cols)
    Z_te_df = pd.DataFrame(Z_te, columns=z_cols)
    model = LGBMRegressor(random_state=ctx.seed + 7000, n_jobs=8, verbose=-1, **lgbm_params)
    callbacks = [early_stopping(stopping_rounds=stop_rounds, verbose=False)] if stop_rounds > 0 else None
    model.fit(
        Z_tr_df,
        ctx.y_train,
        eval_set=[(Z_va_df, ctx.y_val_used)],
        eval_metric="l1",
        callbacks=callbacks,
    )
    val_pred = _inverse_target(
        model.predict(Z_va_df),
        ctx.target_transform,
        ctx.target_inv_scale,
        ctx.target_t0_sec,
    )
    test_pred = _inverse_target(
        model.predict(Z_te_df),
        ctx.target_transform,
        ctx.target_inv_scale,
        ctx.target_t0_sec,
    )
    return CandidateOutput(
        name="HYPER_EMB_LGBM",
        val_pred=val_pred,
        test_pred=test_pred,
        val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
        model=model,
    )
