from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

from ...metrics import compute_metrics
from ..trees import CandidateOutput, _inverse_target
from .common import CandidateBuildContext


def build_local_fast_candidates(ctx: CandidateBuildContext) -> list[CandidateOutput]:
    if not bool(ctx.model_cfg.get("ENABLE_LOCAL_FAST_CANDIDATES", False)):
        return []
    if len(ctx.y_train) < int(ctx.model_cfg.get("LOCAL_FAST_MIN_TRAIN_ROWS", 24)):
        return []

    cfg = dict(ctx.model_cfg.get("LOCAL_FAST", {}) or {})
    X_tv = np.concatenate([ctx.X_train, ctx.X_val], axis=0)
    y_tv = np.concatenate([ctx.y_train, ctx.y_val_used], axis=0)
    candidates: list[CandidateOutput] = []

    if bool(cfg.get("enable_extra_trees", True)):
        params = dict(
            cfg.get(
                "extra_trees",
                {
                    "n_estimators": 512,
                    "max_features": 0.6,
                    "min_samples_leaf": 1,
                    "bootstrap": False,
                },
            )
        )
        model = ExtraTreesRegressor(random_state=ctx.seed + 6100, n_jobs=8, **params)
        model.fit(ctx.X_train, ctx.y_train)
        val_pred = _inverse_target(model.predict(ctx.X_val), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        final_model = ExtraTreesRegressor(random_state=ctx.seed + 6100, n_jobs=8, **params)
        final_model.fit(X_tv, y_tv)
        candidates.append(
            CandidateOutput(
                name="LOCAL_FAST_ET",
                val_pred=val_pred,
                test_pred=_inverse_target(final_model.predict(ctx.X_test), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=final_model,
            )
        )

    if bool(cfg.get("enable_random_forest", False)):
        params = dict(
            cfg.get(
                "random_forest",
                {
                    "n_estimators": 512,
                    "max_features": 0.7,
                    "min_samples_leaf": 1,
                    "bootstrap": True,
                },
            )
        )
        model = RandomForestRegressor(random_state=ctx.seed + 6200, n_jobs=8, **params)
        model.fit(ctx.X_train, ctx.y_train)
        val_pred = _inverse_target(model.predict(ctx.X_val), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        final_model = RandomForestRegressor(random_state=ctx.seed + 6200, n_jobs=8, **params)
        final_model.fit(X_tv, y_tv)
        candidates.append(
            CandidateOutput(
                name="LOCAL_FAST_RF",
                val_pred=val_pred,
                test_pred=_inverse_target(final_model.predict(ctx.X_test), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=final_model,
            )
        )

    if bool(cfg.get("enable_catboost", False)) and CatBoostRegressor is not None:
        params = dict(
            cfg.get(
                "catboost",
                {
                    "iterations": 900,
                    "depth": 6,
                    "learning_rate": 0.03,
                    "loss_function": "MAE",
                    "l2_leaf_reg": 3.0,
                },
            )
        )
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        params.setdefault("thread_count", 8)
        model = CatBoostRegressor(random_seed=ctx.seed + 6300, **params)
        model.fit(ctx.X_train, ctx.y_train)
        val_pred = _inverse_target(model.predict(ctx.X_val), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        final_model = CatBoostRegressor(random_seed=ctx.seed + 6300, **params)
        final_model.fit(X_tv, y_tv)
        candidates.append(
            CandidateOutput(
                name="LOCAL_FAST_CAT",
                val_pred=val_pred,
                test_pred=_inverse_target(final_model.predict(ctx.X_test), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=final_model,
            )
        )

    if bool(cfg.get("enable_knn", True)):
        n_neighbors = int(cfg.get("knn_neighbors", min(7, max(2, len(ctx.y_train) // 8))))
        n_neighbors = max(1, min(n_neighbors, len(ctx.y_train)))
        model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights=str(cfg.get("knn_weights", "distance")), metric="minkowski"),
        )
        model.fit(ctx.X_train, ctx.y_train)
        val_pred = _inverse_target(model.predict(ctx.X_val), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec)
        final_neighbors = max(1, min(n_neighbors, len(y_tv)))
        final_model = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=final_neighbors, weights=str(cfg.get("knn_weights", "distance")), metric="minkowski"),
        )
        final_model.fit(X_tv, y_tv)
        candidates.append(
            CandidateOutput(
                name="LOCAL_FAST_KNN",
                val_pred=val_pred,
                test_pred=_inverse_target(final_model.predict(ctx.X_test), ctx.target_transform, ctx.target_inv_scale, ctx.target_t0_sec),
                val_metrics=compute_metrics(ctx.y_val_sec, val_pred),
                model=final_model,
            )
        )
    return candidates
