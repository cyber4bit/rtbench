from __future__ import annotations

import numpy as np
import torch

from rtbench.hyper import HyperTLBundle, _HyperNet
from rtbench.metrics import compute_metrics
from rtbench.models.candidates import CandidateBuildContext, build_candidates
from rtbench.models.trees import CandidateOutput

from benchmarks._support import benchmark_callable


class _FastRegressor:
    def __init__(self, *args, **kwargs) -> None:
        self.bias = 0.0

    def fit(self, _X, y, *args, **kwargs) -> "_FastRegressor":
        self.bias = float(np.mean(np.asarray(y, dtype=np.float32)))
        return self

    def predict(self, X, *args, **kwargs) -> np.ndarray:
        n = len(X)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return (self.bias + ramp).astype(np.float32)

    def set_params(self, **kwargs) -> "_FastRegressor":
        return self


def run_benchmark(*, repeats: int = 5) -> dict[str, float | int]:
    ctx = _make_context()

    def _run() -> int:
        return len(
            build_candidates(
                ctx,
                fit_tree_models_fn=_fit_tree_models_fn,
                fit_ridge_models_fn=_fit_ridge_models_fn,
                train_mlp_fn=_train_mlp_fn,
                xgb_regressor_cls=_FastRegressor,
                lgbm_regressor_cls=_FastRegressor,
            )
        )

    candidate_count = _run()
    metrics = benchmark_callable(_run, repeats=repeats, warmups=1)
    metrics["candidate_count"] = int(candidate_count)
    metrics["source_rows"] = int(len(ctx.X_src))
    metrics["target_rows"] = int(len(ctx.X_train) + len(ctx.X_val) + len(ctx.X_test))
    return metrics


def _make_context() -> CandidateBuildContext:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    mol_dim = 4
    cp_dim = 3
    train_rows = 8
    val_rows = 4
    test_rows = 4
    source_rows = 10
    feature_dim = 6

    model = _HyperNet(mol_dim=mol_dim, cp_dim=cp_dim, embed_dim=5, mol_hidden=8, cp_hidden=7, dropout=0.0)
    bundle = HyperTLBundle(
        model=model,
        device=torch.device("cpu"),
        mol_mean=np.zeros(mol_dim, dtype=np.float32),
        mol_std=np.ones(mol_dim, dtype=np.float32),
        cp_mean=np.zeros(cp_dim, dtype=np.float32),
        cp_std=np.ones(cp_dim, dtype=np.float32),
        ridge_lambdas=[0.0, 0.1],
        ridge_lambda_b=0.1,
    )

    X_train = rng.normal(size=(train_rows, feature_dim)).astype(np.float32)
    X_val = rng.normal(size=(val_rows, feature_dim)).astype(np.float32)
    X_test = rng.normal(size=(test_rows, feature_dim)).astype(np.float32)
    X_src = rng.normal(size=(source_rows, feature_dim)).astype(np.float32)

    return CandidateBuildContext(
        model_cfg={
            "ENABLE_HYPER_TL": True,
            "HYPER_TL": {"ensemble_lambdas": True},
            "ENABLE_MLP": True,
            "ENABLE_MDL_SUBSET_CANDIDATES": True,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": True,
            "TRANSFER_TARGET_TRANSFORMS": ["log1p"],
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": True,
            "LOCAL_TARGET_TRANSFORMS": ["log1p"],
            "ENABLE_ANCHOR_TL": True,
            "ANCHOR_TL": {"min_train_points": 4},
            "EARLY_STOPPING_ROUNDS": 0,
        },
        X_src=X_src,
        X_src_cp=rng.normal(size=(source_rows, cp_dim)).astype(np.float32),
        y_src=np.linspace(0.5, 2.5, source_rows, dtype=np.float32),
        y_src_sec=np.linspace(0.5, 2.5, source_rows, dtype=np.float32),
        X_train=X_train,
        X_train_mol=rng.normal(size=(train_rows, mol_dim)).astype(np.float32),
        X_train_cp=rng.normal(size=(train_rows, cp_dim)).astype(np.float32),
        y_train=np.linspace(1.0, 3.0, train_rows, dtype=np.float32),
        y_train_sec=np.linspace(1.0, 3.0, train_rows, dtype=np.float32),
        X_val=X_val,
        X_val_mol=rng.normal(size=(val_rows, mol_dim)).astype(np.float32),
        y_val_used=np.linspace(1.5, 2.1, val_rows, dtype=np.float32),
        y_val_sec=np.linspace(1.5, 2.1, val_rows, dtype=np.float32),
        X_test=X_test,
        X_test_mol=rng.normal(size=(test_rows, mol_dim)).astype(np.float32),
        seed=0,
        source_weight=0.2,
        target_weight=1.0,
        group_sizes={"descriptor": 2, "fingerprint": 2, "meta": 2},
        hyper_bundle=bundle,
        source_row_dataset_ids=np.array(["0001"] * 5 + ["0002"] * 5, dtype=object),
        source_mol_keys=np.array([f"mol-{i % 6}" for i in range(source_rows)], dtype=object),
        target_mol_keys=np.array([f"mol-{i % 6}" for i in range(train_rows + val_rows + test_rows)], dtype=object),
        target_cp_reference=np.ones(cp_dim, dtype=np.float32),
    )


def _fit_tree_models_fn(**kwargs) -> list[CandidateOutput]:
    y_val = np.asarray(kwargs["y_val_sec"], dtype=np.float32)
    test_rows = len(kwargs["X_test"])
    prefix = str(kwargs.get("name_prefix", ""))
    return [
        _candidate(f"{prefix}TREE_A", y_val + 0.10, test_rows, 0.20),
        _candidate(f"{prefix}TREE_B", y_val + 0.15, test_rows, 0.25),
    ]


def _fit_ridge_models_fn(**kwargs) -> list[CandidateOutput]:
    y_val = np.asarray(kwargs["y_val_sec"], dtype=np.float32)
    test_rows = len(kwargs["X_test"])
    prefix = str(kwargs.get("name_prefix", ""))
    return [
        _candidate(f"{prefix}RIDGE_A", y_val + 0.05, test_rows, 0.10),
        _candidate(f"{prefix}RIDGE_B", y_val + 0.08, test_rows, 0.12),
    ]


def _train_mlp_fn(**kwargs) -> CandidateOutput:
    y_val = np.asarray(kwargs["y_val_sec"], dtype=np.float32)
    return _candidate("MLP_TL", y_val + 0.07, len(kwargs["X_test"]), 0.11)


def _candidate(name: str, val_pred: np.ndarray, test_rows: int, offset: float) -> CandidateOutput:
    test_pred = np.linspace(1.0 + offset, 1.0 + offset + 0.05 * max(test_rows - 1, 0), test_rows, dtype=np.float32)
    return CandidateOutput(
        name=name,
        val_pred=np.asarray(val_pred, dtype=np.float32),
        test_pred=test_pred,
        val_metrics=compute_metrics(np.asarray(val_pred, dtype=np.float32) - offset, np.asarray(val_pred, dtype=np.float32)),
        model={"name": name},
    )


if __name__ == "__main__":
    import json

    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
