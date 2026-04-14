from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from rtbench.hyper import HyperTLBundle, _HyperNet, head_prior, mol_embeddings, pretrain_hyper_tl, ridge_prior_fit_predict
from rtbench.metrics import compute_metrics
from rtbench.models.candidates.common import CandidateBuildContext
from rtbench.models.candidates.hyper_candidates import build_hyper_candidates


def _make_hyper_data(n_rows: int, *, mol_dim: int = 6, cp_dim: int = 4, embed_dim: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_mol = rng.normal(size=(n_rows, mol_dim)).astype(np.float32)
    X_cp = rng.normal(size=(n_rows, cp_dim)).astype(np.float32)
    mol_proj = rng.normal(size=(mol_dim, embed_dim)).astype(np.float32)
    cp_proj = rng.normal(size=(cp_dim, embed_dim)).astype(np.float32)
    mol_embed = np.maximum(X_mol @ mol_proj, 0.0)
    cp_head = X_cp @ cp_proj
    y = (mol_embed * cp_head).sum(axis=1) + 0.25 * X_cp[:, 0]
    return X_mol, X_cp, y.astype(np.float32)


def _hyper_cfg() -> dict[str, float | int | list[float]]:
    return {
        "embed_dim": 5,
        "mol_hidden": 16,
        "cp_hidden": 12,
        "dropout": 0.0,
        "epochs": 40,
        "batch_size": 16,
        "lr": 0.01,
        "weight_decay": 1e-4,
        "val_frac": 0.2,
        "patience": 6,
        "ridge_lambdas": [0.0, 0.1, 1.0],
        "ridge_lambda_b": 0.2,
    }


def _train_bundle(seed: int = 0) -> tuple[HyperTLBundle, np.ndarray, np.ndarray, np.ndarray]:
    X_mol, X_cp, y = _make_hyper_data(80, seed=seed)
    bundle = pretrain_hyper_tl(X_mol, X_cp, y, cfg=_hyper_cfg(), seed=seed)
    return bundle, X_mol, X_cp, y


def test_hypernet_forward_returns_batch_predictions() -> None:
    model = _HyperNet(mol_dim=6, cp_dim=4, embed_dim=5, mol_hidden=8, cp_hidden=7, dropout=0.0)
    pred = model(torch.randn(9, 6), torch.randn(9, 4))
    assert pred.shape == (9,)


def test_pretrain_hyper_tl_returns_bundle_and_embeddings() -> None:
    bundle, X_mol, X_cp, _y = _train_bundle(seed=1)

    embeddings = mol_embeddings(bundle, X_mol[:12])
    prior_w, prior_b = head_prior(bundle, X_cp[0])

    assert bundle.mol_mean.shape == (6,)
    assert bundle.mol_std.shape == (6,)
    assert bundle.cp_mean.shape == (4,)
    assert bundle.cp_std.shape == (4,)
    assert embeddings.shape == (12, 5)
    assert prior_w.shape == (5,)
    assert isinstance(prior_b, float)
    assert np.all(np.isfinite(embeddings))
    assert np.all(np.isfinite(prior_w))


def test_ridge_prior_fit_predict_keeps_embeddings_frozen() -> None:
    bundle, X_mol, X_cp, y = _train_bundle(seed=2)

    embeddings_before = mol_embeddings(bundle, X_mol[:14])
    prior_w, prior_b = head_prior(bundle, X_cp[0])
    pred = ridge_prior_fit_predict(
        Z_train=embeddings_before[:10],
        y_train=y[:10],
        Z_eval=embeddings_before[10:14],
        w0=prior_w,
        b0=prior_b,
        lam=0.1,
        lam_b=0.2,
    )
    embeddings_after = mol_embeddings(bundle, X_mol[:14])

    assert pred.shape == (4,)
    assert np.all(np.isfinite(pred))
    np.testing.assert_allclose(embeddings_before, embeddings_after, atol=1e-6)


def test_hyper_stats_save_and_reload_consistently(tmp_path: Path) -> None:
    bundle, X_mol, X_cp, _y = _train_bundle(seed=3)
    stats_path = tmp_path / "hyper_stats.npz"
    np.savez(
        stats_path,
        mol_mean=bundle.mol_mean,
        mol_std=bundle.mol_std,
        cp_mean=bundle.cp_mean,
        cp_std=bundle.cp_std,
    )
    loaded = np.load(stats_path)
    clone = HyperTLBundle(
        model=bundle.model,
        device=bundle.device,
        mol_mean=loaded["mol_mean"].astype(np.float32),
        mol_std=loaded["mol_std"].astype(np.float32),
        cp_mean=loaded["cp_mean"].astype(np.float32),
        cp_std=loaded["cp_std"].astype(np.float32),
        ridge_lambdas=list(bundle.ridge_lambdas),
        ridge_lambda_b=float(bundle.ridge_lambda_b),
    )

    np.testing.assert_allclose(mol_embeddings(bundle, X_mol[:10]), mol_embeddings(clone, X_mol[:10]), atol=1e-6)
    prior_w_1, prior_b_1 = head_prior(bundle, X_cp[0])
    prior_w_2, prior_b_2 = head_prior(clone, X_cp[0])
    np.testing.assert_allclose(prior_w_1, prior_w_2, atol=1e-6)
    assert prior_b_1 == prior_b_2


def test_build_hyper_candidates_selects_best_lambda() -> None:
    bundle, X_src_mol, X_src_cp, y_src = _train_bundle(seed=4)
    X_tgt_mol, X_tgt_cp, y_tgt = _make_hyper_data(18, seed=14)

    ctx = CandidateBuildContext(
        model_cfg={"ENABLE_HYPER_TL": True, "HYPER_TL": {"ensemble_lambdas": False}},
        X_src=np.zeros((len(y_src), 1), dtype=np.float32),
        X_src_cp=X_src_cp,
        y_src=y_src,
        y_src_sec=y_src,
        X_train=np.zeros((8, 1), dtype=np.float32),
        X_train_mol=X_tgt_mol[:8],
        X_train_cp=X_tgt_cp[:8],
        y_train=y_tgt[:8],
        y_train_sec=y_tgt[:8],
        X_val=np.zeros((5, 1), dtype=np.float32),
        X_val_mol=X_tgt_mol[8:13],
        y_val_used=y_tgt[8:13],
        y_val_sec=y_tgt[8:13],
        X_test=np.zeros((5, 1), dtype=np.float32),
        X_test_mol=X_tgt_mol[13:18],
        seed=0,
        source_weight=0.2,
        target_weight=1.0,
        group_sizes={"mol": 1},
        hyper_bundle=bundle,
    )

    Z_tr = mol_embeddings(bundle, ctx.X_train_mol)
    Z_va = mol_embeddings(bundle, ctx.X_val_mol)
    prior_w, prior_b = head_prior(bundle, ctx.X_train_cp[0])
    best_val = None
    best_mae = float("inf")
    for lam in bundle.ridge_lambdas:
        val_pred = ridge_prior_fit_predict(
            Z_train=Z_tr,
            y_train=ctx.y_train,
            Z_eval=Z_va,
            w0=prior_w,
            b0=prior_b,
            lam=float(lam),
            lam_b=float(bundle.ridge_lambda_b),
        )
        mae = float(compute_metrics(ctx.y_val_sec, val_pred)["mae"])
        if mae < best_mae:
            best_mae = mae
            best_val = val_pred

    candidates = build_hyper_candidates(ctx)

    assert len(candidates) == 1
    assert candidates[0].name == "HYPER_TL_ENS(n=1,lam_ens=False)"
    np.testing.assert_allclose(candidates[0].val_pred, best_val, atol=1e-6)


def test_ridge_prior_fit_predict_lambda_zero_matches_unregularized_solution() -> None:
    z_train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    y_train = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    z_eval = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    x_train = np.concatenate([z_train, np.ones((len(z_train), 1), dtype=np.float32)], axis=1)
    theta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    expected = np.concatenate([z_eval, np.ones((len(z_eval), 1), dtype=np.float32)], axis=1) @ theta

    pred = ridge_prior_fit_predict(
        Z_train=z_train,
        y_train=y_train,
        Z_eval=z_eval,
        w0=np.array([100.0, -100.0], dtype=np.float32),
        b0=50.0,
        lam=0.0,
        lam_b=0.0,
    )

    np.testing.assert_allclose(pred, expected.astype(np.float32), atol=1e-5)


def test_pretrain_hyper_tl_falls_back_from_single_dataset_holdout() -> None:
    x_mol, x_cp, y = _make_hyper_data(24, seed=9)
    bundle = pretrain_hyper_tl(
        x_mol,
        x_cp,
        y,
        cfg=dict(_hyper_cfg(), val_split="dataset"),
        seed=9,
        dataset_ids=np.array(["0001"] * len(y), dtype=object),
    )

    embeddings = mol_embeddings(bundle, x_mol[:6])
    assert embeddings.shape == (6, 5)
    assert np.isfinite(embeddings).all()


def test_head_prior_supports_empty_cp_vector() -> None:
    rng = np.random.default_rng(12)
    x_mol = rng.normal(size=(20, 6)).astype(np.float32)
    x_cp = np.zeros((20, 0), dtype=np.float32)
    y = np.maximum(x_mol[:, 0] + 0.5 * x_mol[:, 1], 0.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
        bundle = pretrain_hyper_tl(
            x_mol,
            x_cp,
            y,
            cfg={
                "embed_dim": 4,
                "mol_hidden": 12,
                "cp_hidden": 8,
                "dropout": 0.0,
                "epochs": 10,
                "batch_size": 8,
                "lr": 0.01,
                "weight_decay": 1e-4,
                "val_frac": 0.2,
                "patience": 3,
                "ridge_lambdas": [0.0, 0.1],
                "ridge_lambda_b": 0.1,
            },
            seed=12,
        )

    prior_w, prior_b = head_prior(bundle, np.zeros(0, dtype=np.float32))
    assert prior_w.shape == (4,)
    assert np.isfinite(prior_w).all()
    assert isinstance(prior_b, float)
