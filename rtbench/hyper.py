from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _HyperNet(nn.Module):
    def __init__(
        self,
        mol_dim: int,
        cp_dim: int,
        embed_dim: int,
        mol_hidden: int,
        cp_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mol = nn.Sequential(
            nn.Linear(mol_dim, mol_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mol_hidden, embed_dim),
            nn.ReLU(),
        )
        self.cp = nn.Sequential(
            nn.Linear(cp_dim, cp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cp_hidden, embed_dim + 1),
        )

    def encode_mol(self, x_mol: torch.Tensor) -> torch.Tensor:
        return self.mol(x_mol)

    def head_from_cp(self, x_cp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        wb = self.cp(x_cp)
        w = wb[:, : self.embed_dim]
        b = wb[:, self.embed_dim :].squeeze(-1)
        return w, b

    def forward(self, x_mol: torch.Tensor, x_cp: torch.Tensor) -> torch.Tensor:
        z = self.encode_mol(x_mol)
        w, b = self.head_from_cp(x_cp)
        return (z * w).sum(dim=1) + b


@dataclass
class HyperTLBundle:
    model: _HyperNet
    device: torch.device
    mol_mean: np.ndarray
    mol_std: np.ndarray
    cp_mean: np.ndarray
    cp_std: np.ndarray
    ridge_lambdas: list[float]
    ridge_lambda_b: float


def _compute_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def pretrain_hyper_tl(
    X_src_mol: np.ndarray,
    X_src_cp: np.ndarray,
    y_src: np.ndarray,
    cfg: dict[str, Any],
    seed: int = 0,
    sample_weights: np.ndarray | None = None,
    dataset_ids: np.ndarray | None = None,
) -> HyperTLBundle:
    """Pretrain a CP-conditioned hypernetwork on source datasets once, reuse for all targets."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mol_dim = int(X_src_mol.shape[1])
    cp_dim = int(X_src_cp.shape[1])
    embed_dim = int(cfg.get("embed_dim", 128))
    mol_hidden = int(cfg.get("mol_hidden", 512))
    cp_hidden = int(cfg.get("cp_hidden", 256))
    dropout = float(cfg.get("dropout", 0.10))
    epochs = int(cfg.get("epochs", 60))
    batch_size = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    val_frac = float(cfg.get("val_frac", 0.10))
    patience = int(cfg.get("patience", 8))
    val_split = str(cfg.get("val_split", "sample")).strip().lower()

    ridge_lambdas = [float(x) for x in cfg.get("ridge_lambdas", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])]
    ridge_lambda_b = float(cfg.get("ridge_lambda_b", 1e-2))

    mol_mean, mol_std = _compute_mean_std(X_src_mol)
    cp_mean, cp_std = _compute_mean_std(X_src_cp)
    Xm = _standardize(X_src_mol, mol_mean, mol_std)
    Xc = _standardize(X_src_cp, cp_mean, cp_std)
    y = np.asarray(y_src, dtype=np.float32)

    n = len(y)
    w = np.ones(n, dtype=np.float32)
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if len(sw) != n:
            raise ValueError(f"sample_weights length mismatch: got={len(sw)}, expected={n}")
        sw = np.where(np.isfinite(sw) & (sw > 0.0), sw, 0.0).astype(np.float32)
        if float(np.sum(sw)) > 0.0:
            # Keep weight scale comparable to unweighted training (mean ~ 1.0).
            w = (sw / max(float(np.mean(sw)), 1e-12)).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    if dataset_ids is not None and val_split in ("dataset", "dataset_holdout", "dataset-level"):
        ds_arr = np.asarray(dataset_ids, dtype=object).reshape(-1)
        if len(ds_arr) != n:
            raise ValueError(f"dataset_ids length mismatch: got={len(ds_arr)}, expected={n}")
        uniq = sorted(set(str(x) for x in ds_arr.tolist()))
        if len(uniq) >= 2:
            ds_perm = np.array(uniq, dtype=object)
            rng.shuffle(ds_perm)
            n_val_ds = int(max(1, round(len(ds_perm) * val_frac)))
            val_ds = set(str(x) for x in ds_perm[:n_val_ds].tolist())
            mask = np.array([str(x) in val_ds for x in ds_arr.tolist()], dtype=bool)
            val_idx = np.where(mask)[0]
            tr_idx = np.where(~mask)[0]
        else:
            val_idx = np.array([], dtype=int)
            tr_idx = np.array([], dtype=int)
        # Safety: fall back to sample split if the dataset split is degenerate.
        if val_idx.size == 0 or tr_idx.size == 0:
            idx = np.arange(n)
            rng.shuffle(idx)
            n_val = int(max(1, round(n * val_frac)))
            val_idx = idx[:n_val]
            tr_idx = idx[n_val:]
    else:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(max(1, round(n * val_frac)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

    tr = TensorDataset(
        torch.from_numpy(Xm[tr_idx]),
        torch.from_numpy(Xc[tr_idx]),
        torch.from_numpy(y[tr_idx]),
        torch.from_numpy(w[tr_idx]),
    )
    va_m = torch.from_numpy(Xm[val_idx]).to(device)
    va_c = torch.from_numpy(Xc[val_idx]).to(device)
    va_y = torch.from_numpy(y[val_idx]).to(device)
    dl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)

    model = _HyperNet(
        mol_dim=mol_dim,
        cp_dim=cp_dim,
        embed_dim=embed_dim,
        mol_hidden=mol_hidden,
        cp_hidden=cp_hidden,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for b_m, b_c, b_y, b_w in dl:
            b_m = b_m.to(device)
            b_c = b_c.to(device)
            b_y = b_y.to(device)
            b_w = b_w.to(device)
            pred = model(b_m, b_c)
            loss = torch.mean(torch.abs(pred - b_y) * b_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(va_m, va_c)
            v_mae = float(torch.mean(torch.abs(v_pred - va_y)).item())
        if v_mae < best_val:
            best_val = v_mae
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    return HyperTLBundle(
        model=model,
        device=device,
        mol_mean=mol_mean,
        mol_std=mol_std,
        cp_mean=cp_mean,
        cp_std=cp_std,
        ridge_lambdas=ridge_lambdas,
        ridge_lambda_b=ridge_lambda_b,
    )


def mol_embeddings(bundle: HyperTLBundle, X_mol: np.ndarray) -> np.ndarray:
    Xm = _standardize(X_mol, bundle.mol_mean, bundle.mol_std)
    x = torch.from_numpy(Xm).to(bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        z = bundle.model.encode_mol(x).detach().cpu().numpy()
    return np.asarray(z, dtype=np.float32)


def head_prior(bundle: HyperTLBundle, cp_vec: np.ndarray) -> tuple[np.ndarray, float]:
    cp_vec = np.asarray(cp_vec, dtype=np.float32).reshape(1, -1)
    Xc = _standardize(cp_vec, bundle.cp_mean, bundle.cp_std)
    x = torch.from_numpy(Xc).to(bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        w, b = bundle.model.head_from_cp(x)
    w_np = w.detach().cpu().numpy().reshape(-1).astype(np.float32)
    b_np = float(b.detach().cpu().numpy().reshape(-1)[0])
    return w_np, b_np


def ridge_prior_fit_predict(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_eval: np.ndarray,
    w0: np.ndarray,
    b0: float,
    lam: float,
    lam_b: float,
) -> np.ndarray:
    """Ridge with Gaussian prior centered at (w0,b0): returns predictions for Z_eval."""
    Zt = np.asarray(Z_train, dtype=np.float32)
    ye = np.asarray(y_train, dtype=np.float32).reshape(-1)
    Ze = np.asarray(Z_eval, dtype=np.float32)
    n, d = Zt.shape
    X = np.concatenate([Zt, np.ones((n, 1), dtype=np.float32)], axis=1)
    XTX = X.T @ X
    rhs = X.T @ ye
    lam = float(lam)
    lam_b = float(lam_b)
    if lam > 0:
        XTX[:d, :d] += lam * np.eye(d, dtype=np.float32)
        rhs[:d] += lam * np.asarray(w0, dtype=np.float32).reshape(-1)
    if lam_b > 0:
        XTX[d, d] += lam_b
        rhs[d] += lam_b * float(b0)
    # Always add a tiny jitter for numerical stability (n << d is common in external datasets).
    XTX += 1e-6 * np.eye(d + 1, dtype=np.float32)
    try:
        theta = np.linalg.solve(XTX, rhs)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(XTX, rhs, rcond=None)[0]
    Xe = np.concatenate([Ze, np.ones((len(Ze), 1), dtype=np.float32)], axis=1)
    return (Xe @ theta).astype(np.float32)
