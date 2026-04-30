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
        use_film: bool = False,
        film_scale: float = 0.25,
        num_tasks: int = 0,
        use_task_adapters: bool = False,
        adapter_reduction: int = 4,
        use_cross_stitch: bool = False,
        cross_stitch_init: float = 0.05,
        mol_sequence_len: int = 0,
        mol_sequence_vocab_size: int = 0,
        mol_sequence_embed_dim: int = 16,
        mol_sequence_channels: int = 32,
        mol_sequence_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.use_film = bool(use_film)
        self.film_scale = float(film_scale)
        self.num_tasks = int(max(0, num_tasks))
        self.use_task_adapters = bool(use_task_adapters) and self.num_tasks > 0
        self.use_cross_stitch = bool(use_cross_stitch) and self.use_task_adapters
        self.mol_sequence_len = int(max(0, mol_sequence_len))
        self.mol_sequence_vocab_size = int(max(0, mol_sequence_vocab_size))
        self.use_mol_sequence = self.mol_sequence_len > 0 and self.mol_sequence_vocab_size > 2
        self.mol_numeric_dim = int(mol_dim) - self.mol_sequence_len if self.use_mol_sequence else int(mol_dim)
        if self.mol_numeric_dim <= 0:
            self.use_mol_sequence = False
            self.mol_sequence_len = 0
            self.mol_numeric_dim = int(mol_dim)
        self.mol = nn.Sequential(
            nn.Linear(self.mol_numeric_dim, mol_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mol_hidden, embed_dim),
            nn.ReLU(),
        )
        self.mol_sequence_weight = float(mol_sequence_weight)
        if self.use_mol_sequence:
            seq_embed_dim = int(max(4, mol_sequence_embed_dim))
            seq_channels = int(max(4, mol_sequence_channels))
            self.seq_embedding = nn.Embedding(self.mol_sequence_vocab_size, seq_embed_dim, padding_idx=0)
            self.seq_convs = nn.ModuleList(
                [
                    nn.Conv1d(seq_embed_dim, seq_channels, kernel_size=k)
                    for k in (3, 5, 7)
                    if self.mol_sequence_len >= k
                ]
            )
            if not self.seq_convs:
                self.seq_convs = nn.ModuleList([nn.Conv1d(seq_embed_dim, seq_channels, kernel_size=1)])
            self.seq_proj = nn.Sequential(
                nn.Linear(seq_channels * len(self.seq_convs), embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
        cp_out_dim = embed_dim + 1
        if self.use_film:
            cp_out_dim += 2 * embed_dim
        self.cp = nn.Sequential(
            nn.Linear(cp_dim, cp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cp_hidden, cp_out_dim),
        )
        if self.use_task_adapters:
            bottleneck = int(max(1, embed_dim // max(1, int(adapter_reduction))))
            self.adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embed_dim, bottleneck),
                        nn.ReLU(),
                        nn.Linear(bottleneck, embed_dim),
                    )
                    for _ in range(self.num_tasks)
                ]
            )
            if self.use_cross_stitch:
                alpha = torch.full((self.num_tasks, self.num_tasks), float(cross_stitch_init), dtype=torch.float32)
                alpha.fill_diagonal_(1.0)
                self.cross_stitch = nn.Parameter(alpha)

    def encode_mol(self, x_mol: torch.Tensor) -> torch.Tensor:
        if not self.use_mol_sequence:
            return self.mol(x_mol)
        x_num = x_mol[:, : self.mol_numeric_dim]
        x_seq = x_mol[:, self.mol_numeric_dim : self.mol_numeric_dim + self.mol_sequence_len]
        z_num = self.mol(x_num)
        seq_ids = torch.clamp(torch.round(x_seq), min=0, max=self.mol_sequence_vocab_size - 1).long()
        seq_emb = self.seq_embedding(seq_ids).transpose(1, 2)
        pooled = []
        for conv in self.seq_convs:
            h = torch.relu(conv(seq_emb))
            pooled.append(torch.amax(h, dim=2))
        z_seq = self.seq_proj(torch.cat(pooled, dim=1))
        return torch.relu(z_num + self.mol_sequence_weight * z_seq)

    def _split_cp(self, x_cp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        wb = self.cp(x_cp)
        w = wb[:, : self.embed_dim]
        b = wb[:, self.embed_dim :].squeeze(-1)
        gamma = None
        beta = None
        if self.use_film:
            b = wb[:, self.embed_dim]
            start = self.embed_dim + 1
            gamma = wb[:, start : start + self.embed_dim]
            beta = wb[:, start + self.embed_dim : start + 2 * self.embed_dim]
        return w, b, gamma, beta

    def head_from_cp(self, x_cp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w, b, _gamma, _beta = self._split_cp(x_cp)
        return w, b

    def conditioned_embedding(self, x_mol: torch.Tensor, x_cp: torch.Tensor) -> torch.Tensor:
        z = self.encode_mol(x_mol)
        if not self.use_film:
            return z
        _w, _b, gamma, beta = self._split_cp(x_cp)
        if gamma is None or beta is None:
            return z
        scale = 1.0 + self.film_scale * torch.tanh(gamma)
        shift = self.film_scale * beta
        return z * scale + shift

    def task_adapt(self, z: torch.Tensor, task_id: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_task_adapters or task_id is None:
            return z
        task_id = task_id.to(device=z.device, dtype=torch.long).view(-1)
        if task_id.numel() != z.shape[0]:
            return z
        valid = (task_id >= 0) & (task_id < self.num_tasks)
        if int(valid.sum().item()) == 0:
            return z
        adapted = z.clone()
        present: list[int] = []
        task_means: dict[int, torch.Tensor] = {}
        for t_id in range(self.num_tasks):
            mask = valid & (task_id == t_id)
            if int(mask.sum().item()) == 0:
                continue
            present.append(t_id)
            adapted_t = z[mask] + self.adapters[t_id](z[mask])
            adapted[mask] = adapted_t
            task_means[t_id] = adapted_t.mean(dim=0, keepdim=True)
        if self.use_cross_stitch and len(present) > 1:
            stitched = adapted.clone()
            for t_id in present:
                mask = valid & (task_id == t_id)
                mixed = self.cross_stitch[t_id, t_id] * adapted[mask]
                for other_id in present:
                    if other_id == t_id:
                        continue
                    mixed = mixed + self.cross_stitch[t_id, other_id] * task_means[other_id]
                stitched[mask] = mixed
            adapted = stitched
        return adapted

    def forward(self, x_mol: torch.Tensor, x_cp: torch.Tensor, task_id: torch.Tensor | None = None) -> torch.Tensor:
        z = self.conditioned_embedding(x_mol, x_cp)
        z = self.task_adapt(z, task_id)
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
    use_conditioned_embeddings: bool = False
    task_to_index: dict[str, int] | None = None
    use_task_adapters: bool = False
    aux_X_mol: np.ndarray | None = None
    aux_X_cp: np.ndarray | None = None
    aux_X_full: np.ndarray | None = None
    aux_y: np.ndarray | None = None
    aux_y_sec: np.ndarray | None = None
    aux_dataset_ids: np.ndarray | None = None
    aux_mol_keys: np.ndarray | None = None


def _compute_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _compute_mol_mean_std(X: np.ndarray, sequence_len: int) -> tuple[np.ndarray, np.ndarray]:
    mean, std = _compute_mean_std(X)
    sequence_len = int(max(0, sequence_len))
    if sequence_len > 0 and sequence_len < X.shape[1]:
        mean[-sequence_len:] = 0.0
        std[-sequence_len:] = 1.0
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
    use_film = bool(cfg.get("use_film", False))
    film_scale = float(cfg.get("film_scale", 0.25))
    use_conditioned_embeddings = bool(cfg.get("use_conditioned_embeddings", use_film))
    use_task_adapters = bool(cfg.get("use_task_adapters", False))
    adapter_reduction = int(cfg.get("adapter_reduction", 4))
    use_cross_stitch = bool(cfg.get("use_cross_stitch", False))
    cross_stitch_init = float(cfg.get("cross_stitch_init", 0.05))
    use_mol_sequence = bool(cfg.get("use_mol_sequence", False))
    mol_sequence_len = int(cfg.get("mol_sequence_len", 0) or 0) if use_mol_sequence else 0
    mol_sequence_vocab_size = int(cfg.get("mol_sequence_vocab_size", 0) or 0)
    mol_sequence_embed_dim = int(cfg.get("mol_sequence_embed_dim", 16))
    mol_sequence_channels = int(cfg.get("mol_sequence_channels", 32))
    mol_sequence_weight = float(cfg.get("mol_sequence_weight", 1.0))
    epochs = int(cfg.get("epochs", 60))
    batch_size = int(cfg.get("batch_size", 256))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    val_frac = float(cfg.get("val_frac", 0.10))
    patience = int(cfg.get("patience", 8))
    val_split = str(cfg.get("val_split", "sample")).strip().lower()
    ranking_loss_weight = float(cfg.get("ranking_loss_weight", 0.0) or 0.0)
    ranking_loss_temp = float(cfg.get("ranking_loss_temp", 0.05) or 0.05)
    ranking_min_delta = float(cfg.get("ranking_min_delta", 0.0) or 0.0)

    ridge_lambdas = [float(x) for x in cfg.get("ridge_lambdas", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])]
    ridge_lambda_b = float(cfg.get("ridge_lambda_b", 1e-2))

    mol_mean, mol_std = _compute_mol_mean_std(X_src_mol, mol_sequence_len)
    cp_mean, cp_std = _compute_mean_std(X_src_cp)
    Xm = _standardize(X_src_mol, mol_mean, mol_std)
    Xc = _standardize(X_src_cp, cp_mean, cp_std)
    y = np.asarray(y_src, dtype=np.float32)

    n = len(y)
    group_index = np.zeros(n, dtype=np.int64)
    if dataset_ids is not None:
        ds_for_groups = np.asarray(dataset_ids, dtype=object).reshape(-1)
        if len(ds_for_groups) != n:
            raise ValueError(f"dataset_ids length mismatch: got={len(ds_for_groups)}, expected={n}")
        group_to_index = {str(raw).zfill(4): i for i, raw in enumerate(sorted(set(str(x).zfill(4) for x in ds_for_groups.tolist())))}
        group_index = np.array([group_to_index[str(raw).zfill(4)] for raw in ds_for_groups.tolist()], dtype=np.int64)
    task_to_index: dict[str, int] | None = None
    task_index = np.full(n, -1, dtype=np.int64)
    if use_task_adapters and dataset_ids is not None:
        ds_for_tasks = np.asarray(dataset_ids, dtype=object).reshape(-1)
        if len(ds_for_tasks) != n:
            raise ValueError(f"dataset_ids length mismatch: got={len(ds_for_tasks)}, expected={n}")
        task_to_index = {str(raw).zfill(4): i for i, raw in enumerate(sorted(set(str(x).zfill(4) for x in ds_for_tasks.tolist())))}
        task_index = np.array([task_to_index[str(raw).zfill(4)] for raw in ds_for_tasks.tolist()], dtype=np.int64)
    else:
        use_task_adapters = False
        use_cross_stitch = False

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
        torch.from_numpy(task_index[tr_idx]),
        torch.from_numpy(group_index[tr_idx]),
    )
    va_m = torch.from_numpy(Xm[val_idx]).to(device)
    va_c = torch.from_numpy(Xc[val_idx]).to(device)
    va_y = torch.from_numpy(y[val_idx]).to(device)
    va_t = torch.from_numpy(task_index[val_idx]).to(device)
    dl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=False)

    model = _HyperNet(
        mol_dim=mol_dim,
        cp_dim=cp_dim,
        embed_dim=embed_dim,
        mol_hidden=mol_hidden,
        cp_hidden=cp_hidden,
        dropout=dropout,
        use_film=use_film,
        film_scale=film_scale,
        num_tasks=len(task_to_index or {}),
        use_task_adapters=use_task_adapters,
        adapter_reduction=adapter_reduction,
        use_cross_stitch=use_cross_stitch,
        cross_stitch_init=cross_stitch_init,
        mol_sequence_len=mol_sequence_len,
        mol_sequence_vocab_size=mol_sequence_vocab_size,
        mol_sequence_embed_dim=mol_sequence_embed_dim,
        mol_sequence_channels=mol_sequence_channels,
        mol_sequence_weight=mol_sequence_weight,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(epochs):
        model.train()
        for b_m, b_c, b_y, b_w, b_t, b_g in dl:
            b_m = b_m.to(device)
            b_c = b_c.to(device)
            b_y = b_y.to(device)
            b_w = b_w.to(device)
            b_t = b_t.to(device)
            b_g = b_g.to(device)
            pred = model(b_m, b_c, b_t)
            loss = torch.mean(torch.abs(pred - b_y) * b_w)
            if ranking_loss_weight > 0.0 and b_y.numel() >= 2:
                pair_group = b_g[:, None] == b_g[None, :]
                target_diff = b_y[:, None] - b_y[None, :]
                pair_mask = pair_group & (torch.abs(target_diff) > ranking_min_delta)
                if bool(pair_mask.any().item()):
                    pred_diff = pred[:, None] - pred[None, :]
                    sign = torch.sign(target_diff[pair_mask])
                    rank_logits = pred_diff[pair_mask] * sign / max(ranking_loss_temp, 1e-6)
                    pair_weight = torch.sqrt((b_w[:, None] * b_w[None, :]).clamp_min(0.0))[pair_mask]
                    rank_loss = torch.nn.functional.softplus(-rank_logits)
                    loss = loss + ranking_loss_weight * torch.mean(rank_loss * pair_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(va_m, va_c, va_t)
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
        use_conditioned_embeddings=use_conditioned_embeddings,
        task_to_index=task_to_index,
        use_task_adapters=use_task_adapters,
    )


def mol_embeddings(
    bundle: HyperTLBundle,
    X_mol: np.ndarray,
    cp_vec: np.ndarray | None = None,
    dataset_id: str | None = None,
) -> np.ndarray:
    Xm = _standardize(X_mol, bundle.mol_mean, bundle.mol_std)
    x = torch.from_numpy(Xm).to(bundle.device)
    bundle.model.eval()
    with torch.no_grad():
        if (
            bool(getattr(bundle, "use_conditioned_embeddings", False))
            and cp_vec is not None
            and bool(getattr(bundle.model, "use_film", False))
        ):
            cp_arr = np.asarray(cp_vec, dtype=np.float32).reshape(1, -1)
            cp_arr = np.repeat(_standardize(cp_arr, bundle.cp_mean, bundle.cp_std), repeats=len(Xm), axis=0)
            x_cp = torch.from_numpy(cp_arr).to(bundle.device)
            z = bundle.model.conditioned_embedding(x, x_cp).detach().cpu().numpy()
        else:
            z = bundle.model.encode_mol(x).detach().cpu().numpy()
        if bool(getattr(bundle, "use_task_adapters", False)) and dataset_id is not None:
            task_to_index = getattr(bundle, "task_to_index", None) or {}
            task_index = task_to_index.get(str(dataset_id).zfill(4))
            if task_index is not None:
                z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).to(bundle.device)
                task_t = torch.full((len(z),), int(task_index), dtype=torch.long, device=bundle.device)
                z = bundle.model.task_adapt(z_t, task_t).detach().cpu().numpy()
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
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Ridge with Gaussian prior centered at (w0,b0): returns predictions for Z_eval."""
    Zt = np.asarray(Z_train, dtype=np.float32)
    ye = np.asarray(y_train, dtype=np.float32).reshape(-1)
    Ze = np.asarray(Z_eval, dtype=np.float32)
    n, d = Zt.shape
    X = np.concatenate([Zt, np.ones((n, 1), dtype=np.float32)], axis=1)
    if sample_weight is None:
        w_sample = np.ones(n, dtype=np.float32)
    else:
        w_sample = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if len(w_sample) != n:
            raise ValueError(f"sample_weight length mismatch: got={len(w_sample)}, expected={n}")
        w_sample = np.where(np.isfinite(w_sample) & (w_sample > 0.0), w_sample, 0.0).astype(np.float32)
    Xw = X * w_sample[:, None]
    XTX = X.T @ Xw
    rhs = X.T @ (ye * w_sample)
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
