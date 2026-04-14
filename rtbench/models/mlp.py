from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..metrics import compute_metrics
from .trees import CandidateOutput, _inverse_target


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden:
            h_i = int(max(4, h))
            layers.append(nn.Linear(prev, h_i))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h_i
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    model_cfg: dict[str, Any],
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_t_train: np.ndarray,
    y_t_train: np.ndarray,
    X_val: np.ndarray,
    y_val_used: np.ndarray,
    y_val_sec: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    target_transform: str = "none",
    target_inv_scale: float = 1.0,
    target_t0_sec: float = 1.0,
) -> CandidateOutput:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLP cannot handle NaNs; impute with feature-wise median and append missing-value masks.
    X_fit = np.concatenate([X_src, X_t_train], axis=0).astype(np.float32)
    med = np.nanmedian(X_fit, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)

    def impute_and_mask(X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32, copy=True)
        mask = np.isnan(X).astype(np.float32)
        idx = np.where(np.isnan(X))
        if idx[0].size:
            X[idx] = med[idx[1]]
        return np.concatenate([X, mask], axis=1)

    X_src_i = impute_and_mask(X_src)
    X_t_i = impute_and_mask(X_t_train)
    X_val_i = impute_and_mask(X_val)
    X_test_i = impute_and_mask(X_test)

    xs = StandardScaler().fit(np.concatenate([X_src_i, X_t_i], axis=0))
    ys = StandardScaler().fit(np.concatenate([y_src, y_t_train], axis=0).reshape(-1, 1))
    X_src_s = xs.transform(X_src_i).astype(np.float32)
    y_src_s = ys.transform(y_src.reshape(-1, 1)).ravel().astype(np.float32)
    X_t_s = xs.transform(X_t_i).astype(np.float32)
    y_t_s = ys.transform(y_t_train.reshape(-1, 1)).ravel().astype(np.float32)
    X_val_s = xs.transform(X_val_i).astype(np.float32)
    X_test_s = xs.transform(X_test_i).astype(np.float32)
    y_val_s = ys.transform(np.asarray(y_val_used, dtype=np.float32).reshape(-1, 1)).ravel().astype(np.float32)

    mlp_cfg = dict(model_cfg.get("MLP_TL", {}) or {})
    style = str(mlp_cfg.get("style", "small")).strip().lower()
    if style == "mdl":
        base = int(10 ** max(0, len(str(int(X_src_s.shape[1]))) - 1))
        hidden = [
            int(base * 1.0),
            int(base * 0.8),
            int(base * 0.4),
            int(base * 0.2),
            int(max(16, base * 0.1)),
        ]
    else:
        hidden = [int(x) for x in mlp_cfg.get("hidden", [256, 128, 64])]
    dropout = float(mlp_cfg.get("dropout", 0.2))

    model = _MLP(X_src_s.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(mlp_cfg["lr_pretrain"]), weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    def run_train(X_np: np.ndarray, y_np: np.ndarray, epochs: int, lr: float, batch_size: int) -> None:
        for group in opt.param_groups:
            group["lr"] = float(lr)
        X_tn = torch.tensor(X_np, dtype=torch.float32, device=device)
        y_tn = torch.tensor(y_np, dtype=torch.float32, device=device)
        n = len(X_tn)
        for _ in range(int(epochs)):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, int(batch_size)):
                batch = perm[i : i + int(batch_size)]
                pred = model(X_tn[batch])
                loss = loss_fn(pred, y_tn[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()

    run_train(
        X_src_s,
        y_src_s,
        epochs=int(mlp_cfg["epochs_pretrain"]),
        lr=float(mlp_cfg["lr_pretrain"]),
        batch_size=int(mlp_cfg["batch_size"]),
    )

    pretrained_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    X_val_tn = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    search_finetune = bool(mlp_cfg.get("search_finetune", False))
    if search_finetune:
        bs_list = [int(x) for x in mlp_cfg.get("finetune_batch_sizes", [16, 32])]
        lr_list = [float(x) for x in mlp_cfg.get("finetune_lrs", [1e-4, 5e-4, 1e-3, 5e-3])]
        ep_list = [int(x) for x in mlp_cfg.get("finetune_epochs_grid", [50, 100, 150, 200, 250, 300])]
        best_state = None
        best_mae = float("inf")
        for batch_size in bs_list:
            for lr in lr_list:
                for epochs in ep_list:
                    model.load_state_dict(pretrained_state)
                    run_train(X_t_s, y_t_s, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
                    with torch.no_grad():
                        pred_val_s = model(X_val_tn).detach().cpu().numpy()
                    mae = float(np.mean(np.abs(pred_val_s - y_val_s)))
                    if mae < best_mae:
                        best_mae = mae
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        best_state = None
        best_mae = float("inf")
        patience = int(mlp_cfg["patience"])
        stale = 0
        for _ in range(int(mlp_cfg["epochs_finetune"])):
            run_train(
                X_t_s,
                y_t_s,
                epochs=1,
                lr=float(mlp_cfg["lr_finetune"]),
                batch_size=int(mlp_cfg["batch_size"]),
            )
            with torch.no_grad():
                pred_val_s = model(X_val_tn).detach().cpu().numpy()
            mae = float(np.mean(np.abs(pred_val_s - y_val_s)))
            if mae < best_mae:
                best_mae = mae
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

    with torch.no_grad():
        val_pred_s = model(torch.tensor(X_val_s, dtype=torch.float32, device=device)).cpu().numpy()
        test_pred_s = model(torch.tensor(X_test_s, dtype=torch.float32, device=device)).cpu().numpy()
    val_pred = ys.inverse_transform(val_pred_s.reshape(-1, 1)).ravel()
    test_pred = ys.inverse_transform(test_pred_s.reshape(-1, 1)).ravel()
    val_pred_sec = _inverse_target(val_pred, target_transform, target_inv_scale, target_t0_sec)
    test_pred_sec = _inverse_target(test_pred, target_transform, target_inv_scale, target_t0_sec)
    return CandidateOutput(
        name="MLP_TL",
        val_pred=val_pred_sec,
        test_pred=test_pred_sec,
        val_metrics=compute_metrics(y_val_sec, val_pred_sec),
        model=model,
    )
