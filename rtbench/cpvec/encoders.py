from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import _as_float, _gradient_segments, _slug, _split_words, _unit_scale


class _AE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return self.dec(z)


def _fit_norm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def _train_ae(
    X: np.ndarray,
    latent_dim: int,
    hidden: int,
    dropout: float,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int = 256,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = _fit_norm(X)
    Xn = _norm(X, mean, std)

    model = _AE(in_dim=Xn.shape[1], latent_dim=int(latent_dim), hidden=int(hidden), dropout=float(dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(Xn).to(device)
    n = len(x_t)
    for _ in range(int(epochs)):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, int(batch_size)):
            idx = perm[i : i + int(batch_size)]
            xb = x_t[idx]
            pred = model(xb)
            loss = loss_fn(pred, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, mean, std


def _build_vocab(seqs: list[list[str]], min_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in seqs:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    vocab = {"<UNK>": 0}
    for tok, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt < int(min_count):
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


def _seqs_to_ids(seqs: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    unk = int(vocab.get("<UNK>", 0))
    return [[int(vocab.get(t, unk)) for t in s] for s in seqs]


def _make_pairs(docs_ids: list[list[int]], window: int) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []
    w = int(window)
    for doc in docs_ids:
        n = len(doc)
        for i in range(n):
            c = int(doc[i])
            start = max(0, i - w)
            end = min(n, i + w + 1)
            for j in range(start, end):
                if j == i:
                    continue
                centers.append(c)
                contexts.append(int(doc[j]))
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def _train_word2vec(
    docs_ids: list[list[int]],
    vocab_size: int,
    dim: int,
    window: int,
    neg_k: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    centers, contexts = _make_pairs(docs_ids, window=window)
    counts = np.bincount(np.concatenate([np.asarray(d, dtype=np.int64) for d in docs_ids if d]), minlength=vocab_size)
    dist = counts.astype(np.float64) ** 0.75
    dist = dist / max(dist.sum(), 1.0)
    dist_t = torch.tensor(dist, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_in = nn.Embedding(int(vocab_size), int(dim)).to(device)
    emb_out = nn.Embedding(int(vocab_size), int(dim)).to(device)
    bound = 0.5 / max(1, int(dim))
    nn.init.uniform_(emb_in.weight, -bound, bound)
    nn.init.zeros_(emb_out.weight)

    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=float(lr))

    n = len(centers)
    idx = np.arange(n)
    for _ in range(int(epochs)):
        rng.shuffle(idx)
        for start in range(0, n, int(batch_size)):
            b = idx[start : start + int(batch_size)]
            c = torch.from_numpy(centers[b]).to(device)
            o = torch.from_numpy(contexts[b]).to(device)
            v_c = emb_in(c)
            v_o = emb_out(o)
            pos = (v_c * v_o).sum(dim=1)
            pos_loss = -F.logsigmoid(pos)

            neg = torch.multinomial(dist_t, num_samples=len(b) * int(neg_k), replacement=True).to(device)
            neg = neg.view(len(b), int(neg_k))
            v_n = emb_out(neg)
            neg_score = (v_n * v_c.unsqueeze(1)).sum(dim=2)
            neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

            loss = (pos_loss + neg_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    wv = (emb_in.weight.detach().cpu().numpy() + emb_out.weight.detach().cpu().numpy()) / 2.0
    return np.asarray(wv, dtype=np.float32)


@dataclass(frozen=True)
class CPEncoderArtifacts:
    col_vocab: dict[str, int]
    col_word_vecs: np.ndarray
    col_dim: int
    ae1_state: dict[str, Any]
    ae1_in_dim: int
    ae1_latent: int
    ae1_hidden: int
    ae1_dropout: float
    seg_mean: np.ndarray
    seg_std: np.ndarray
    ae2_state: dict[str, Any]
    ae2_in_dim: int
    ae2_latent: int
    ae2_hidden: int
    ae2_dropout: float
    mp_mean: np.ndarray
    mp_std: np.ndarray
    eluent_cols: list[str]
    ccp_mean: np.ndarray
    ccp_std: np.ndarray


class CPEncoder:
    def __init__(self, art: CPEncoderArtifacts, device: torch.device) -> None:
        self.col_vocab = art.col_vocab
        self.col_word_vecs = art.col_word_vecs.astype(np.float32, copy=False)
        self.col_dim = int(art.col_dim)
        self.eluent_cols = list(art.eluent_cols)
        self.seg_mean = art.seg_mean.astype(np.float32, copy=False)
        self.seg_std = art.seg_std.astype(np.float32, copy=False)
        self.mp_mean = art.mp_mean.astype(np.float32, copy=False)
        self.mp_std = art.mp_std.astype(np.float32, copy=False)
        self.ccp_mean = art.ccp_mean.astype(np.float32, copy=False)
        self.ccp_std = art.ccp_std.astype(np.float32, copy=False)

        self._device = device
        self._ae1 = _AE(
            in_dim=int(art.ae1_in_dim),
            latent_dim=int(art.ae1_latent),
            hidden=int(art.ae1_hidden),
            dropout=float(art.ae1_dropout),
        ).to(device)
        self._ae1.load_state_dict(art.ae1_state)
        self._ae1.eval()

        self._ae2 = _AE(
            in_dim=int(art.ae2_in_dim),
            latent_dim=int(art.ae2_latent),
            hidden=int(art.ae2_hidden),
            dropout=float(art.ae2_dropout),
        ).to(device)
        self._ae2.load_state_dict(art.ae2_state)
        self._ae2.eval()

        self.grad_dim = int(art.ae1_latent)
        self.mp_dim = int(art.ae2_latent)
        self.cp_dim = self.grad_dim + self.mp_dim + self.col_dim + 6

    def _encode_column_type(self, col_name: str, usp_code: str) -> np.ndarray:
        unk = int(self.col_vocab.get("<UNK>", 0))
        toks = _split_words(col_name) + [_slug(usp_code)]
        ids = [int(self.col_vocab.get(t, unk)) for t in toks]
        vec = self.col_word_vecs[np.array(ids, dtype=np.int64)]
        return vec.mean(axis=0).astype(np.float32)

    def _encode_mobile_phase(self, meta_row: pd.Series) -> np.ndarray:
        vals = []
        for c in self.eluent_cols:
            v = _as_float(meta_row.get(c, 0.0))
            if not np.isfinite(v):
                v = 0.0
            unit = meta_row.get(f"{c}.unit", "")
            v = float(v) * _unit_scale(str(unit))
            vals.append(v)
        x = np.asarray(vals, dtype=np.float32)
        x_n = (x - self.mp_mean) / np.where(self.mp_std > 1e-6, self.mp_std, 1.0)
        t = torch.from_numpy(x_n.reshape(1, -1)).to(self._device)
        with torch.no_grad():
            z = self._ae2.encode(t).detach().cpu().numpy().reshape(-1)
        return z.astype(np.float32)

    def _encode_gradient(self, gradient_df: pd.DataFrame) -> np.ndarray:
        segs = _gradient_segments(gradient_df)
        if not segs:
            return np.zeros(self.grad_dim, dtype=np.float32)
        X = np.vstack(segs).astype(np.float32)
        Xn = (X - self.seg_mean) / np.where(self.seg_std > 1e-6, self.seg_std, 1.0)
        t = torch.from_numpy(Xn).to(self._device)
        with torch.no_grad():
            z = self._ae1.encode(t).detach().cpu().numpy()
        return np.sum(z, axis=0).astype(np.float32)

    def _encode_ccp(self, meta_row: pd.Series) -> np.ndarray:
        feats = [
            _as_float(meta_row.get("column.temperature", np.nan)),
            _as_float(meta_row.get("column.flowrate", np.nan)),
            _as_float(meta_row.get("column.length", np.nan)),
            _as_float(meta_row.get("column.id", np.nan)),
            _as_float(meta_row.get("column.particle.size", np.nan)),
            _as_float(meta_row.get("column.t0", np.nan)),
        ]
        x = np.asarray([0.0 if not np.isfinite(v) else float(v) for v in feats], dtype=np.float32)
        return ((x - self.ccp_mean) / np.where(self.ccp_std > 1e-6, self.ccp_std, 1.0)).astype(np.float32)

    def cp_vector_for_dataset(self, ds_root: Path, ds: str) -> np.ndarray:
        meta_path = ds_root / f"{ds}_metadata.tsv"
        grad_path = ds_root / f"{ds}_gradient.tsv"
        if not meta_path.exists():
            return np.zeros(self.cp_dim, dtype=np.float32)
        meta = pd.read_csv(meta_path, sep="\t", encoding="utf-8")
        if meta.empty:
            return np.zeros(self.cp_dim, dtype=np.float32)
        row = meta.iloc[0]
        col_name = str(row.get("column.name", "__missing__"))
        usp = str(row.get("column.usp.code", "__missing__"))
        z_ct = self._encode_column_type(col_name, usp)
        z_mp = self._encode_mobile_phase(row)
        if grad_path.exists():
            g = pd.read_csv(grad_path, sep="\t", encoding="utf-8")
        else:
            g = pd.DataFrame()
        z_grad = self._encode_gradient(g)
        z_ccp = self._encode_ccp(row)
        out = np.concatenate([z_grad, z_mp, z_ct, z_ccp], axis=0).astype(np.float32)
        if out.size != self.cp_dim:
            raise AssertionError(f"CP dim mismatch for {ds}: got {out.size}, expected {self.cp_dim}")
        return out


__all__ = [
    "CPEncoder",
    "CPEncoderArtifacts",
    "_AE",
    "_build_vocab",
    "_fit_norm",
    "_make_pairs",
    "_norm",
    "_seqs_to_ids",
    "_train_ae",
    "_train_word2vec",
]
