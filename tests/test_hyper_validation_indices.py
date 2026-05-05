from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

import rtbench.hyper as hyper


def _make_data(n_rows: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_id = np.arange(n_rows, dtype=np.float32)
    x_mol = np.stack([row_id, row_id * 2.0 + 1.0, (row_id % 3.0) - 1.0], axis=1).astype(np.float32)
    x_cp = np.stack([row_id * -0.5, row_id + 0.25], axis=1).astype(np.float32)
    y = (row_id * 0.1 + 1.0).astype(np.float32)
    dataset_ids = np.array([f"{int(i % 3) + 1:04d}" for i in row_id], dtype=object)
    return x_mol, x_cp, y, dataset_ids


def _cfg(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "embed_dim": 3,
        "mol_hidden": 8,
        "cp_hidden": 6,
        "dropout": 0.0,
        "epochs": 1,
        "batch_size": 64,
        "lr": 0.01,
        "weight_decay": 0.0,
        "val_frac": 0.25,
        "patience": 3,
    }
    cfg.update(overrides)
    return cfg


def _install_recording_hypernet(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[np.ndarray]]:
    records: dict[str, list[np.ndarray]] = {"train_mol": [], "val_mol": [], "train_task": [], "val_task": []}

    class RecordingHyperNet(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.use_film = False

        def forward(self, mol: torch.Tensor, cp: torch.Tensor, task_idx: torch.Tensor | None = None) -> torch.Tensor:
            prefix = "train" if self.training else "val"
            records[f"{prefix}_mol"].append(mol.detach().cpu().numpy())
            if task_idx is not None:
                records[f"{prefix}_task"].append(task_idx.detach().cpu().numpy())
            return self.bias.expand(mol.shape[0])

    monkeypatch.setattr(hyper, "_HyperNet", RecordingHyperNet)
    return records


def _row_ids_from_standardized_rows(rows: np.ndarray, x_mol: np.ndarray) -> list[int]:
    mol_mean, mol_std = hyper._compute_mol_mean_std(x_mol, sequence_len=0)
    expected = hyper._standardize(x_mol, mol_mean, mol_std)
    row_ids: list[int] = []
    for row in rows:
        matches = np.flatnonzero(np.all(np.isclose(expected, row, atol=1e-6), axis=1))
        assert matches.size == 1
        row_ids.append(int(matches[0]))
    return row_ids


def test_pretrain_hyper_tl_honors_explicit_train_and_validation_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _install_recording_hypernet(monkeypatch)
    x_mol, x_cp, y, dataset_ids = _make_data()
    train_idx = np.array([7, 2, 5, 0], dtype=int)
    val_idx = np.array([1, 4, 6], dtype=int)

    bundle = hyper.pretrain_hyper_tl(
        x_mol,
        x_cp,
        y,
        cfg=_cfg(
            val_split="dataset",
            use_task_adapters=True,
            ranking_loss_weight=0.05,
            ranking_loss_temp=0.1,
        ),
        seed=17,
        sample_weights=np.linspace(0.5, 1.5, len(y), dtype=np.float32),
        dataset_ids=dataset_ids,
        train_idx=train_idx,
        val_idx=val_idx,
    )

    train_seen = np.concatenate(records["train_mol"], axis=0)
    val_seen = np.concatenate(records["val_mol"], axis=0)
    train_rows = _row_ids_from_standardized_rows(train_seen, x_mol)
    val_rows = _row_ids_from_standardized_rows(val_seen, x_mol)

    assert sorted(train_rows) == sorted(train_idx.tolist())
    assert val_rows == val_idx.tolist()
    assert bundle.task_to_index == {"0001": 0, "0002": 1, "0003": 2}


def test_pretrain_hyper_tl_default_sample_split_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _install_recording_hypernet(monkeypatch)
    x_mol, x_cp, y, _dataset_ids = _make_data()
    seed = 23

    hyper.pretrain_hyper_tl(x_mol, x_cp, y, cfg=_cfg(val_split="sample"), seed=seed)

    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(max(1, round(len(y) * 0.25)))
    expected_val = idx[:n_val].tolist()
    expected_train = idx[n_val:].tolist()

    train_seen = np.concatenate(records["train_mol"], axis=0)
    val_seen = np.concatenate(records["val_mol"], axis=0)
    assert sorted(_row_ids_from_standardized_rows(train_seen, x_mol)) == sorted(expected_train)
    assert _row_ids_from_standardized_rows(val_seen, x_mol) == expected_val


@pytest.mark.parametrize(
    ("train_idx_factory", "val_idx_factory", "match"),
    [
        (lambda n: np.array([], dtype=int), lambda n: np.array([0], dtype=int), "train_idx must be non-empty"),
        (lambda n: np.array([0, 1], dtype=int), lambda n: np.array([1, 2], dtype=int), "must be disjoint"),
        (lambda n: np.array([0, n], dtype=int), lambda n: np.array([1], dtype=int), "out of bounds"),
        (lambda n: np.array([0.0, 1.0], dtype=np.float32), lambda n: np.array([2], dtype=int), "integer"),
    ],
)
def test_pretrain_hyper_tl_rejects_invalid_explicit_indices(
    train_idx_factory: Callable[[int], np.ndarray],
    val_idx_factory: Callable[[int], np.ndarray],
    match: str,
) -> None:
    x_mol, x_cp, y, _dataset_ids = _make_data()

    with pytest.raises(ValueError, match=match):
        hyper.pretrain_hyper_tl(
            x_mol,
            x_cp,
            y,
            cfg=_cfg(),
            seed=5,
            train_idx=train_idx_factory(len(y)),
            val_idx=val_idx_factory(len(y)),
        )


def test_pretrain_hyper_tl_requires_explicit_indices_together() -> None:
    x_mol, x_cp, y, _dataset_ids = _make_data()

    with pytest.raises(ValueError, match="supplied together"):
        hyper.pretrain_hyper_tl(x_mol, x_cp, y, cfg=_cfg(), seed=5, train_idx=np.array([0, 1], dtype=int))
