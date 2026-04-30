from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from rtbench.bench.prepare import prepare
from rtbench.config import parse_seed_range, resolve_config
from rtbench.models import random_split, stratified_split


DEFAULT_V3_WORKTREE = Path(".claude/worktrees/romantic-wilson-87e23a")
COLUMNS = [
    "dataset",
    "sheet",
    "split_id",
    "n_train",
    "n_val",
    "n_test",
    "RT_min",
    "RT_max",
    "RT_IQR",
    "dup_rate",
    "source_CP_distance",
    "residual_MAE",
    "residual_R2",
    "residual_skew",
    "film_grad_ok",
    "cp_sensitivity_p",
    "failure_tag",
    "loss_weight_rule",
]


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0.0:
        return 1.0
    return float(1.0 - np.dot(av, bv) / denom)


def _dup_rate(keys: list[Any] | np.ndarray) -> float:
    vals = [str(x).strip() for x in list(keys)]
    vals = [x for x in vals if x and x not in {"nan", "NA", "None"}]
    if not vals:
        return 0.0
    return float((len(vals) - len(set(vals))) / len(vals))


def _failure_tag(beat_both: bool, delta_mae: float, delta_r2: float) -> str:
    if beat_both:
        return "win"
    if delta_mae > -2.0 and delta_r2 > -0.01:
        return "near_miss"
    if delta_mae > -2.0 and delta_r2 <= -0.01:
        return "near_miss_r2"
    if delta_mae <= -2.0 and delta_r2 > -0.01:
        return "near_miss_mae"
    if delta_mae < -25.0 or delta_r2 < -0.15:
        return "catastrophic"
    return "hard_loss"


def _float_text(value: Any) -> str:
    try:
        x = float(value)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    return f"{x:.10g}"


def _cp_audit_values(repo_root: Path) -> tuple[str, str]:
    path = repo_root / "v4/reports/cp_audit.json"
    if not path.exists():
        return "n/a", ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "n/a", ""
    grad = data.get("film_grad_ok", "n/a")
    if isinstance(grad, bool):
        grad_text = "true" if grad else "false"
    else:
        grad_text = str(grad)
    p = data.get("cp_sensitivity_p")
    return grad_text, _float_text(p)


def _unirt_baseline(repo_root: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(repo_root / "data/baseline/unirt_sota_28.csv", dtype={"dataset": str}, encoding="utf-8")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["dataset"] = df["dataset"].astype(str).str.zfill(4)
    df = df.loc[df["mode"].astype(str).str.upper() == mode.upper()].copy()
    df = df.loc[df["method"].astype(str).str.lower() == "uni-rt"].copy()
    return df.set_index("dataset")


def _residual_skew(pred_root: Path, dataset: str, seeds: list[int]) -> float:
    residuals: list[np.ndarray] = []
    for seed in seeds:
        path = pred_root / dataset / f"seed_{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8")
        if {"y_true_sec", "y_pred_sec"}.issubset(df.columns):
            residuals.append((df["y_pred_sec"].to_numpy(dtype=np.float64) - df["y_true_sec"].to_numpy(dtype=np.float64)))
    if not residuals:
        return 0.0
    vals = np.concatenate(residuals)
    if len(vals) < 3 or np.allclose(vals, vals[0]):
        return 0.0
    out = float(stats.skew(vals, bias=False))
    return out if np.isfinite(out) else 0.0


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _sheet_rows(repo_root: Path, v3_root: Path, sheet: str, seeds: list[int]) -> list[dict[str, str]]:
    mode = "RPLC" if sheet == "S4" else "HILIC"
    out_dir = v3_root / f"outputs_v3_{sheet.lower()}"
    cfg_path = out_dir / "config.resolved.yaml"
    per_seed_path = out_dir / "metrics/per_seed.csv"
    if not cfg_path.exists() or not per_seed_path.exists():
        raise FileNotFoundError(f"Missing v3 tuning inputs for {sheet}: {out_dir}")

    resolved = resolve_config(cfg_path)
    prep = prepare(resolved.config, no_download=True)
    per_seed = pd.read_csv(per_seed_path, dtype={"dataset": str}, encoding="utf-8")
    per_seed["dataset"] = per_seed["dataset"].astype(str).str.zfill(4)
    per_seed["seed"] = per_seed["seed"].astype(int)
    present_seeds = sorted(int(x) for x in per_seed["seed"].unique())
    if any(0 <= seed <= 9 for seed in present_seeds):
        raise ValueError(f"Seed leakage in {per_seed_path}: found final seed in {present_seeds}")
    missing = sorted(set(seeds) - set(present_seeds))
    if missing:
        raise ValueError(f"{per_seed_path} missing tuning seeds: {missing}")

    baseline = _unirt_baseline(repo_root, mode)
    source_cp = [np.asarray(prep.mats[ds].X_cp[0], dtype=np.float32) for ds in prep.pretrain_ids]
    film_grad_ok, cp_sensitivity_p = _cp_audit_values(repo_root)
    split_cfg = resolved.config.split
    strategy = str(split_cfg.get("strategy", "stratified")).strip().lower()

    rows: list[dict[str, str]] = []
    for dataset in sorted(prep.external_ids):
        if dataset not in baseline.index:
            raise ValueError(f"Uni-RT baseline missing {sheet}/{dataset}")
        cur = per_seed.loc[(per_seed["dataset"] == dataset) & (per_seed["seed"].isin(seeds))].copy()
        if cur.empty:
            raise ValueError(f"No tuning rows for {sheet}/{dataset}")
        mae = float(cur["mae"].mean())
        r2 = float(cur["r2"].mean())
        uni = baseline.loc[dataset]
        delta_mae = float(uni["mae"]) - mae
        delta_r2 = r2 - float(uni["r2"])
        beat_both = bool(mae < float(uni["mae"]) and r2 > float(uni["r2"]))
        tag = _failure_tag(beat_both, delta_mae, delta_r2)
        skew = _residual_skew(out_dir / "predictions", dataset, seeds)
        target_cp = np.asarray(prep.mats[dataset].X_cp[0], dtype=np.float32)
        cp_dist = min(_cosine_distance(target_cp, src) for src in source_cp)
        mat = prep.mats[dataset]
        for seed in seeds:
            if strategy == "random":
                split = random_split(mat.y_sec, seed, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
            else:
                split = stratified_split(mat.y_sec, seed, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]))
            y_train = np.asarray(mat.y_sec, dtype=np.float64)[split.train_idx]
            rows.append(
                {
                    "dataset": dataset,
                    "sheet": sheet,
                    "split_id": f"{sheet}_{dataset}_{seed}",
                    "n_train": str(int(len(split.train_idx))),
                    "n_val": str(int(len(split.val_idx))),
                    "n_test": str(int(len(split.test_idx))),
                    "RT_min": _float_text(np.min(y_train)),
                    "RT_max": _float_text(np.max(y_train)),
                    "RT_IQR": _float_text(np.percentile(y_train, 75) - np.percentile(y_train, 25)),
                    "dup_rate": _float_text(_dup_rate(np.asarray(mat.mol_keys, dtype=object)[split.train_idx])),
                    "source_CP_distance": _float_text(cp_dist),
                    "residual_MAE": _float_text(mae),
                    "residual_R2": _float_text(r2),
                    "residual_skew": _float_text(skew),
                    "film_grad_ok": film_grad_ok,
                    "cp_sensitivity_p": cp_sensitivity_p,
                    "failure_tag": tag,
                    "loss_weight_rule": "",
                }
            )
    return rows


def build_profile(repo_root: Path, v3_root: Path, seeds: list[int]) -> list[dict[str, str]]:
    if any(0 <= seed <= 9 for seed in seeds):
        raise ValueError("L2 profile cannot use final seeds 0..9")
    if "l6" in v3_root.as_posix().lower():
        raise ValueError(f"Refusing L6-derived input path: {v3_root}")
    rows: list[dict[str, str]] = []
    for sheet in ("S4", "S5"):
        rows.extend(_sheet_rows(repo_root, v3_root, sheet, seeds))
    rows.sort(key=lambda r: (r["sheet"], r["dataset"], r["split_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked v4 L2 preregistered failure profile.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--v3-root", default=str(DEFAULT_V3_WORKTREE))
    parser.add_argument("--tuning-seeds", default="23,24,25")
    parser.add_argument("--out", default="v4/reports/preregistered_profile.csv")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    v3_root = Path(args.v3_root)
    if not v3_root.is_absolute():
        v3_root = repo_root / v3_root
    seeds = parse_seed_range(str(args.tuning_seeds).replace(",", ",")) if ":" in str(args.tuning_seeds) else [int(x.strip()) for x in str(args.tuning_seeds).split(",") if x.strip()]
    rows = build_profile(repo_root, v3_root, seeds)

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "rtbench_commit": _git_head(repo_root),
        "tuning_seeds": seeds,
        "cp_cache_path": str(repo_root / "data/repoRT/cpvec_cache"),
        "v3_outputs_root": str(v3_root),
        "row_count": len(rows),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stamp = repo_root / "v4/reports/_stamps/profile.ok"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {meta_path}")
    print(f"wrote {stamp}")


if __name__ == "__main__":
    main()
