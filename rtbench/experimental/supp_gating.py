from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..logging_utils import configure_logging, default_run_log_path


logger = logging.getLogger("rtbench.experimental.supp_gating")


BASELINE_MODELS = ("Uni_RT", "MDL_TL", "GNN_RT", "DeepGCN_RT")
META_NUM_COLS = (
    "column.length",
    "column.id",
    "column.particle.size",
    "column.temperature",
    "column.flowrate",
    "column.t0",
    "gradient.start.A",
    "gradient.start.B",
    "gradient.start.C",
    "gradient.start.D",
    "gradient.end.A",
    "gradient.end.B",
    "gradient.end.C",
    "gradient.end.D",
)


def _resolve_processed_root(policy_cfg: dict[str, Any], cli_processed_root: str) -> Path:
    if str(cli_processed_root).strip():
        return Path(str(cli_processed_root))
    data_cfg = dict(policy_cfg.get("data", {}) or {})
    processed_root = str(data_cfg.get("processed_root", "")).strip()
    if processed_root:
        return Path(processed_root)
    local_root = str(data_cfg.get("local_root", "")).strip()
    if local_root:
        return Path(local_root) / "processed_data"
    return Path("data/repoRT/processed_data")


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df["dataset"] = df["dataset"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    return df


def _read_num(v: Any) -> float:
    try:
        out = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return 0.0 if pd.isna(out) else float(out)
    except Exception:
        return 0.0


def _merge_rt_tables(processed_root: Path, ds: str) -> pd.DataFrame:
    ds_root = processed_root / ds
    can_p = ds_root / f"{ds}_rtdata_canonical_success.tsv"
    iso_p = ds_root / f"{ds}_rtdata_isomeric_success.tsv"
    can = pd.read_csv(can_p, sep="\t", encoding="utf-8") if can_p.exists() else pd.DataFrame(columns=["id", "rt"])
    iso = pd.read_csv(iso_p, sep="\t", encoding="utf-8") if iso_p.exists() else pd.DataFrame(columns=["id", "rt"])
    if "id" not in can.columns:
        can = pd.DataFrame(columns=["id", "rt"])
    if "id" not in iso.columns:
        iso = pd.DataFrame(columns=["id", "rt"])
    can = can.drop_duplicates(subset="id", keep="first")
    iso = iso.drop_duplicates(subset="id", keep="first")
    add = iso.loc[~iso["id"].isin(can["id"])]
    return pd.concat([can, add], ignore_index=True)


def _dataset_features(processed_root: Path, ds: str) -> dict[str, Any]:
    ds_root = processed_root / ds
    meta_p = ds_root / f"{ds}_metadata.tsv"
    grad_p = ds_root / f"{ds}_gradient.tsv"
    if not (meta_p.exists() and grad_p.exists()):
        return {}

    meta = pd.read_csv(meta_p, sep="\t", encoding="utf-8")
    if meta.empty:
        return {}
    meta_row = meta.iloc[0]

    grad = pd.read_csv(grad_p, sep="\t", encoding="utf-8")
    time_col = grad.columns[0] if not grad.empty else ""
    grad_end = _read_num(pd.to_numeric(grad[time_col], errors="coerce").max()) if time_col else 0.0
    grad_steps = float(len(grad))

    rt = _merge_rt_tables(processed_root=processed_root, ds=ds)
    y_min = pd.to_numeric(rt["rt"], errors="coerce").dropna().to_numpy(dtype=float) if "rt" in rt.columns else np.array([])

    out: dict[str, Any] = {
        "feat_rt_count": float(len(y_min)),
        "feat_rt_mean_min": float(np.mean(y_min)) if len(y_min) else 0.0,
        "feat_rt_std_min": float(np.std(y_min)) if len(y_min) else 0.0,
        "feat_rt_median_min": float(np.median(y_min)) if len(y_min) else 0.0,
        "feat_gradient_steps": grad_steps,
        "feat_gradient_end_min": grad_end,
        "feat_column_name": str(meta_row.get("column.name", "__MISSING__")),
        "feat_column_usp": str(meta_row.get("column.usp.code", "__MISSING__")),
    }
    for c in META_NUM_COLS:
        out[f"feat_{c.replace('.', '_')}"] = _read_num(meta_row.get(c, 0.0))
    return out


def _baseline_features(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in row.index:
        lc = str(c).lower()
        if lc.startswith("our_") or lc.startswith("delta_") or lc.startswith("p_") or lc.startswith("better_"):
            continue
        if lc == "win_both":
            continue
        if lc.endswith("_mae") or lc.endswith("_r2"):
            out[f"feat_{c}"] = _read_num(row[c])
    return out


def _utility(
    row: pd.Series,
    *,
    baseline_model: str,
    weight_win: float,
    weight_mae: float,
    weight_r2: float,
) -> float:
    b_mae_col = f"{baseline_model}_mae"
    b_r2_col = f"{baseline_model}_r2"
    if b_mae_col not in row.index or b_r2_col not in row.index:
        raise ValueError(f"Missing baseline columns '{b_mae_col}'/'{b_r2_col}' in run table")
    base_mae = _read_num(row[b_mae_col])
    base_r2 = _read_num(row[b_r2_col])
    mae = _read_num(row.get("our_mae_mean", np.nan))
    r2 = _read_num(row.get("our_r2_mean", np.nan))
    win = 1.0 if (mae < base_mae and r2 > base_r2) else 0.0
    mae_rel = (base_mae - mae) / max(abs(base_mae), 1e-6)
    r2_delta = (r2 - base_r2)
    return float(weight_win * win + weight_mae * mae_rel + weight_r2 * r2_delta)


def _choose_best_available(
    classes: np.ndarray,
    probs: np.ndarray,
    available: set[str],
    fallback: str,
    id_to_run: dict[int, str],
) -> str:
    order = np.argsort(-np.asarray(probs, dtype=float))
    for i in order.tolist():
        raw = classes[int(i)]
        run = id_to_run.get(int(raw), str(raw))
        if run in available:
            return run
    return fallback


def _mode_label_ids(labels: np.ndarray, available: set[str], fallback: str, id_to_run: dict[int, str]) -> str:
    if labels.size == 0:
        return fallback
    s = pd.Series(labels.astype(int).tolist(), dtype=int)
    for v in s.value_counts().index.tolist():
        run = id_to_run.get(int(v), "")
        if run in available:
            return run
    return fallback


@dataclass
class PolicyEval:
    avg_mae: float
    avg_r2: float
    better_both: dict[str, int]
    n: int
    table: pd.DataFrame


@dataclass
class CandidateSpec:
    name: str
    feature_set: str
    estimator: Any


def _evaluate_policy(policy: dict[str, str], run_idx: dict[str, pd.DataFrame]) -> PolicyEval:
    rows: list[dict[str, Any]] = []
    for ds, pick in sorted(policy.items()):
        src = run_idx[pick].loc[ds]
        base = None
        for rdf in run_idx.values():
            if ds in rdf.index:
                base = rdf.loc[ds]
                break
        if base is None:
            continue
        rec: dict[str, Any] = {
            "dataset": ds,
            "picked_run": pick,
            "our_mae_mean": _read_num(src.get("our_mae_mean")),
            "our_r2_mean": _read_num(src.get("our_r2_mean")),
        }
        for m in BASELINE_MODELS:
            mae_col = f"{m}_mae"
            r2_col = f"{m}_r2"
            if mae_col in base.index and r2_col in base.index:
                rec[mae_col] = _read_num(base[mae_col])
                rec[r2_col] = _read_num(base[r2_col])
                rec[f"better_both_vs_{m}"] = bool(
                    (rec["our_mae_mean"] < rec[mae_col]) and (rec["our_r2_mean"] > rec[r2_col])
                )
        rows.append(rec)
    out = pd.DataFrame(rows)
    better_both = {m: int(out[f"better_both_vs_{m}"].sum()) for m in BASELINE_MODELS if f"better_both_vs_{m}" in out}
    return PolicyEval(
        avg_mae=float(out["our_mae_mean"].mean()) if not out.empty else float("nan"),
        avg_r2=float(out["our_r2_mean"].mean()) if not out.empty else float("nan"),
        better_both=better_both,
        n=int(len(out)),
        table=out,
    )


def _build_feature_views(train_feat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feat_cols = [c for c in train_feat.columns if c.startswith("feat_")]
    base_prefixes = tuple(f"feat_{m}_" for m in BASELINE_MODELS)
    base_cols = [c for c in feat_cols if c.startswith(base_prefixes)]
    meta_cols = [c for c in feat_cols if c not in base_cols]
    num_cols = [c for c in feat_cols if train_feat[c].dtype != object]
    base_num_cols = [c for c in base_cols if train_feat[c].dtype != object]
    meta_num_cols = [c for c in meta_cols if train_feat[c].dtype != object]

    views: dict[str, pd.DataFrame] = {}
    feature_sets: dict[str, list[str]] = {
        "all": feat_cols,
        "all_numeric": num_cols,
        "base_numeric": base_num_cols,
        "meta_numeric": meta_num_cols,
    }
    for name, cols in feature_sets.items():
        if not cols:
            continue
        x = pd.get_dummies(train_feat[cols], columns=[c for c in cols if train_feat[c].dtype == object], dummy_na=False)
        x = x.fillna(0.0).astype(float)
        views[name] = x
    return views


def _predict_one(
    model: Any,
    x_row: pd.DataFrame,
    *,
    available: set[str],
    fallback: str,
    id_to_run: dict[int, str],
) -> str:
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_row)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                last = list(model.named_steps.values())[-1]
                classes = getattr(last, "classes_", None)
            if classes is not None:
                pick = _choose_best_available(
                    classes=np.asarray(classes),
                    probs=np.asarray(probs, dtype=float),
                    available=available,
                    fallback=fallback,
                    id_to_run=id_to_run,
                )
                if pick in available:
                    return pick
        raw = model.predict(x_row)[0]
        run = id_to_run.get(int(raw), str(raw))
        return run if run in available else fallback
    except Exception:
        return fallback


def _build_candidates(max_trees: int, seed: int, feature_views: dict[str, pd.DataFrame]) -> list[CandidateSpec]:
    trees = sorted(set([max(100, int(max_trees // 4)), int(max_trees)]))
    out: list[CandidateSpec] = []
    if "all" in feature_views:
        for n in trees:
            out.append(
                CandidateSpec(
                    name=f"rf_{n}_all",
                    feature_set="all",
                    estimator=RandomForestClassifier(
                        n_estimators=int(n),
                        random_state=int(seed),
                        class_weight="balanced_subsample",
                    ),
                )
            )
            out.append(
                CandidateSpec(
                    name=f"et_{n}_all",
                    feature_set="all",
                    estimator=ExtraTreesClassifier(
                        n_estimators=int(n),
                        random_state=int(seed),
                        class_weight="balanced_subsample",
                    ),
                )
            )

    for fs in ("all", "all_numeric", "base_numeric", "meta_numeric"):
        if fs not in feature_views:
            continue
        for n_neighbors in (1, 2, 3):
            for p in (1, 2):
                for weights in ("uniform", "distance"):
                    out.append(
                        CandidateSpec(
                            name=f"knn_{fs}_k{n_neighbors}_p{p}_{weights}",
                            feature_set=fs,
                            estimator=Pipeline(
                                [
                                    ("scaler", StandardScaler()),
                                    (
                                        "knn",
                                        KNeighborsClassifier(
                                            n_neighbors=int(n_neighbors),
                                            weights=str(weights),
                                            p=int(p),
                                        ),
                                    ),
                                ]
                            ),
                        )
                    )
    return out


def _oof_predict_candidate(
    *,
    spec: CandidateSpec,
    x: pd.DataFrame,
    y_ids: np.ndarray,
    datasets: list[str],
    available_map: dict[str, set[str]],
    id_to_run: dict[int, str],
) -> list[str]:
    pred: list[str] = []
    n = len(datasets)
    for i, ds in enumerate(datasets):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        y_tr = y_ids[mask]
        available = available_map.get(ds, set())
        if not available:
            pred.append("")
            continue
        fallback = sorted(available)[0]
        if len(np.unique(y_tr)) <= 1:
            pred.append(_mode_label_ids(labels=y_tr, available=available, fallback=fallback, id_to_run=id_to_run))
            continue
        mdl = clone(spec.estimator)
        try:
            mdl.fit(x.loc[mask, :], y_tr)
        except Exception:
            pred.append(fallback)
            continue
        pred.append(_predict_one(mdl, x.iloc[[i]], available=available, fallback=fallback, id_to_run=id_to_run))
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dataset-level gating selector for supplementary combo policy")
    parser.add_argument("--policy", required=True, help="YAML with 'runs' section. Existing 'policy' is optional.")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--processed-root",
        default="",
        help="Optional RepoRT processed_data root. Defaults to policy.data.processed_root or policy.data.local_root/processed_data.",
    )
    parser.add_argument("--baseline-model", default="MDL_TL", help="Baseline model to optimize against (default: MDL_TL)")
    parser.add_argument("--weight-win", type=float, default=4.0, help="Utility weight for better-both indicator")
    parser.add_argument("--weight-mae", type=float, default=1.0, help="Utility weight for relative MAE improvement")
    parser.add_argument("--weight-r2", type=float, default=1.0, help="Utility weight for R2 improvement")
    parser.add_argument("--n-estimators", type=int, default=800, help="Max tree count used in RF/ET candidates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(out_dir))
    # This CLI consumes a fixed policy yaml plus CLI args and does not synthesize
    # per-dataset runtime configs, so there is no supplementary config.resolved.yaml
    # snapshot to persist here.
    policy_cfg = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    processed_root = _resolve_processed_root(policy_cfg=policy_cfg, cli_processed_root=args.processed_root)
    logger.info(
        "Starting supplementary gating run.",
        extra={
            "policy_path": args.policy,
            "out_dir": out_dir.as_posix(),
            "baseline_model": args.baseline_model,
            "processed_root": processed_root.as_posix(),
            "seed": int(args.seed),
        },
    )
    runs = dict(policy_cfg.get("runs", {}) or {})
    if not runs:
        raise ValueError("policy yaml must include non-empty 'runs'")

    run_df = {name: _load_df(path) for name, path in runs.items()}
    run_idx = {name: df.set_index("dataset") for name, df in run_df.items()}

    datasets = sorted(set().union(*[set(df["dataset"].tolist()) for df in run_df.values()]))
    run_names = list(runs.keys())
    feat_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    available_map: dict[str, set[str]] = {}

    for ds in datasets:
        available = {rn for rn, rdf in run_idx.items() if ds in rdf.index}
        if not available:
            continue
        available_map[ds] = set(available)

        base_row = None
        for rn in run_names:
            rdf = run_idx[rn]
            if ds in rdf.index:
                base_row = rdf.loc[ds]
                break
        if base_row is None:
            continue

        feat = {"dataset": ds}
        feat.update(_baseline_features(base_row))
        feat.update(_dataset_features(processed_root=processed_root, ds=ds))
        feat_rows.append(feat)

        best_u = -1e18
        best_run = ""
        u_rec: dict[str, Any] = {"dataset": ds}
        for rn in run_names:
            if rn not in available:
                u_rec[f"utility_{rn}"] = np.nan
                continue
            row = run_idx[rn].loc[ds]
            u = _utility(
                row,
                baseline_model=str(args.baseline_model),
                weight_win=float(args.weight_win),
                weight_mae=float(args.weight_mae),
                weight_r2=float(args.weight_r2),
            )
            u_rec[f"utility_{rn}"] = u
            if u > best_u:
                best_u = u
                best_run = rn
        if not best_run:
            best_run = sorted(available)[0]
        u_rec["best_run_oracle"] = best_run
        u_rec["best_utility_oracle"] = float(best_u)
        utility_rows.append(u_rec)

    feat_df = pd.DataFrame(feat_rows).sort_values("dataset").reset_index(drop=True)
    util_df = pd.DataFrame(utility_rows).sort_values("dataset").reset_index(drop=True)
    train_df = feat_df.merge(util_df[["dataset", "best_run_oracle", "best_utility_oracle"]], on="dataset", how="inner")
    if train_df.empty:
        raise ValueError("No trainable datasets assembled from runs")

    ds_order = train_df["dataset"].astype(str).tolist()
    y_oracle = train_df["best_run_oracle"].astype(str).to_numpy()
    run_to_id = {rn: i for i, rn in enumerate(run_names)}
    id_to_run = {i: rn for rn, i in run_to_id.items()}
    y_ids = np.asarray([run_to_id[str(v)] for v in y_oracle], dtype=int)

    x_views = _build_feature_views(train_feat=train_df)
    if not x_views:
        raise ValueError("No feature views are available for gating")

    candidates = _build_candidates(max_trees=int(args.n_estimators), seed=int(args.seed), feature_views=x_views)
    if not candidates:
        raise ValueError("No candidate selectors available")
    logger.info(
        "Prepared supplementary gating search space.",
        extra={
            "dataset_count": len(ds_order),
            "run_count": len(run_names),
            "feature_view_count": len(x_views),
            "candidate_count": len(candidates),
        },
    )

    search_rows: list[dict[str, Any]] = []
    best_spec: CandidateSpec | None = None
    best_oof_pred: list[str] = []
    best_score: tuple[float, float, float, float, str] | None = None
    for spec in candidates:
        x = x_views[spec.feature_set]
        oof_pred = _oof_predict_candidate(
            spec=spec,
            x=x,
            y_ids=y_ids,
            datasets=ds_order,
            available_map=available_map,
            id_to_run=id_to_run,
        )
        policy_oof = {ds: run for ds, run in zip(ds_order, oof_pred)}
        ev = _evaluate_policy(policy_oof, run_idx=run_idx)
        win = int(ev.better_both.get(str(args.baseline_model), 0))
        acc = float(np.mean(np.asarray(oof_pred, dtype=object) == y_oracle))
        score = (float(win), float(-ev.avg_mae), float(ev.avg_r2), float(acc), str(spec.name))
        search_rows.append(
            {
                "candidate": spec.name,
                "feature_set": spec.feature_set,
                "oof_better_both_vs_baseline": win,
                "oof_avg_mae": float(ev.avg_mae),
                "oof_avg_r2": float(ev.avg_r2),
                "oof_label_accuracy": acc,
            }
        )
        if (best_score is None) or (score > best_score):
            best_score = score
            best_spec = spec
            best_oof_pred = oof_pred

    if best_spec is None:
        raise RuntimeError("Failed to select a gating candidate")

    x_best = x_views[best_spec.feature_set]
    final_model = clone(best_spec.estimator)
    final_model.fit(x_best, y_ids)
    in_pred: list[str] = []
    for i, ds in enumerate(ds_order):
        available = available_map.get(ds, set())
        if not available:
            in_pred.append("")
            continue
        fallback = sorted(available)[0]
        in_pred.append(
            _predict_one(
                final_model,
                x_best.iloc[[i]],
                available=available,
                fallback=fallback,
                id_to_run=id_to_run,
            )
        )

    oof_pred = best_oof_pred

    decisions = train_df[["dataset", "best_run_oracle", "best_utility_oracle"]].copy()
    decisions["pred_run_oof"] = oof_pred
    decisions["pred_run_final"] = in_pred
    decisions["correct_oof"] = decisions["pred_run_oof"] == decisions["best_run_oracle"]
    decisions["correct_final"] = decisions["pred_run_final"] == decisions["best_run_oracle"]
    decisions["selected_candidate"] = str(best_spec.name)
    decisions["selected_feature_set"] = str(best_spec.feature_set)
    util_cols = [c for c in util_df.columns if c.startswith("utility_")]
    if util_cols:
        decisions = decisions.merge(util_df[["dataset"] + util_cols], on="dataset", how="left")
    decisions.to_csv(out_dir / "gating_decisions.csv", index=False, encoding="utf-8")

    search_df = pd.DataFrame(search_rows).sort_values(
        ["oof_better_both_vs_baseline", "oof_avg_mae", "oof_avg_r2", "oof_label_accuracy"],
        ascending=[False, True, False, False],
    )
    search_df.to_csv(out_dir / "gating_search.csv", index=False, encoding="utf-8")
    feat_df.to_csv(out_dir / "gating_dataset_features.csv", index=False, encoding="utf-8")

    policy_oof = {ds: run for ds, run in zip(ds_order, oof_pred)}
    policy_final = {ds: run for ds, run in zip(ds_order, in_pred)}
    eval_oof = _evaluate_policy(policy_oof, run_idx=run_idx)
    eval_final = _evaluate_policy(policy_final, run_idx=run_idx)

    out_yaml = {
        "runs": runs,
        "policy": {str(ds): str(policy_final[ds]) for ds in ds_order},
        "meta": {
            "generated_by": "rtbench.experimental.supp_gating",
            "source_policy": str(args.policy),
            "baseline_model": str(args.baseline_model),
            "weights": {
                "win": float(args.weight_win),
                "mae": float(args.weight_mae),
                "r2": float(args.weight_r2),
            },
            "selected_candidate": str(best_spec.name),
            "selected_feature_set": str(best_spec.feature_set),
            "n_datasets": int(len(ds_order)),
            "oof_label_accuracy": float(decisions["correct_oof"].mean()),
            "final_label_accuracy": float(decisions["correct_final"].mean()),
            "oof_avg_mae": float(eval_oof.avg_mae),
            "oof_avg_r2": float(eval_oof.avg_r2),
            "final_avg_mae": float(eval_final.avg_mae),
            "final_avg_r2": float(eval_final.avg_r2),
            "oof_better_both_vs_MDL_TL": int(eval_oof.better_both.get("MDL_TL", 0)),
            "final_better_both_vs_MDL_TL": int(eval_final.better_both.get("MDL_TL", 0)),
        },
    }
    with open(out_dir / "policy.auto.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, allow_unicode=False)

    lines = [
        "# Supplement Gating Summary",
        "",
        f"- Source policy/runs: `{args.policy}`",
        f"- Baseline model: `{args.baseline_model}`",
        f"- Datasets: {len(ds_order)}",
        f"- Selected candidate: `{best_spec.name}`",
        f"- Selected feature set: `{best_spec.feature_set}`",
        f"- OOF label accuracy: {decisions['correct_oof'].mean():.4f}",
        f"- Final label accuracy: {decisions['correct_final'].mean():.4f}",
        "",
        "## OOF Policy Quality",
        f"- Avg MAE: {eval_oof.avg_mae:.4f}",
        f"- Avg R2: {eval_oof.avg_r2:.4f}",
    ]
    for m, cnt in eval_oof.better_both.items():
        lines.append(f"- better both vs {m}: {cnt}/{eval_oof.n}")
    lines.extend(
        [
            "",
            "## Final Policy Quality",
            f"- Avg MAE: {eval_final.avg_mae:.4f}",
            f"- Avg R2: {eval_final.avg_r2:.4f}",
        ]
    )
    for m, cnt in eval_final.better_both.items():
        lines.append(f"- better both vs {m}: {cnt}/{eval_final.n}")
    lines.extend(
        [
            "",
            "## Output Files",
            f"- `{(out_dir / 'policy.auto.yaml').as_posix()}`",
            f"- `{(out_dir / 'gating_decisions.csv').as_posix()}`",
            f"- `{(out_dir / 'gating_dataset_features.csv').as_posix()}`",
            f"- `{(out_dir / 'gating_search.csv').as_posix()}`",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "Supplementary gating outputs written.",
        extra={
            "policy_yaml": (out_dir / "policy.auto.yaml").as_posix(),
            "summary_md": (out_dir / "summary.md").as_posix(),
            "gating_decisions_csv": (out_dir / "gating_decisions.csv").as_posix(),
            "gating_search_csv": (out_dir / "gating_search.csv").as_posix(),
            "oof_avg_mae": float(eval_oof.avg_mae),
            "oof_avg_r2": float(eval_oof.avg_r2),
            "final_avg_mae": float(eval_final.avg_mae),
            "final_avg_r2": float(eval_final.avg_r2),
        },
    )


if __name__ == "__main__":
    main()
