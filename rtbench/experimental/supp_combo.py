from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..logging_utils import configure_logging, default_run_log_path


logger = logging.getLogger("rtbench.experimental.supp_combo")


def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"dataset": str}, encoding="utf-8")
    df["dataset"] = df["dataset"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    return df


def _bh_adjust(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine supplementary single-task runs by dataset policy")
    parser.add_argument("--policy", required=True, help="YAML with 'runs' and 'policy' sections")
    parser.add_argument("--out-dir", required=True, help="Output directory (writes comparison.csv and summary.md)")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(out_dir))
    logger.info("Starting supplementary combo policy merge.", extra={"policy_path": args.policy, "out_dir": out_dir.as_posix()})

    # This merger reads fixed policy inputs and emits combined reports only; it does
    # not rewrite per-run configs at runtime, so there is no config.resolved.yaml
    # snapshot to materialize here.
    policy_cfg = yaml.safe_load(Path(args.policy).read_text(encoding="utf-8"))
    runs = dict(policy_cfg.get("runs", {}) or {})
    policy = {str(k).zfill(4): str(v) for k, v in dict(policy_cfg.get("policy", {}) or {}).items()}
    if not runs or not policy:
        raise ValueError("policy yaml must include non-empty 'runs' and 'policy'")

    run_df = {name: _load_df(path) for name, path in runs.items()}
    run_idx = {name: df.set_index("dataset") for name, df in run_df.items()}

    rows: list[dict[str, object]] = []
    for ds, pick in sorted(policy.items()):
        if pick not in run_idx:
            raise ValueError(f"Unknown run key '{pick}' for dataset {ds}")
        cur = run_idx[pick]
        if ds not in cur.index:
            raise ValueError(f"Dataset {ds} missing in picked run '{pick}'")
        src = cur.loc[ds]
        # Baseline columns (e.g., Uni_RT_mae / MDL_TL_r2) are sheet-level constants;
        # when combining subset runs, read them from any run that contains this dataset.
        base = None
        for _rn, rdf in run_idx.items():
            if ds in rdf.index:
                base = rdf.loc[ds]
                break
        if base is None:
            raise ValueError(f"Dataset {ds} not found in any run dataframe")

        rec: dict[str, object] = {
            "dataset": ds,
            "picked_run": pick,
            "our_mae_mean": float(src["our_mae_mean"]),
            "our_r2_mean": float(src["our_r2_mean"]),
        }
        for c in ("p_mae", "p_r2"):
            if c in src.index:
                try:
                    rec[c] = float(src[c])
                except Exception:
                    pass
        for m in ("Uni_RT", "MDL_TL", "GNN_RT", "DeepGCN_RT"):
            mae_col = f"{m}_mae"
            r2_col = f"{m}_r2"
            if mae_col not in base.index or r2_col not in base.index:
                continue
            rec[mae_col] = float(base[mae_col])
            rec[r2_col] = float(base[r2_col])
            rec[f"better_both_vs_{m}"] = bool(
                (rec["our_mae_mean"] < rec[mae_col]) and (rec["our_r2_mean"] > rec[r2_col])
            )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "p_mae" in out.columns:
        out["p_adj_mae"] = _bh_adjust(out["p_mae"].astype(float).tolist())
    if "p_r2" in out.columns:
        out["p_adj_r2"] = _bh_adjust(out["p_r2"].astype(float).tolist())
    out_csv = out_dir / "comparison.csv"
    out_md = out_dir / "summary.md"
    out.to_csv(out_csv, index=False, encoding="utf-8")

    lines = [
        "# Supplement Combo Summary",
        "",
        f"- Datasets: {len(out)}",
        f"- Our avg MAE: {out['our_mae_mean'].mean():.4f}",
        f"- Our avg R2: {out['our_r2_mean'].mean():.4f}",
    ]
    for m in ("Uni_RT", "MDL_TL", "GNN_RT", "DeepGCN_RT"):
        col = f"better_both_vs_{m}"
        if col in out.columns:
            lines.append(f"- better both vs {m}: {int(out[col].sum())}/{len(out)}")
    lines.append("")
    lines.append("## Dataset Policy")
    for ds, pick in sorted(policy.items()):
        lines.append(f"- {ds}: {pick}")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "Supplementary combo outputs written.",
        extra={
            "comparison_csv": out_csv.as_posix(),
            "summary_md": out_md.as_posix(),
            "dataset_count": len(out),
        },
    )


if __name__ == "__main__":
    main()
