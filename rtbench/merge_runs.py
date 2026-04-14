from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

from .experiments import record_experiment, status_for_run_dir
from .logging_utils import configure_logging, default_run_log_path
from .report import write_report
from .stats import summarize_vs_paper


logger = logging.getLogger("rtbench.merge_runs")


def parse_override(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Invalid override '{text}', expected DATASET=RUN_ROOT")
    ds, root = text.split("=", 1)
    ds = str(ds).strip().zfill(4)
    return ds, Path(root.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge benchmark runs with per-dataset overrides")
    parser.add_argument("--base", required=True, help="Base output root directory")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dataset override in the form DATASET=RUN_ROOT (can be used multiple times)",
    )
    parser.add_argument("--out", required=True, help="Merged output root")
    parser.add_argument("--baseline-csv", required=True, help="Paper baseline CSV path")
    parser.add_argument("--paper-avg-mae", type=float, required=True)
    parser.add_argument("--paper-avg-r2", type=float, required=True)
    parser.add_argument("--required-win-both", type=int, required=True)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    args = parser.parse_args()

    base_root = Path(args.base)
    out_root = Path(args.out)
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(out_root))
    logger.info(
        "Starting merged run assembly.",
        extra={"base_root": base_root.as_posix(), "out_root": out_root.as_posix(), "override_count": len(args.override)},
    )
    metrics_root = out_root / "metrics"
    pred_root = out_root / "predictions"
    metrics_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    base_per_seed = pd.read_csv(base_root / "metrics" / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
    base_per_seed["dataset"] = base_per_seed["dataset"].astype(str).str.zfill(4)

    overrides = {}
    for item in args.override:
        ds, root = parse_override(item)
        overrides[ds] = root

    parts = [base_per_seed.loc[~base_per_seed["dataset"].isin(set(overrides.keys()))].copy()]
    for ds, root in sorted(overrides.items()):
        df = pd.read_csv(root / "metrics" / "per_seed.csv", dtype={"dataset": str}, encoding="utf-8")
        df["dataset"] = df["dataset"].astype(str).str.zfill(4)
        sel = df.loc[df["dataset"] == ds].copy()
        if sel.empty:
            raise ValueError(f"Override run '{root}' has no rows for dataset '{ds}'")
        parts.append(sel)

    merged = pd.concat(parts, ignore_index=True)
    merged = (
        merged.drop_duplicates(subset=["dataset", "seed"], keep="last")
        .sort_values(["dataset", "seed"])
        .reset_index(drop=True)
    )
    merged.to_csv(metrics_root / "per_seed.csv", index=False, encoding="utf-8")

    baseline_df = pd.read_csv(args.baseline_csv, dtype={"dataset": str}, encoding="utf-8")
    baseline_df["dataset"] = baseline_df["dataset"].astype(str).str.zfill(4)
    summary = summarize_vs_paper(per_seed_df=merged, baseline_df=baseline_df, fdr_q=float(args.fdr_q))
    summary = summary[
        [
            "dataset",
            "paper_mae",
            "paper_r2",
            "our_mae_mean",
            "our_r2_mean",
            "delta_mae",
            "delta_r2",
            "p_mae",
            "p_r2",
            "p_adj_mae",
            "p_adj_r2",
            "win_both",
        ]
    ]
    summary.to_csv(metrics_root / "summary_vs_paper.csv", index=False, encoding="utf-8")

    # Copy predictions for all datasets from base, then overwrite override datasets.
    if (base_root / "predictions").exists():
        for ds_dir in (base_root / "predictions").glob("*"):
            if not ds_dir.is_dir():
                continue
            ds = ds_dir.name
            if ds in overrides:
                continue
            dst = pred_root / ds
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ds_dir, dst)
    for ds, root in sorted(overrides.items()):
        src = root / "predictions" / ds
        if not src.exists():
            continue
        dst = pred_root / ds
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    write_report(
        out_path=out_root / "report.md",
        per_seed=merged,
        summary=summary,
        baseline_avg_mae=float(args.paper_avg_mae),
        baseline_avg_r2=float(args.paper_avg_r2),
        success_win_required=int(args.required_win_both),
        feature_importance_failed={},
    )

    avg_mae = float(summary["our_mae_mean"].mean())
    avg_r2 = float(summary["our_r2_mean"].mean())
    wins = int(summary["win_both"].sum())
    success = (avg_mae < float(args.paper_avg_mae)) and (avg_r2 > float(args.paper_avg_r2)) and (
        wins >= int(args.required_win_both)
    )
    record_experiment(
        Path.cwd(),
        run_dir=out_root,
        status=status_for_run_dir(out_root, has_summary=True),
        summary_df=summary,
        config_source="merge",
        extra_hparams={
            "merge.base": str(args.base),
            "merge.override_count": len(overrides),
            "merge.overrides": sorted(args.override),
            "stats.fdr_q": float(args.fdr_q),
            "metrics.required_win_both": int(args.required_win_both),
        },
    )
    logger.info(
        "Merged run completed: avg_mae=%.6f avg_r2=%.6f win_both=%d success=%s",
        avg_mae,
        avg_r2,
        wins,
        bool(success),
        extra={
            "out_root": out_root.as_posix(),
            "avg_mae": float(avg_mae),
            "avg_r2": float(avg_r2),
            "win_both": int(wins),
            "success": bool(success),
            "summary_csv": (metrics_root / "summary_vs_paper.csv").as_posix(),
        },
    )


if __name__ == "__main__":
    main()
