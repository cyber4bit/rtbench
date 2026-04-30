from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .bench import config_sha1_from_raw, normalize_target_transform, parse_list_expr, prepare, run_trial
from .config import parse_seed_range, resolve_config
from .experiments import record_experiment, status_for_run_dir, write_effective_config_snapshot
from .logging_utils import configure_logging, default_run_log_path


logger = logging.getLogger("rtbench.run")


def main() -> None:
    parser = argparse.ArgumentParser(description="RepoRT 14+14 transfer-learning benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seeds", default="", help="Override seeds, e.g. 0:29 or 0,1,2")
    parser.add_argument("--datasets", default="", help="Override external datasets (comma-separated)")
    parser.add_argument(
        "--eval-datasets",
        default="",
        help="Evaluate only these external datasets while preparing the full configured external source pool.",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=[],
        help="Override config values, e.g. --override models.FUSION_TOP_K=8 transfer_weights.adaptive_source=true",
    )
    parser.add_argument("--no-download", action="store_true", help="Use local data only, no network download")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    override_exprs = [item for group in args.override for item in group]
    resolved = resolve_config(args.config, overrides=override_exprs)
    cfg = resolved.config
    raw_cfg = resolved.raw
    seed_expr = args.seeds if args.seeds else str(cfg.seeds["default"])
    seeds = parse_seed_range(seed_expr)
    run_dir = Path(cfg.outputs["root"])
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(run_dir))

    external_ids = None
    if args.datasets:
        external_ids = [str(x).zfill(4) for x in parse_list_expr(args.datasets)]
    eval_external_ids = None
    if args.eval_datasets:
        eval_external_ids = [str(x).zfill(4) for x in parse_list_expr(args.eval_datasets)]

    logger.info(
        "Starting benchmark run.",
        extra={
            "config_path": args.config,
            "run_dir": run_dir.as_posix(),
            "seed_expr": seed_expr,
            "dataset_override_count": 0 if external_ids is None else len(external_ids),
            "eval_dataset_override_count": 0 if eval_external_ids is None else len(eval_external_ids),
            "override_count": len(override_exprs),
            "no_download": bool(args.no_download),
        },
    )
    target_transform = normalize_target_transform(cfg.transfer_weights)
    if target_transform == "gradient_norm":
        logger.info("Target transform: gradient_norm (train on rt_sec / gradient_end_sec, score in seconds).")
    elif target_transform == "logk":
        logger.info("Target transform: logk (train on log((rt-t0)/t0), score in seconds).")
    elif target_transform == "log1p":
        logger.info("Target transform: log1p (train on log1p(rt_sec), score in seconds).")

    cfg_sha1 = config_sha1_from_raw(raw_cfg)
    write_effective_config_snapshot(Path.cwd(), run_dir=run_dir, config_raw=raw_cfg, config_sha1=cfg_sha1)
    try:
        logger.info("[1/3] Preparing data and features.")
        prep = prepare(cfg, external_ids=external_ids, no_download=bool(args.no_download))
        external_pool_ids = getattr(prep, "external_ids", None)
        if external_pool_ids is None:
            external_pool_ids = external_ids

        logger.info("[2/3] Running training and evaluation.")
        res = run_trial(
            prep,
            cfg,
            seeds=seeds,
            external_ids=eval_external_ids if eval_external_ids is not None else external_ids,
            external_pool_ids=external_pool_ids,
            config_sha1=cfg_sha1,
            early_stop=False,
            write_predictions=True,
        )
    except Exception as exc:
        record_experiment(
            Path.cwd(),
            run_dir=run_dir,
            status=status_for_run_dir(run_dir, has_summary=False),
            config_sha1=cfg_sha1,
            config_hash_type="normalized_sha1",
            config_path=args.config,
            config_raw=raw_cfg,
            error=str(exc),
            extra_hparams={"config.overrides": override_exprs} if override_exprs else None,
        )
        logger.exception(
            "Benchmark run failed.",
            extra={"run_dir": run_dir.as_posix(), "config_sha1": cfg_sha1},
        )
        raise

    per_seed_csv = res.out_root / "metrics" / "per_seed.csv"
    summary_csv = res.out_root / "metrics" / "summary_vs_paper.csv"
    report_md = res.out_root / "report.md"
    record_experiment(
        Path.cwd(),
        run_dir=res.out_root,
        status=status_for_run_dir(res.out_root, has_summary=True),
        config_sha1=cfg_sha1,
        config_hash_type="normalized_sha1",
        config_path=args.config,
        config_raw=raw_cfg,
        summary_df=res.summary_df,
        extra_hparams={"config.overrides": override_exprs} if override_exprs else None,
    )
    logger.info(
        "[3/3] Run completed.",
        extra={
            "run_dir": res.out_root.as_posix(),
            "avg_mae": float(res.avg_mae),
            "avg_r2": float(res.avg_r2),
            "win_both": int(res.wins),
            "success": bool(res.success),
        },
    )
    logger.info(
        "Output files written.",
        extra={
            "per_seed_metrics": per_seed_csv.as_posix(),
            "summary_metrics": summary_csv.as_posix(),
            "report_path": report_md.as_posix(),
            "json_log_path": default_run_log_path(res.out_root).as_posix(),
        },
    )


if __name__ == "__main__":
    main()
