from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ..bench import config_from_raw, config_sha1_from_raw, normalize_target_transform, prepare, run_trial
from ..config import parse_seed_range, resolve_config
from ..experiments import record_experiment, status_for_run_dir, write_effective_config_snapshot
from ..logging_utils import attach_json_log, configure_logging, default_run_log_path


logger = logging.getLogger("rtbench.experimental.sweep")


def _fmt_clip(x: float) -> str:
    # 2.5 -> "2p5"
    text = f"{float(x):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _has_tree_blocks(models: dict[str, Any]) -> bool:
    return all(k in models for k in ("XGB_A", "XGB_B", "LGBM_A", "LGBM_B"))


def _resolve_sweep_output_dir(base_raw: dict[str, Any], cli_output_dir: str) -> Path:
    configured_root = str(base_raw.get("outputs", {}).get("root", "")).strip()
    chosen = cli_output_dir.strip() if cli_output_dir.strip() else configured_root
    return Path(chosen or "outputs_sweep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweep runner for RepoRT benchmark (in-process)")
    parser.add_argument("--config", required=True, help="Base YAML config path")
    parser.add_argument("--seeds", default="", help="Override seeds (default: config.seeds.default)")
    parser.add_argument("--no-download", action="store_true", help="Use local data only, no network download")
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        default="",
        help="Directory to write trial outputs under. Defaults to cfg.outputs.root, else outputs_sweep.",
    )
    parser.add_argument("--write-predictions", action="store_true", help="Write per-seed prediction CSVs")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Also sweep hybrid fusion knobs (ONLY_HYPER_TL/FUSION_TOP_K) if tree blocks exist in base config",
    )
    args = parser.parse_args()

    resolved = resolve_config(args.config)
    base_cfg = resolved.config
    seed_expr = args.seeds if args.seeds else str(base_cfg.seeds["default"])
    seeds = parse_seed_range(seed_expr)
    external_ids = [str(x).zfill(4) for x in base_cfg.datasets["external"]]

    out_dir = _resolve_sweep_output_dir(base_raw=resolved.raw, cli_output_dir=args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(level=args.log_level, json_log_path=default_run_log_path(out_dir, filename="sweep.jsonl"))

    tf0 = normalize_target_transform(base_cfg.transfer_weights)
    logger.info(
        "Starting sweep run.",
        extra={
            "config_path": args.config,
            "out_dir": out_dir.as_posix(),
            "seed_expr": seed_expr,
            "external_dataset_count": len(external_ids),
            "include_hybrid": bool(args.include_hybrid),
            "write_predictions": bool(args.write_predictions),
        },
    )
    logger.info("Base target_transform=%s", tf0)
    logger.info("Preparing benchmark for %d external datasets, seeds=%s.", len(external_ids), seed_expr)
    prep = prepare(base_cfg, external_ids=external_ids, no_download=bool(args.no_download))

    base_raw = copy.deepcopy(resolved.raw)

    ridge_lambdas = [0.0, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    target_transforms = ["none", "logk", "gradient_norm"]
    balance_opts = [False, True]
    clip_opts = [1.5, 2.5, 3.0]
    calibrate_opts = [False, True]

    include_hybrid = bool(args.include_hybrid) and _has_tree_blocks(base_raw.get("models", {}))
    if bool(args.include_hybrid) and not include_hybrid:
        logger.warning("Requested --include-hybrid but the base config lacks tree blocks; skipping hybrid trials.")

    trial_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []

    for tfm in target_transforms:
        for bal in balance_opts:
            for clip in clip_opts:
                for cal in calibrate_opts:
                    only_hyper_list = [bool(base_raw["models"].get("ONLY_HYPER_TL", False))]
                    fusion_top_k_list = [int(base_raw["models"].get("FUSION_TOP_K", 1))]
                    if include_hybrid:
                        only_hyper_list = [True, False]
                        fusion_top_k_list = [2, 3]
                    for only_hyper in only_hyper_list:
                        for top_k in fusion_top_k_list:
                            if only_hyper and top_k != 1:
                                # Hyper-only: fusion over candidates is irrelevant; keep deterministic.
                                continue
                            trial_name = (
                                f"tf_{tfm}"
                                f"_bal{1 if bal else 0}"
                                f"_clip{_fmt_clip(clip)}"
                                f"_cal{1 if cal else 0}"
                                + (f"_hyb{0 if only_hyper else 1}_k{top_k}" if include_hybrid else "")
                            )
                            trials.append(
                                {
                                    "name": trial_name,
                                    "tfm": tfm,
                                    "bal": bal,
                                    "clip": float(clip),
                                    "cal": cal,
                                    "only_hyper": only_hyper,
                                    "top_k": int(top_k),
                                }
                            )

    logger.info("Planned trials: %d", len(trials), extra={"trial_count": len(trials)})

    for i, spec in enumerate(trials, start=1):
        name = spec["name"]
        logger.info("[%d/%d] %s", i, len(trials), name, extra={"trial": name, "trial_index": i, "trial_count": len(trials)})

        raw = copy.deepcopy(base_raw)
        raw["outputs"]["root"] = str(out_dir / name)
        raw["outputs"]["resume"] = False

        # Sweep knobs.
        raw["transfer_weights"]["target_transform"] = str(spec["tfm"])
        raw["transfer_weights"]["target_normalize"] = False
        raw["models"]["CLIP_MULT"] = float(spec["clip"])
        raw["models"]["CALIBRATE"] = bool(spec["cal"])
        raw["models"]["ENABLE_HYPER_TL"] = True
        raw["models"]["ONLY_HYPER_TL"] = bool(spec["only_hyper"])
        raw["models"]["FUSION_TOP_K"] = int(spec["top_k"])

        raw["models"].setdefault("HYPER_TL", {})
        raw["models"]["HYPER_TL"]["balance_pretrain_by_dataset"] = bool(spec["bal"])
        raw["models"]["HYPER_TL"]["ridge_lambdas"] = [float(x) for x in ridge_lambdas]

        trial_cfg = config_from_raw(raw)
        sha1 = config_sha1_from_raw(raw)
        run_dir = Path(raw["outputs"]["root"])
        write_effective_config_snapshot(Path.cwd(), run_dir=run_dir, config_raw=raw, config_sha1=sha1)

        with attach_json_log(default_run_log_path(run_dir)):
            logger.info(
                "Running sweep trial.",
                extra={
                    "trial": name,
                    "run_dir": run_dir.as_posix(),
                    "trial_index": i,
                    "trial_count": len(trials),
                    "target_transform": spec["tfm"],
                    "balance_pretrain_by_dataset": bool(spec["bal"]),
                    "clip_mult": float(spec["clip"]),
                    "calibrate": bool(spec["cal"]),
                    "only_hyper_tl": bool(spec["only_hyper"]),
                    "fusion_top_k": int(spec["top_k"]),
                },
            )
            try:
                res = run_trial(
                    prep,
                    trial_cfg,
                    seeds=seeds,
                    external_ids=external_ids,
                    config_sha1=sha1,
                    resume_enabled=False,
                    write_predictions=bool(args.write_predictions),
                    early_stop=True,
                )
            except Exception as exc:
                record_experiment(
                    Path.cwd(),
                    run_dir=run_dir,
                    status=status_for_run_dir(run_dir, has_summary=False),
                    config_sha1=sha1,
                    config_hash_type="normalized_sha1",
                    config_path=args.config,
                    config_raw=raw,
                    error=str(exc),
                    extra_hparams={"sweep.trial": name},
                )
                logger.exception("Sweep trial failed.", extra={"trial": name, "run_dir": run_dir.as_posix()})
                raise

            record_experiment(
                Path.cwd(),
                run_dir=res.out_root,
                status=status_for_run_dir(res.out_root, has_summary=True),
                config_sha1=sha1,
                config_hash_type="normalized_sha1",
                config_path=args.config,
                config_raw=raw,
                summary_df=res.summary_df,
                extra_hparams={"sweep.trial": name},
            )
            logger.info(
                "Sweep trial completed.",
                extra={
                    "trial": name,
                    "run_dir": res.out_root.as_posix(),
                    "avg_mae": float(res.avg_mae),
                    "avg_r2": float(res.avg_r2),
                    "win_both": int(res.wins),
                    "success": bool(res.success),
                    "early_stop_reason": str(res.early_stop_reason),
                },
            )

        trial_rows.append(
            {
                "trial": name,
                "target_transform": spec["tfm"],
                "balance_pretrain_by_dataset": bool(spec["bal"]),
                "clip_mult": float(spec["clip"]),
                "calibrate": bool(spec["cal"]),
                "only_hyper_tl": bool(spec["only_hyper"]),
                "fusion_top_k": int(spec["top_k"]),
                "avg_mae": float(res.avg_mae),
                "avg_r2": float(res.avg_r2),
                "wins": int(res.wins),
                "success": bool(res.success),
                "early_stop_reason": str(res.early_stop_reason),
                "out_root": str(res.out_root),
            }
        )

    df = pd.DataFrame(trial_rows)
    paper_mae = float(base_cfg.metrics["paper_avg_mae"])
    paper_r2 = float(base_cfg.metrics["paper_avg_r2"])
    df["delta_mae"] = df["avg_mae"] - paper_mae
    df["delta_r2"] = df["avg_r2"] - paper_r2

    df = df.sort_values(["success", "delta_mae", "delta_r2"], ascending=[False, True, False]).reset_index(drop=True)
    out_csv = out_dir / "sweep_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    logger.info("Wrote sweep summary: %s", out_csv.as_posix(), extra={"summary_csv": out_csv.as_posix()})
    if not df.empty:
        top = df.iloc[0]
        logger.info(
            "Best trial: %s success=%s avg_mae=%.4f avg_r2=%.4f wins=%d",
            top["trial"],
            bool(top["success"]),
            float(top["avg_mae"]),
            float(top["avg_r2"]),
            int(top["wins"]),
            extra={
                "best_trial": str(top["trial"]),
                "success": bool(top["success"]),
                "avg_mae": float(top["avg_mae"]),
                "avg_r2": float(top["avg_r2"]),
                "win_both": int(top["wins"]),
            },
        )


if __name__ == "__main__":
    main()
