from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..logging_utils import configure_logging, default_run_log_path
from .gc import garbage_collect_experiments
from .query import compare_experiments, query_experiments
from .registry import (
    DEFAULT_CLEANUP_MANIFEST,
    _registry_cli_default,
    cleanup_tmp_outputs,
    migrate_registry,
)


logger = logging.getLogger("rtbench.experiments")


def _cmd_migrate(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    configure_logging(
        level=args.log_level,
        json_log_path=default_run_log_path(project_root / "experiments", filename="experiments.jsonl"),
    )
    summary = migrate_registry(
        project_root,
        registry_path=Path(args.registry),
        cleanup_manifest_path=Path(args.cleanup_manifest),
    )
    logger.info(
        "Registry migration completed.",
        extra={
            "project_root": project_root.as_posix(),
            "record_count": int(summary.record_count),
            "output_root_count": int(summary.output_root_count),
            "cleanable_output_root_count": int(summary.cleanable_root_count),
            "registry_path": summary.registry_path.as_posix(),
            "cleanup_manifest_path": summary.cleanup_manifest_path.as_posix(),
        },
    )
    return 0


def _cmd_cleanup_tmp(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    configure_logging(
        level=args.log_level,
        json_log_path=default_run_log_path(project_root / "experiments", filename="experiments.jsonl"),
    )
    candidates = cleanup_tmp_outputs(project_root, delete=bool(args.delete))
    action = "deleted" if args.delete else "candidate"
    logger.info(
        "Tmp outputs %s complete.",
        action,
        extra={
            "project_root": project_root.as_posix(),
            "action": action,
            "candidate_count": len(candidates),
        },
    )
    for path in candidates:
        logger.info("Tmp output root: %s", path.name, extra={"tmp_output_root": path.name, "action": action})
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    table = query_experiments(
        project_root,
        metric=args.metric,
        sort=args.sort,
        top=args.top,
        status=args.status,
        registry_path=Path(args.registry),
    )
    if table.empty:
        print("No matching experiments.")
    else:
        print(table.to_string(index=False))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    payload = compare_experiments(
        project_root,
        args.run_a,
        args.run_b,
        registry_path=Path(args.registry),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_gc(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    configure_logging(
        level=args.log_level,
        json_log_path=default_run_log_path(project_root / "experiments", filename="experiments.jsonl"),
    )
    payload = garbage_collect_experiments(
        project_root,
        status=args.status,
        dry_run=not bool(args.delete),
        registry_path=Path(args.registry),
    )
    logger.info(
        "Experiment GC completed.",
        extra={
            "project_root": project_root.as_posix(),
            "status": str(payload["status"]),
            "dry_run": bool(payload["dry_run"]),
            "candidate_count": int(payload["candidate_count"]),
            "deleted_count": int(payload["deleted_count"]),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment registry and migration utilities")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_parser = subparsers.add_parser("migrate", help="Scan historical outputs and rebuild experiments/registry.csv")
    migrate_parser.add_argument("--project-root", default=".")
    migrate_parser.add_argument(
        "--registry",
        default=_registry_cli_default(),
        help="Registry CSV path. Defaults to RTBENCH_REGISTRY_PATH when set, else experiments/registry.csv.",
    )
    migrate_parser.add_argument("--cleanup-manifest", default=str(DEFAULT_CLEANUP_MANIFEST))
    migrate_parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    migrate_parser.set_defaults(func=_cmd_migrate)

    cleanup_parser = subparsers.add_parser("cleanup-tmp", help="List or delete outputs_tmp* directories")
    cleanup_parser.add_argument("--project-root", default=".")
    cleanup_parser.add_argument("--delete", action="store_true", help="Actually delete the tmp directories")
    cleanup_parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    cleanup_parser.set_defaults(func=_cmd_cleanup_tmp)

    query_parser = subparsers.add_parser("query", help="Query top experiments from experiments/registry.csv")
    query_parser.add_argument("--project-root", default=".")
    query_parser.add_argument(
        "--registry",
        default=_registry_cli_default(),
        help="Registry CSV path. Defaults to RTBENCH_REGISTRY_PATH when set, else experiments/registry.csv.",
    )
    query_parser.add_argument("--metric", default="avg_mae", help="Registry column used for ranking, e.g. avg_mae or avg_r2")
    query_parser.add_argument("--sort", default="asc", choices=["asc", "desc"], help="Sort order for the ranking metric")
    query_parser.add_argument("--top", type=int, default=10, help="Maximum number of experiments to print")
    query_parser.add_argument("--status", default="", help="Optional status filter, e.g. success or tmp")
    query_parser.set_defaults(func=_cmd_query)

    compare_parser = subparsers.add_parser("compare", help="Compare two experiments dataset-by-dataset")
    compare_parser.add_argument("run_a", help="Reference experiment id or run_dir")
    compare_parser.add_argument("run_b", help="Experiment id or run_dir to compare against run_a")
    compare_parser.add_argument("--project-root", default=".")
    compare_parser.add_argument(
        "--registry",
        default=_registry_cli_default(),
        help="Registry CSV path. Defaults to RTBENCH_REGISTRY_PATH when set, else experiments/registry.csv.",
    )
    compare_parser.set_defaults(func=_cmd_compare)

    gc_parser = subparsers.add_parser("gc", help="Safely clean registry-backed temporary experiment roots")
    gc_parser.add_argument("--project-root", default=".")
    gc_parser.add_argument(
        "--registry",
        default=_registry_cli_default(),
        help="Registry CSV path. Defaults to RTBENCH_REGISTRY_PATH when set, else experiments/registry.csv.",
    )
    gc_parser.add_argument("--status", default="tmp", help="Registry status to target, default: tmp")
    gc_parser.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting anything")
    gc_parser.add_argument("--delete", action="store_true", help="Actually delete the selected output roots")
    gc_parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, or WARNING")
    gc_parser.set_defaults(func=_cmd_gc)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


__all__ = ["main"]
