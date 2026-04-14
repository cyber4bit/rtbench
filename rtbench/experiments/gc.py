from __future__ import annotations

from pathlib import Path
from typing import Any

from .archive import _normalize_relpath
from .registry import load_registry, migrate_registry


def garbage_collect_experiments(
    project_root: Path,
    *,
    status: str = "tmp",
    dry_run: bool = True,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    df, registry_abs = load_registry(project_root, registry_path=registry_path)
    view = df[df["cleanable"].str.lower() == "true"].copy()
    if status:
        view = view[view["status"].str.lower() == status.lower()].copy()

    output_roots = sorted({_normalize_relpath(path) for path in view["output_root"].tolist() if str(path).strip()})
    deleted_paths: list[str] = []
    if not dry_run:
        root_resolved = project_root.resolve()
        for rel in output_roots:
            target = (project_root / rel).resolve()
            if target.parent != root_resolved or not target.name.startswith("outputs_tmp"):
                raise ValueError(f"Refusing to delete unexpected path: {rel}")
            if target.exists():
                import shutil

                shutil.rmtree(target)
                deleted_paths.append(rel)
        migrate_registry(project_root, registry_path=registry_path)

    return {
        "status": status,
        "dry_run": bool(dry_run),
        "registry_path": (
            _normalize_relpath(registry_abs.relative_to(project_root))
            if registry_abs.is_relative_to(project_root)
            else _normalize_relpath(registry_abs)
        ),
        "candidate_roots": output_roots,
        "candidate_count": int(len(output_roots)),
        "matching_run_count": int(len(view)),
        "deleted_roots": deleted_paths,
        "deleted_count": int(len(deleted_paths)),
    }


__all__ = ["garbage_collect_experiments"]
