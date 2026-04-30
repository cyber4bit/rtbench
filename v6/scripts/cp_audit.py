from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from verify_cp_injection import scan


def _film_present(target: Path) -> bool:
    files = list(target.rglob("*.py")) if target.is_dir() else [target]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        lower = text.lower()
        if "film" in lower or ("gamma" in lower and "beta" in lower and "cp" in lower):
            return True
    return False


def build_audit(repo_root: Path, target: Path) -> dict[str, Any]:
    target_path = target if target.is_absolute() else repo_root / target
    if not target_path.exists():
        raise FileNotFoundError(f"Target does not exist: {target_path}")
    findings = scan(target_path)
    film_present = _film_present(target_path)
    hard_bug = bool(findings)
    film_grad_ok: bool | str = "n/a"
    cp_sensitivity_p: float | None = None
    if film_present:
        hard_bug = True
        film_grad_ok = False
    return {
        "schema_version": 1,
        "target": str(target).replace("\\", "/"),
        "film_present": bool(film_present),
        "film_grad_ok": film_grad_ok,
        "cp_sensitivity_p": cp_sensitivity_p,
        "hard_bug": bool(hard_bug),
        "findings": [
            {"kind": kind, "path": str(path).replace("\\", "/"), "pattern": pattern}
            for kind, path, pattern in findings
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v6 CP injection audit.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--target", default="rtbench")
    parser.add_argument("--out", default="v6/reports/cp_audit.json")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    audit = build_audit(repo_root, Path(args.target))
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    if not audit["hard_bug"]:
        stamp = repo_root / "v6/reports/_stamps/cp_audit.ok"
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("ok\n", encoding="utf-8")
        print(f"wrote {stamp}")
    else:
        raise SystemExit("CP audit found a hard bug")


if __name__ == "__main__":
    main()
