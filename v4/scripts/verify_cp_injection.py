from __future__ import annotations

import argparse
from pathlib import Path


BAD_PATTERNS = {
    "zero_cp_conditioning": [
        "h.new_zeros((task_ids.size(0), 64))",
        "new_zeros((task_ids.size(0), 64))",
        "zeros((task_ids.size(0), 64))",
    ],
    "copy_method_bug": [
        "true_raw.copy\n",
        "true_raw.copy\r\n",
        "true_raw.copy;",
        "true_raw.copy ",
    ],
    "merged_split_function": [
        "def merged_stratified_split",
    ],
    "main_task_rotation": [
        "for main_task in range(num_tasks)",
    ],
}


def scan(target: Path) -> list[tuple[str, Path, str]]:
    findings: list[tuple[str, Path, str]] = []
    files = list(target.rglob("*.py")) if target.is_dir() else [target]
    self_path = Path(__file__).resolve()
    for path in files:
        if path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        for name, patterns in BAD_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    findings.append((name, path, pattern.strip()))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Static guard for v4 CP-conditioning anti-patterns.")
    parser.add_argument(
        "--target",
        default=str(Path(".claude") / "worktrees" / "romantic-wilson-87e23a" / "v5"),
        help="File or directory to scan.",
    )
    args = parser.parse_args()
    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"Target does not exist: {target}")

    findings = scan(target)
    if not findings:
        print(f"PASS: no known v5 anti-patterns found in {target}")
        return

    print(f"FAIL: found {len(findings)} known anti-pattern(s) in {target}")
    for name, path, pattern in findings:
        print(f"- {name}: {path} contains {pattern!r}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
