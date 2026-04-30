from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CANONICAL_PROFILE = Path("v4/reports/preregistered_profile.csv")
BOOST_TAGS = {"hard_loss", "catastrophic", "near_miss_r2", "near_miss_mae"}
KNOWN_TAGS = {"win", "near_miss", "near_miss_r2", "near_miss_mae", "hard_loss", "catastrophic"}


def derive(profile: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with open(profile, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"Profile is empty: {profile}")
    rules: dict[str, dict[str, float]] = {"S4": {}, "S5": {}}
    for row in rows:
        sheet = str(row.get("sheet", "")).strip()
        dataset = str(row.get("dataset", "")).zfill(4)
        tag = str(row.get("failure_tag", "")).strip()
        if sheet not in rules:
            raise ValueError(f"Unknown sheet in profile: {sheet!r}")
        if tag not in KNOWN_TAGS:
            raise ValueError(f"Unknown failure_tag for {sheet}/{dataset}: {tag!r}")
        if tag in BOOST_TAGS:
            rules[sheet][dataset] = 3.0
    return {
        "schema_version": 1,
        "source_profile": CANONICAL_PROFILE.as_posix(),
        "rules": {sheet: dict(sorted(values.items())) for sheet, values in rules.items()},
        "default_weight": 1.0,
        "derivation": "failure_tag in hard_loss,catastrophic,near_miss_r2,near_miss_mae -> 3.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive v4 loss weight rules from the locked L2 profile.")
    parser.add_argument("--profile", default=CANONICAL_PROFILE.as_posix())
    parser.add_argument("--out", default="v4/reports/loss_weight_rules.json")
    parser.add_argument("--allow-noncanonical-profile", action="store_true")
    args = parser.parse_args()
    profile_arg = Path(args.profile)
    if not args.allow_noncanonical_profile and profile_arg.as_posix() != CANONICAL_PROFILE.as_posix():
        raise SystemExit(f"Refusing noncanonical profile path: {profile_arg}")
    payload = derive(profile_arg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# v4 Loss Weight Rules",
        "",
        f"- Source profile: `{payload['source_profile']}`",
        f"- Default weight: `{payload['default_weight']}`",
        f"- Derivation: `{payload['derivation']}`",
        "",
        "| Sheet | Dataset | Weight |",
        "| --- | --- | ---: |",
    ]
    for sheet in ("S4", "S5"):
        for dataset, weight in payload["rules"][sheet].items():
            md_lines.append(f"| {sheet} | {dataset} | {weight:.1f} |")
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    stamp = Path("v4/reports/_stamps/rules.ok")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    print(f"wrote {stamp}")


if __name__ == "__main__":
    main()
