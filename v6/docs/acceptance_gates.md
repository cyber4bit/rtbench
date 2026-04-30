# v6 Acceptance Gates

`v6/scripts/finalize_v6.py` applies the same numeric acceptance thresholds as v4, but on fresh v6 seed bands.

## Validate gate

Phase: `V6_validate`  
Seeds: 30..39

| Code | Check | Threshold |
| --- | --- | --- |
| A1 | S4 avg MAE | `< 25.5482` |
| A2 | S4 avg R2 | `> 0.9144` |
| A3 | S4 beat_both | `>= 10` |
| A4 | S5 avg MAE | `< 48.0916` |
| A5 | S5 avg R2 | `> 0.8305` |
| A6 | S5 beat_both | `>= 10` |
| A7 | CP static guard and audit | PASS |
| A8 | Seed set per dataset | exactly 30..39 |
| A9 | Config diff vs frozen v3 baseline | within whitelist |

## Final gate

Phase: `V6_final`  
Seeds: 40..49

All validate checks, plus:

| Code | Check | Threshold |
| --- | --- | --- |
| B1 | Registry row | `experiments/registry.csv` row exists and config hash matches |
| B2 | Publish protocol | no rule or config changes are derived from final outputs |

Any final gate failure makes the v6 final output diagnostic only.

## Reseal gate

If `V6_final` fails, no configs, rules, or code may be selected from `outputs_v6_final_*`.
The only permitted continuation is a resealed evaluation:

- `V6_reseal_validate`: seeds 50..59; same checks as validate; writes `v6/reports/_stamps/reseal_gate.ok`.
- `V6_reseal_final`: seeds 60..69; same checks as final; writes `v6/reports/_stamps/reseal_report.ok` only on pass.

The resealed final remains publish-only: any failure is diagnostic and cannot be used for another tuning pass.
