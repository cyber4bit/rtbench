# v4 Acceptance Gates

Gates are evaluated by `v4/scripts/finalize_v4.py --gate-check`. Each gate is binary; a single FAIL aborts promotion.

## Gate Set A — L5 → L6 promotion (seeds 26–29)

| Code | Check | Threshold |
| --- | --- | --- |
| A1 | S4 avg MAE | `< 25.5482` (Uni-RT baseline) |
| A2 | S4 avg R² | `> 0.9144` |
| A3 | S4 beat_both | `≥ 10` |
| A4 | S5 avg MAE | `< 48.0916` |
| A5 | S5 avg R² | `> 0.8305` |
| A6 | S5 beat_both | `≥ 10` |
| A7 | `verify_cp_injection.py --target rtbench` | PASS |
| A8 | `cp_audit.json.film_grad_ok` | `true` (or `n/a` if no FiLM enabled) |
| A9 | Phase config diff vs v3 baseline | within whitelist (only `seeds`, `transfer_weights.per_dataset_multiplier`, `outputs.root`) |

## Gate Set B — L6 → publish (seeds 0–9)

All of A1–A9, plus:

| Code | Check | Threshold |
| --- | --- | --- |
| B1 | Seeds present per dataset | `== 10` for all S4 and S5 datasets |
| B2 | `experiments/registry.csv` row | exists and matches `config.sha1` |
| B3 | Reproducibility hash (`config.resolved.yaml`) | unchanged after rerun |

## Rollback Triggers

If any of the following occurs after L6 produces results, **revert to v3 immediately** by re-publishing v3 outputs as the active result:

- Any A or B gate FAIL.
- Discovery of a v5-class anti-pattern in `rtbench/` (`verify_cp_injection.py` newly FAILing).
- Reproducibility hash mismatch.
- Any indication that seeds 0–9 metrics influenced rule derivation or model selection.

Rollback is a strictly mechanical procedure: rename L6 outputs to `*__retracted/`, write a one-paragraph note in `reports/v4_rollback.md` citing the failed gate code, and stop.

## Diagnostic-only outputs

If the run completed but `cp_audit.json.film_grad_ok == false` or `verify_cp_injection.py` FAIL, the L5/L6 outputs are renamed `*__bugged/` automatically by `finalize_v4.py`. They may be inspected for diagnosis but **must not be cited as v4 results.**
