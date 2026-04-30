# RTBench v6

v6 is a fresh, locked iteration after the v4 final set was observed. The v4 final seeds are treated as development history only and are not used for v6 model selection.

## Protocol

- Validate on seeds 30..39 with `V6_validate`.
- Run final only after `v6/reports/_stamps/gate.ok` exists.
- Publish on seeds 40..49 with `V6_final`.
- Do not tune configs, rules, or code from `outputs_v6_final_*`.
- If `V6_final` fails, its output is diagnostic only. Any further work must be resealed on fresh bands: validate seeds 50..59 with `V6_reseal_validate`, then final seeds 60..69 with `V6_reseal_final`.

This follows the nested/held-out evaluation guidance in Cawley & Talbot (2010), Varma & Simon (2006), and Krstajic et al. (2014): model selection and final assessment must be separated to avoid selection bias.

## Commands

```powershell
python v6\scripts\cp_audit.py --repo-root . --target rtbench
python v6\scripts\run_phase.py --phase V6_validate --repo-root .
python v6\scripts\finalize_v6.py --gate-check --repo-root .
python v6\scripts\run_phase.py --phase V6_final --repo-root .
python v6\scripts\finalize_v6.py --emit-report --repo-root .
python v6\scripts\run_phase.py --phase V6_reseal_validate --repo-root .
python v6\scripts\finalize_v6.py --gate-check --phase V6_reseal_validate --repo-root .
python v6\scripts\run_phase.py --phase V6_reseal_final --repo-root .
python v6\scripts\finalize_v6.py --emit-report --phase V6_reseal_final --repo-root .
```
