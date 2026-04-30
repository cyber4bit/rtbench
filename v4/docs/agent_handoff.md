# Agent Handoff Prompt

Use this file as the copy-ready handoff for the next implementation agent.

## Task

Implement the RTBench v4 pipeline exactly as specified in `v4/`. Start from
`v4/AGENT_BRIEF.md`, `v4/ARCHITECTURE.md`, and
`v4/docs/script_contracts.md`. Do not change the strategy.

## Scope You Own

Implement or update:

- `v4/scripts/profile_failures.py`
- `v4/scripts/cp_audit.py`
- `v4/scripts/derive_loss_rules.py`
- `v4/scripts/run_phase.py`
- `v4/scripts/finalize_v4.py`
- focused tests under `tests/`
- optional minimal shim in `rtbench/bench/weighting.py` and
  `rtbench/bench/runner.py` only if `per_dataset_multiplier` is not already
  supported

Do not rewrite unrelated model code. Do not change v3 baseline outputs. Do not
edit v5.

## Inputs

Frozen v3 inputs live at:

```text
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/
```

Debate source:

```text
D:\fenxi-codex\辩论\辩论.md
```

## Required Execution Flow

```powershell
python v4\scripts\failure_matrix.py
python v4\scripts\verify_cp_injection.py --target rtbench
python v4\scripts\profile_failures.py --v3-root .claude\worktrees\romantic-wilson-87e23a --tuning-seeds 23,24,25 --out v4\reports\preregistered_profile.csv
python v4\scripts\cp_audit.py --target rtbench --out v4\reports\cp_audit.json
python v4\scripts\run_phase.py --phase L1_focal_3x
python v4\scripts\derive_loss_rules.py --profile v4\reports\preregistered_profile.csv --out v4\reports\loss_weight_rules.json
python v4\scripts\run_phase.py --phase L5_validate
python v4\scripts\finalize_v4.py --gate-check
python v4\scripts\run_phase.py --phase L6_final
python v4\scripts\finalize_v4.py --emit-report
```

L1, L2, and L3 can run in parallel, but the implementation must make the
dependency stamps explicit.

## Forbidden Moves

- No seed `0..9` access before `gate.ok`.
- No L1 focal result used in rule derivation.
- No CP audit value used as a weight-selection input.
- No merged split.
- No per-dataset lr/dropout/hidden-dim/model-family override.
- No architecture change unless a hard bug fix is required.

## Acceptance Tests Before Final Answer

Run at least:

```powershell
python -m pytest tests
python v4\scripts\failure_matrix.py
python v4\scripts\verify_cp_injection.py --target rtbench
python v4\scripts\derive_loss_rules.py --profile v4\reports\preregistered_profile.csv --out v4\reports\loss_weight_rules.json
```

If full L1/L5/L6 training is too long, run smoke mode only if you add an
explicit `--dry-run` or `--generate-config-only` option and document that the
full training was not executed.

## Final Deliverables

The implementation is acceptable only when these files can be produced:

```text
v4/reports/preregistered_profile.csv
v4/reports/cp_audit.json
v4/reports/loss_weight_rules.json
outputs_v4_L1_focal_3x_S4/metrics/summary_vs_paper.csv
outputs_v4_L1_focal_3x_S5/metrics/summary_vs_paper.csv
outputs_v4_L5_validate_S4/metrics/summary_vs_paper.csv
outputs_v4_L5_validate_S5/metrics/summary_vs_paper.csv
v4/reports/gate_report.md
outputs_v4_L6_final_S4/metrics/summary_vs_paper.csv
outputs_v4_L6_final_S5/metrics/summary_vs_paper.csv
v4/reports/v4_report.md
```

The final answer must say which gates passed, which scripts were implemented,
which tests ran, and whether full training was executed or only dry-run checks
were executed.
