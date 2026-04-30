# RTBench v4 Agent Execution Brief

This is the single entry point for an agent taking over v4. The strategy is
frozen by the OpenCode-vs-Codex debate in `D:\fenxi-codex\辩论\辩论.md`; do not
change it while implementing.

Read in this order:

1. `ARCHITECTURE.md`
2. `docs/debate_consensus.md`
3. `docs/preregistered_schema.md`
4. `docs/script_contracts.md`
5. `docs/agent_handoff.md`

## 1. Objective

v4 must make both S4 and S5 satisfy all three gates:

- avg MAE below Uni-RT,
- avg R2 above Uni-RT,
- `beat_both >= 10/14`.

v3 already gets S4 avg R2 to `0.9701` and both sheets to `beat_both = 9/14`.
The remaining failures are:

- S4: `0179`, `0180`, `0234`, `0261`, `0264`
- S5: `0027`, `0183`, `0184`, `0185`, `0282`

## 2. Frozen Strategy

v4 is data/loss-first:

- Keep v3 unified HyperNet as rollback baseline.
- Use fixed 3x loss weighting only as a sanity check on seeds `20..22`.
- Build the actual rule from a pre-registered profile on seeds `23..25`.
- Validate the rule on seeds `26..29`.
- Use seeds `0..9` only after L5 gate pass.
- Run CP audit as read-only. It can invalidate results, but it cannot choose
  loss weights.

No architecture change is allowed in the default v4 path.

## 3. Required Paths

Repository root:

```text
D:\fenxi-codex\分析-codex
```

Frozen v3 artifacts:

```text
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s4/
.claude/worktrees/romantic-wilson-87e23a/outputs_v3_s5/
```

Required v3 files:

```text
config.resolved.yaml
config.sha1
v3_vs_uni_rt.csv
metrics/per_seed.csv
metrics/summary_vs_paper.csv
```

## 4. Pipeline

```text
Step 0  Pre-flight
Step 1  L2 profile            seeds 23..25
Step 2  L1 fixed 3x focal     seeds 20..22
Step 3  L3 CP audit           read-only
Step 4  L4 loss rules         consumes L2 only
Step 5  L5 validation         seeds 26..29
Step 6  Gate check            creates gate.ok only on pass
Step 7  L6 final eval         seeds 0..9
Step 8  Final report
```

L1, L2, and L3 may run in parallel. L4 must not read L1 or L3 outputs.

## 5. Commands

```powershell
python v4\scripts\failure_matrix.py
python v4\scripts\verify_cp_injection.py --target rtbench

python v4\scripts\profile_failures.py `
  --v3-root .claude\worktrees\romantic-wilson-87e23a `
  --tuning-seeds 23,24,25 `
  --out v4\reports\preregistered_profile.csv

python v4\scripts\cp_audit.py `
  --target rtbench `
  --out v4\reports\cp_audit.json

python v4\scripts\run_phase.py --phase L1_focal_3x

python v4\scripts\derive_loss_rules.py `
  --profile v4\reports\preregistered_profile.csv `
  --out v4\reports\loss_weight_rules.json

python v4\scripts\run_phase.py --phase L5_validate
python v4\scripts\finalize_v4.py --gate-check
python v4\scripts\run_phase.py --phase L6_final
python v4\scripts\finalize_v4.py --emit-report
```

The missing scripts are not optional; their exact contracts are in
`docs/script_contracts.md`.

## 6. Phase Manifests

Files in `v4/configs/` are manifests for orchestration. They are not all
standalone `rtbench-run` configs.

`run_phase.py` must:

1. load the v3 S4 and S5 `config.resolved.yaml`,
2. apply only whitelisted overrides from the phase manifest,
3. write generated configs under `v4/reports/_generated_configs/`,
4. call `python -m rtbench.run --config <generated-config>` once per sheet.

Allowed config changes:

- `seeds.default`
- `transfer_weights.per_dataset_multiplier`
- `outputs.root`

Everything else must match v3.

## 7. Acceptance Gates

L5 to L6:

- S4 avg MAE < `25.5482`
- S4 avg R2 > `0.9144`
- S4 `beat_both >= 10`
- S5 avg MAE < `48.0916`
- S5 avg R2 > `0.8305`
- S5 `beat_both >= 10`
- static CP guard passes
- CP audit is clean or no FiLM path exists
- generated configs differ from v3 only by whitelisted keys

L6 publish:

- same gates on seeds `0..9`,
- exactly 10 seeds per dataset,
- registry/config hashes recorded,
- `v4/reports/v4_report.md` emitted.

## 8. Hard Stop Rules

Stop immediately if:

- v3 files are missing,
- schema columns do not match `docs/preregistered_schema.md`,
- any seed `0..9` appears before gate pass,
- L4 reads anything other than `preregistered_profile.csv`,
- CP hard bug is detected,
- config diff leaves the whitelist,
- L5 gate fails.

Bugged L5/L6 outputs are diagnostic only. They cannot freeze or publish v4.

## 9. Minimal Code Change Allowed

If the active runner lacks `transfer_weights.per_dataset_multiplier`, add only
the target-train loss-weight shim described in `ARCHITECTURE.md` and
`docs/script_contracts.md`.

Do not change architecture, splits, optimizer, learning rate, dropout, hidden
dimensions, or candidate pools.
