# v7 Architecture Literature Findings

## Evidence Used

- Local vault note: `D:\paper\20_Research\Papers\Chromatography_MS_Compound_ID\Unified_Multitask_Modeling_for_Retention_Time_Prediction_Across_Chromatographic_Conditions.md`
- Uni-RT code inspected at `https://github.com/hcji/Uni-RT` commit `b981fc118b1264f937626b8ec980b426100537b1`
- Uni-RT article: `https://pubs.acs.org/doi/10.1021/acs.analchem.5c07973`

## Literature Methods Mapped to v7

| Literature module | Intended role | v7 implementation path | Probe result |
| --- | --- | --- | --- |
| FiLM condition modulation | Condition-specific feature scaling and shifting | `_HyperNet.use_film`, optional conditioned embeddings | Negative on S5 when used in exported embeddings; also negative as head-only pretraining |
| Adapter task specialization | Small task-specific residual bottleneck | `_HyperNet.use_task_adapters` with per-dataset adapters | Negative on S5 and much slower |
| Cross-Stitch sharing | Learn feature mixing between related tasks | `_HyperNet.use_cross_stitch` over task-adapted embeddings | Strongly negative in current batch/inference setting |
| KAN/nonlinear head | Fit nonlinear structure-retention relation | `HYPER_EMB_LGBM` as practical nonlinear head over Hyper-TL embeddings | Small positive on S5 |
| Sheet-level nonlinear head | Reuse all sheet train/val rows with one shared head | `SHEET_EMB_LGBM`, `SHEET_FULL_LGBM_SEC` | Only marginal S5 gain; does not fix 037x tail |
| CP-similarity source weighting | Prefer source rows under chromatographically similar conditions | `SHEET_HEAD_CP_SIM_WEIGHT` for sheet heads | Best current S5 single-seed probe: Avg MAE `73.1458`, Avg R2 `0.5748`; still far from gate |
| Hyper-prior residual head | Learn a shared residual correction around the CP-conditioned Hyper prior | `SHEET_RESID_LGBM`, `SHEET_FULL_RESID_LGBM` | Negative on 037x because validation improves while test outliers worsen |
| Chromatographic parameter vectorization | Encode condition metadata explicitly | Existing CP vector hypernetwork head plus optional CP/dataset one-hot conditioning | CP works best; dataset one-hot is negative |
| Molecular semantic context | Add formula/ClassyFire molecular context to descriptor block | Optional `data.include_molecule_context` | Fixes Acetyl-CoA high bias in 037x, but hurts other datasets and Folic Acid remains low |
| SMILES sequence encoder | Learn molecule text fragments directly instead of hashed n-grams | Optional `data.molecule_sequence_max_len` + Hyper-TL char-CNN branch | Strongly negative in current sheet-unified pretraining; validation overfits and 037x R2 becomes negative |
| Formula magnitude context | Preserve molecular size missing from formula ratios | Formula context now includes ratios, log-counts, total atom scale, hetero fraction | Does not break gate; useful as an ablation switch but not stable enough for default |
| Sheet memory attention | Soft source-row attention over full/fingerprint molecular features and CP similarity | `ENABLE_SHEET_MEMORY_ATTENTION` research switch | Negative or neutral; raw fingerprint attention slightly changes 0372 but remains far below gate |
| Unified transfer tree | Shared source+target tree candidate without lookup/pool | Lightweight transfer LGBM probe | Small best single-seed average gain (`72.1574` MAE, `0.5811` R2) but still far below S5 gate and not retained as v7 default |

## Probe Summary

Baseline v7 sheet-unified Hyper-TL, no FiLM/one-hot:

- S4: Avg MAE `19.4370`, Avg R2 `0.9295`
- S5: Avg MAE `76.3823`, Avg R2 `0.5361`

Current default with `HYPER_EMB_LGBM`:

- S4: Avg MAE `19.4866`, Avg R2 `0.9301`
- S5: Avg MAE `75.3342`, Avg R2 `0.5424`

Architecture probes:

- FiLM + conditioned embeddings: S5 Avg MAE `89.9864`, Avg R2 `0.3005`
- Dataset one-hot conditioning: S5 degraded versus CP-only conditioning
- Task Adapter + Cross-Stitch: S5 Avg MAE `96.6183`, Avg R2 `0.2204`
- FiLM head-only pretraining: S5 Avg MAE `89.2483`, Avg R2 `0.2988`
- Sheet embedding LGBM head: S5 Avg MAE `75.0790`, Avg R2 `0.5625`
- Sheet full-feature seconds LGBM head: S5 Avg MAE `74.7833`, Avg R2 `0.5459`; S4 Avg MAE `19.5112`, Avg R2 `0.9301`
- Sheet embedding + full seconds with CP-sim weighting: S5 Avg MAE `73.1458`, Avg R2 `0.5748` at `SHEET_HEAD_CP_SIM_POWER=1.0`
- Sheet Hyper-prior residual heads: S5 Avg MAE `73.8398`, Avg R2 `0.5610`; rejected because 0372/0374/0377 regress
- Sheet prior calibration: S5 Avg MAE `75.8312`, Avg R2 `0.5355`
- Molecule context features: S5 Avg MAE `77.5495`, Avg R2 `0.3248`
- Molecule context + CP-sim sheet heads: S5 Avg MAE `75.3952`, Avg R2 `0.1999`; context-only sheet embedding improves some 037x MAE but collapses R2 and non-tail datasets
- Molecule context with formula counts: S5 Avg MAE `82.5967`, Avg R2 `0.4693`; rejected
- Molecule context split away from Hyper-TL and exposed only to full sheet heads: S5 Avg MAE `73.2470`, Avg R2 `0.5743`; effectively neutral versus current best
- SMILES char-CNN sequence branch: S5 Avg MAE `108.8188`, Avg R2 `0.0944`; rejected
- Sheet memory attention over fingerprints, no calibration: S5 Avg MAE `73.1199`, Avg R2 `0.5777`; neutral average, tail still fails
- Lightweight unified transfer LGBM candidates: S5 Avg MAE `72.1574`, Avg R2 `0.5811`; small improvement but not a gate breakthrough
- Pairwise ranking loss in Hyper-TL pretraining: degraded S5 in tested weights `0.02` and `0.05`
- Target transforms: `none`, `log1p`, and `logk` all degraded S5 versus `gradient_norm`

## Current S5 Failure Mode

The remaining S5 gap is concentrated in the 0372-0378 family. Median errors are often modest, but two recurring test compounds dominate MAE:

- `_00003` Acetyl-CoA is overpredicted by roughly 190-287 seconds in the CP-only default. Formula/ClassyFire context reduces this error, but causes broad degradation elsewhere.
- `_00022` Folic Acid is underpredicted by roughly 203-319 seconds across the same family and is not repaired by the tested semantic context or sheet-level heads.

This is a molecular ordering failure rather than a global calibration or target-scale failure.

The latest probes sharpen this diagnosis. ClassyFire/formula context consistently repairs the
Acetyl-CoA overprediction, but Folic Acid remains underpredicted by roughly 250-350 seconds.
Class-level priors are not sufficient either: broad chemical classes over-correct unrelated
compounds, while the Folic Acid level is still pulled down by less-similar source conditions.
The remaining gap therefore requires a stronger unified molecular representation rather than
more validation-set fusion or local lookup-style retrieval.

The follow-up v7 architecture attempts make the failure mode sharper. Adding raw SMILES
sequence learning, formula magnitude features, or memory-style source attention does not
solve the Folic Acid underprediction in a validation-stable way. The strongest safe probe
after these changes is still a unified CP-conditioned Hyper-TL encoder with CP-similarity
weighted sheet heads; transfer trees can slightly improve the single-seed average but do
not close the R2 gap.

## Decision

Keep the literature-inspired modules in code as explicit research switches, but keep them off in the v7 default manifest unless a later multi-seed validation proves otherwise. In this codebase, the strongest architecture lesson from Uni-RT is not to add task identity naively. The reliable transfer path remains a sheet-level unified CP-conditioned Hyper-TL encoder plus carefully selected nonlinear heads, but the current non-lookup unified path has not broken the S5 gate.
