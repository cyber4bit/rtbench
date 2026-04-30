# RTBench v7

v7 starts from the v6 code line but removes the v6 escape hatches that made S5 look better without preserving a single unified model. The main change is a sheet-level unified Hyper-TL path:

- one Hyper-TL bundle per seed and sheet;
- training rows are the base source rows plus every external dataset's train+val rows for that seed;
- every external test fold is excluded from that seed's unified pretrain;
- lookup, HILIC pool, per-dataset auxiliary Hyper-TL, local-fast candidates, and per-dataset overrides stay disabled.

This keeps the v6 lesson that HILIC needs cross-dataset signal, while making that signal part of one unified model rather than a target-specific lookup or pool.

## Commands

```powershell
python v7\scripts\run_phase.py --phase V7_probe --repo-root . --only-sheet S5 --eval-datasets 0282,0283,0372,0374,0378
python v7\scripts\run_phase.py --phase V7_validate --repo-root .
python v7\scripts\run_phase.py --phase V7_final --repo-root .
```

`V7_probe` is diagnostic only. `V7_validate` uses seeds `71..80`; `V7_final` uses `81..90` and should only be run after the validate report is reviewed.
