# Data Provenance

## RepoRT Source

- Upstream repository: `https://github.com/michaelwitting/RepoRT`
- Pinned commit used by the benchmark configs: `c95c1aba24e5472fb2d8d3f0fbd5efe202b3cd60`
- Local checkout root expected by the benchmark: `data/repoRT/`
- Processed study files expected by the runtime: `data/repoRT/processed_data/<dataset_id>/`

## What This Project Uses

This repository consumes the processed RepoRT study files referenced by the benchmark configs. The code does not rewrite the upstream raw study definitions; it reads the processed metadata, gradient, RT, descriptor, and fingerprint files required by the benchmark pipeline.

## Licensing And Usage

Before redistributing RepoRT-derived data, verify the upstream repository license, any data-specific notices, and any publication terms associated with the processed study files. This project pins a specific upstream commit for reproducibility, but downstream teams are still responsible for reviewing the original dataset terms before reuse or redistribution.

## Reproducibility Notes

- Keep the pinned commit hash with every benchmark handoff.
- Record any local preprocessing or file repairs separately if they differ from the upstream RepoRT commit.
- Preserve `config.resolved.yaml` and `config.sha1` for every benchmark run so a registry entry can be traced back to the exact data and config scope used at runtime.
