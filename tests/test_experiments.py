from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from rtbench.bench import config_sha1_from_raw
from rtbench.experiments import (
    archive_effective_config,
    compare_experiments,
    garbage_collect_experiments,
    list_tmp_cleanup_candidates,
    load_registry,
    migrate_registry,
    query_experiments,
)


def _write_run(
    root: Path,
    *,
    output_root: str,
    dataset_rows: list[dict[str, object]],
    config_name: str | None = None,
    config_raw: dict[str, object] | None = None,
) -> Path:
    if config_name and config_raw is not None:
        (root / "configs").mkdir(exist_ok=True)
        config_path = root / "configs" / config_name
        config_path.write_text(yaml.safe_dump(config_raw, sort_keys=False), encoding="utf-8")
        config_sha1 = hashlib.sha1(config_path.read_bytes()).hexdigest()
    else:
        config_sha1 = ""

    run_dir = root / output_root
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(dataset_rows).to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
    if config_sha1:
        run_dir.joinpath("config.sha1").write_text(config_sha1, encoding="utf-8")
    return run_dir


class TestExperimentRegistry(unittest.TestCase):
    def test_migrate_registry_uses_config_catalog_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs").mkdir()
            config_path = root / "configs" / "demo.yaml"
            raw_cfg = {
                "data": {"local_root": "data/repoRT"},
                "datasets": {"pretrain": ["0019"], "external": ["0028"]},
                "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
                "models": {"FUSION_TOP_K": 3, "CALIBRATE": True},
                "transfer_weights": {"adaptive_source": True, "target_transform": "none"},
                "seeds": {"default": "0:1"},
                "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
                "stats": {"fdr_q": 0.05},
                "outputs": {"root": "outputs_demo"},
            }
            config_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

            run_dir = root / "outputs_demo"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"dataset": "0028", "our_mae_mean": 10.0, "our_r2_mean": 0.80, "win_both": True},
                    {"dataset": "0032", "our_mae_mean": 12.0, "our_r2_mean": 0.90, "win_both": False},
                ]
            ).to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
            pd.DataFrame(
                [
                    {"dataset": "0028", "seed": 0},
                    {"dataset": "0028", "seed": 1},
                    {"dataset": "0032", "seed": 0},
                    {"dataset": "0032", "seed": 1},
                ]
            ).to_csv(metrics_dir / "per_seed.csv", index=False, encoding="utf-8")
            run_dir.joinpath("config.sha1").write_text(
                hashlib.sha1(config_path.read_bytes()).hexdigest(),
                encoding="utf-8",
            )

            migrate_registry(root)

            registry = pd.read_csv(root / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
            self.assertEqual(len(registry), 1)
            row = registry.iloc[0]
            self.assertEqual(row["run_dir"], "outputs_demo")
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["config_path"], "configs/demo.yaml")
            self.assertEqual(row["config_hash_type"], "file_sha1")
            self.assertEqual(row["effective_config_sha1"], config_sha1_from_raw(raw_cfg))
            self.assertEqual(row["effective_config_path"], f"experiments/configs/{config_sha1_from_raw(raw_cfg)}.yaml")
            self.assertEqual(row["avg_mae"], "11.000000")
            self.assertEqual(row["avg_r2"], "0.850000")
            self.assertEqual(row["win_both"], "1")
            self.assertEqual(row["seed_count"], "2")
            self.assertIn('"models.FUSION_TOP_K": 3', row["key_hparams"])
            self.assertTrue((root / row["effective_config_path"]).exists())

    def test_tmp_runs_are_marked_cleanable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "outputs_tmp_demo"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"dataset": "0276", "our_mae_mean": 9.5, "our_r2_mean": 0.91, "win_both": True}]
            ).to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")

            migrate_registry(root)

            registry = pd.read_csv(root / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
            self.assertEqual(len(registry), 1)
            row = registry.iloc[0]
            self.assertEqual(row["status"], "tmp")
            self.assertEqual(row["cleanable"], "true")
            self.assertEqual(row["run_dir"], "outputs_tmp_demo")

            manifest = (root / "experiments" / "cleanup_candidates.txt").read_text(encoding="utf-8")
            self.assertIn("outputs_tmp_demo", manifest)
            self.assertEqual([p.name for p in list_tmp_cleanup_candidates(root)], ["outputs_tmp_demo"])

    def test_archived_config_aliases_are_preserved_in_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs").mkdir()
            canonical_path = root / "configs" / "canonical.yaml"
            archived_rel = "configs/archived_dup.yaml"
            raw_cfg = {
                "data": {"local_root": "data/repoRT", "repo_url": "https://example.com/repo", "commit": "abc123", "baseline_csv": "data/baseline/demo.csv", "gradient_points": 20},
                "datasets": {"pretrain": ["0019"], "external": ["0028"], "expected_pretrain_count": 1},
                "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
                "models": {"FUSION_TOP_K": 3, "CALIBRATE": True},
                "transfer_weights": {"source": 0.2, "target": 1.0, "adaptive_source": False, "target_transform": "none"},
                "seeds": {"default": "0:1"},
                "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
                "stats": {"fdr_q": 0.05, "test": "wilcoxon", "correction": "bh_fdr"},
                "outputs": {"root": "outputs_canonical"},
            }
            canonical_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
            archived_file_sha1 = hashlib.sha1(b"archived duplicate config").hexdigest()

            manifest_dir = root / "experiments"
            manifest_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "archived_config_path": archived_rel,
                        "archived_file_sha1": archived_file_sha1,
                        "canonical_config_path": "configs/canonical.yaml",
                        "canonical_normalized_sha1": config_sha1_from_raw(raw_cfg),
                        "note": "Archived duplicate of canonical config",
                    }
                ]
            ).to_csv(manifest_dir / "archived_configs.csv", index=False, encoding="utf-8")

            run_dir = root / "outputs_archived"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"dataset": "0028", "our_mae_mean": 10.0, "our_r2_mean": 0.80, "win_both": True}]
            ).to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
            run_dir.joinpath("config.sha1").write_text(archived_file_sha1, encoding="utf-8")

            migrate_registry(root)

            registry = pd.read_csv(root / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
            row = registry.iloc[0]
            self.assertEqual(row["config_path"], archived_rel)
            self.assertEqual(row["archived"], "true")
            self.assertIn("Archived duplicate", row["archived_note"])

    def test_archived_aliases_can_resolve_via_effective_config_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "configs").mkdir()
            raw_cfg = {
                "data": {"local_root": "data/repoRT", "repo_url": "https://example.com/repo", "commit": "abc123", "baseline_csv": "data/baseline/demo.csv", "gradient_points": 20},
                "datasets": {"pretrain": ["0019"], "external": ["0028"], "expected_pretrain_count": 1},
                "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
                "models": {"FUSION_TOP_K": 3, "CALIBRATE": True},
                "transfer_weights": {"source": 0.2, "target": 1.0, "adaptive_source": False, "target_transform": "none"},
                "seeds": {"default": "0:1"},
                "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
                "stats": {"fdr_q": 0.05, "test": "wilcoxon", "correction": "bh_fdr"},
                "outputs": {"root": "outputs_archived_snapshot"},
            }
            _, normalized_sha1 = archive_effective_config(root, config_raw=raw_cfg)
            archived_rel = "configs/deleted_tmp.yaml"
            archived_file_sha1 = hashlib.sha1(b"deleted tmp config").hexdigest()

            manifest_dir = root / "experiments"
            manifest_dir.mkdir(exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "archived_config_path": archived_rel,
                        "archived_file_sha1": archived_file_sha1,
                        "canonical_config_path": "",
                        "canonical_normalized_sha1": normalized_sha1,
                        "note": "Archived temp config preserved via effective snapshot",
                    }
                ]
            ).to_csv(manifest_dir / "archived_configs.csv", index=False, encoding="utf-8")

            run_dir = root / "outputs_archived_snapshot"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"dataset": "0028", "our_mae_mean": 10.0, "our_r2_mean": 0.80, "win_both": True}]
            ).to_csv(metrics_dir / "summary_vs_paper.csv", index=False, encoding="utf-8")
            run_dir.joinpath("config.sha1").write_text(archived_file_sha1, encoding="utf-8")

            migrate_registry(root)

            registry = pd.read_csv(root / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
            row = registry.iloc[0]
            self.assertEqual(row["config_path"], archived_rel)
            self.assertEqual(row["archived"], "true")
            self.assertIn("effective snapshot", row["archived_note"])

    def test_query_experiments_ranks_by_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_cfg = {
                "data": {"local_root": "data/repoRT"},
                "datasets": {"pretrain": ["0019"], "external": ["0028"]},
                "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
                "models": {"FUSION_TOP_K": 3},
                "transfer_weights": {"source": 0.2, "target": 1.0},
                "seeds": {"default": "0:1"},
                "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
                "stats": {"fdr_q": 0.05},
                "outputs": {"root": "placeholder"},
            }
            cfg_a = dict(base_cfg, outputs={"root": "outputs_rank_a"})
            cfg_b = dict(base_cfg, outputs={"root": "outputs_rank_b"})
            _write_run(
                root,
                output_root="outputs_rank_a",
                config_name="rank_a.yaml",
                config_raw=cfg_a,
                dataset_rows=[
                    {"dataset": "0028", "our_mae_mean": 12.0, "our_r2_mean": 0.81, "win_both": True},
                    {"dataset": "0032", "our_mae_mean": 14.0, "our_r2_mean": 0.82, "win_both": True},
                ],
            )
            _write_run(
                root,
                output_root="outputs_rank_b",
                config_name="rank_b.yaml",
                config_raw=cfg_b,
                dataset_rows=[
                    {"dataset": "0028", "our_mae_mean": 8.0, "our_r2_mean": 0.84, "win_both": True},
                    {"dataset": "0032", "our_mae_mean": 10.0, "our_r2_mean": 0.86, "win_both": True},
                ],
            )

            migrate_registry(root)

            ranked = query_experiments(root, metric="avg_mae", sort="asc", top=1)
            self.assertEqual(list(ranked["run_dir"]), ["outputs_rank_b"])
            self.assertEqual(list(ranked["avg_mae"]), ["9.000000"])

            ranked_r2 = query_experiments(root, metric="avg_r2", sort="desc", top=2)
            self.assertEqual(list(ranked_r2["run_dir"]), ["outputs_rank_b", "outputs_rank_a"])

    def test_compare_experiments_returns_dataset_diffs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run(
                root,
                output_root="outputs_compare_a",
                dataset_rows=[
                    {"dataset": "0028", "paper_mae": 11.0, "paper_r2": 0.70, "our_mae_mean": 10.0, "our_r2_mean": 0.80, "delta_mae": 1.0, "delta_r2": 0.10, "win_both": True},
                    {"dataset": "0032", "paper_mae": 21.0, "paper_r2": 0.60, "our_mae_mean": 20.0, "our_r2_mean": 0.70, "delta_mae": 1.0, "delta_r2": 0.10, "win_both": True},
                ],
            )
            _write_run(
                root,
                output_root="outputs_compare_b",
                dataset_rows=[
                    {"dataset": "0028", "paper_mae": 11.0, "paper_r2": 0.70, "our_mae_mean": 9.0, "our_r2_mean": 0.82, "delta_mae": 2.0, "delta_r2": 0.12, "win_both": True},
                    {"dataset": "0032", "paper_mae": 21.0, "paper_r2": 0.60, "our_mae_mean": 21.0, "our_r2_mean": 0.75, "delta_mae": 0.0, "delta_r2": 0.15, "win_both": False},
                    {"dataset": "0040", "paper_mae": 16.0, "paper_r2": 0.50, "our_mae_mean": 15.0, "our_r2_mean": 0.60, "delta_mae": 1.0, "delta_r2": 0.10, "win_both": True},
                ],
            )

            migrate_registry(root)
            payload = compare_experiments(root, "outputs_compare_a", "outputs_compare_b")

            self.assertEqual(payload["summary"]["dataset_overlap_count"], 2)
            self.assertEqual(payload["summary"]["run_a_only_count"], 0)
            self.assertEqual(payload["summary"]["run_b_only_count"], 1)
            self.assertEqual(payload["summary"]["run_b_better_mae_count"], 1)
            self.assertEqual(payload["summary"]["run_b_better_r2_count"], 2)
            self.assertEqual(payload["summary"]["run_b_better_both_count"], 1)
            self.assertAlmostEqual(payload["summary"]["avg_mae_delta_b_minus_a"], 0.0)
            self.assertAlmostEqual(payload["summary"]["avg_r2_delta_b_minus_a"], -0.026667, places=6)
            self.assertEqual(payload["datasets"][0]["dataset"], "0028")
            self.assertTrue(payload["datasets"][0]["run_b_better_both"])

    def test_gc_dry_run_and_delete_only_tmp_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run(
                root,
                output_root="outputs_tmp_a",
                dataset_rows=[{"dataset": "0028", "our_mae_mean": 9.0, "our_r2_mean": 0.80, "win_both": True}],
            )
            _write_run(
                root,
                output_root="outputs_tmp_b",
                dataset_rows=[{"dataset": "0032", "our_mae_mean": 10.0, "our_r2_mean": 0.81, "win_both": False}],
            )
            _write_run(
                root,
                output_root="outputs_keep",
                dataset_rows=[{"dataset": "0040", "our_mae_mean": 11.0, "our_r2_mean": 0.82, "win_both": True}],
            )

            migrate_registry(root)

            preview = garbage_collect_experiments(root, status="tmp", dry_run=True)
            self.assertEqual(preview["candidate_count"], 2)
            self.assertEqual(preview["deleted_count"], 0)
            self.assertTrue((root / "outputs_tmp_a").exists())
            self.assertTrue((root / "outputs_tmp_b").exists())

            applied = garbage_collect_experiments(root, status="tmp", dry_run=False)
            self.assertEqual(applied["deleted_count"], 2)
            self.assertFalse((root / "outputs_tmp_a").exists())
            self.assertFalse((root / "outputs_tmp_b").exists())
            self.assertTrue((root / "outputs_keep").exists())

            registry = pd.read_csv(root / "experiments" / "registry.csv", dtype=str, encoding="utf-8").fillna("")
            self.assertEqual(sorted(registry["run_dir"].tolist()), ["outputs_keep"])

    def test_query_experiments_returns_empty_frame_when_no_matching_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run(
                root,
                output_root="outputs_failed_only",
                dataset_rows=[{"dataset": "0028", "our_mae_mean": 10.0, "our_r2_mean": 0.8, "win_both": False}],
            )
            migrate_registry(root)
            registry_path = root / "experiments" / "registry.csv"
            registry = pd.read_csv(registry_path, dtype=str, encoding="utf-8").fillna("")
            registry.loc[:, "status"] = "failed"
            registry.to_csv(registry_path, index=False, encoding="utf-8")

            ranked = query_experiments(root, metric="avg_mae", sort="asc", top=5)

            self.assertTrue(ranked.empty)
            self.assertEqual(
                list(ranked.columns),
                ["run_dir", "status", "avg_mae", "avg_r2", "win_both", "dataset_count", "seed_count", "config_path", "archived"],
            )

    def test_compare_experiments_raises_for_missing_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run(
                root,
                output_root="outputs_compare_present",
                dataset_rows=[{"dataset": "0028", "our_mae_mean": 9.0, "our_r2_mean": 0.8, "win_both": True}],
            )
            migrate_registry(root)

            with self.assertRaises(KeyError):
                compare_experiments(root, "outputs_compare_present", "outputs_missing")

    def test_load_registry_recovers_from_corrupted_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_path = root / "experiments" / "registry.csv"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text('"run_dir,status\n"unterminated', encoding="utf-8")

            registry, resolved_path = load_registry(root, refresh=False)

            self.assertEqual(resolved_path, registry_path)
            self.assertTrue(registry.empty)
            self.assertEqual(list(registry.columns), [
                "experiment_name",
                "run_dir",
                "output_root",
                "run_date",
                "status",
                "cleanable",
                "archived",
                "archived_note",
                "config_sha1",
                "config_hash_type",
                "config_path",
                "config_source",
                "effective_config_path",
                "effective_config_sha1",
                "summary_path",
                "avg_mae",
                "avg_r2",
                "win_both",
                "dataset_count",
                "seed_count",
                "key_hparams",
                "error",
            ])

    def test_load_registry_uses_env_override_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alt_registry = root / "custom" / "registry_alt.csv"
            _write_run(
                root,
                output_root="outputs_env_registry",
                dataset_rows=[{"dataset": "0028", "our_mae_mean": 9.5, "our_r2_mean": 0.81, "win_both": True}],
            )

            previous = os.environ.get("RTBENCH_REGISTRY_PATH")
            try:
                os.environ["RTBENCH_REGISTRY_PATH"] = str(alt_registry)
                registry, resolved_path = load_registry(root, refresh=True)
            finally:
                if previous is None:
                    os.environ.pop("RTBENCH_REGISTRY_PATH", None)
                else:
                    os.environ["RTBENCH_REGISTRY_PATH"] = previous

            self.assertEqual(resolved_path, alt_registry)
            self.assertTrue(alt_registry.exists())
            self.assertEqual(registry.iloc[0]["run_dir"], "outputs_env_registry")

    def test_gc_returns_empty_payload_without_tmp_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = garbage_collect_experiments(root, status="tmp", dry_run=True)
            self.assertEqual(payload["candidate_count"], 0)
            self.assertEqual(payload["matching_run_count"], 0)
            self.assertEqual(payload["deleted_count"], 0)
            self.assertEqual(payload["candidate_roots"], [])


if __name__ == "__main__":
    unittest.main()
