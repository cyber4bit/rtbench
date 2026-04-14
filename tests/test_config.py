from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml

import rtbench.run as run_entry
from rtbench.bench import config_sha1_from_raw
from rtbench.config import resolve_config


def _base_config() -> dict:
    return {
        "data": {
            "repo_url": "https://example.com/repo",
            "commit": "abc123",
            "local_root": "data/repoRT",
            "baseline_csv": "data/baseline/demo.csv",
            "gradient_points": 20,
        },
        "datasets": {
            "pretrain": ["0019", "0052"],
            "external": ["0028"],
            "expected_pretrain_count": 2,
        },
        "split": {"train": 0.8, "val": 0.1, "test": 0.1, "strategy": "stratified"},
        "models": {"FUSION_TOP_K": 3, "CALIBRATE": False},
        "transfer_weights": {"source": 0.2, "target": 1.0, "adaptive_source": False, "target_transform": "none"},
        "seeds": {"default": "0:4"},
        "metrics": {"paper_avg_mae": 1.0, "paper_avg_r2": 0.0, "required_win_both": 1},
        "stats": {"test": "wilcoxon", "correction": "bh_fdr", "fdr_q": 0.05},
        "outputs": {"root": "outputs_demo", "resume": False},
    }


class TestConfigResolution(unittest.TestCase):
    def test_all_repo_benchmark_configs_resolve(self):
        config_root = Path("configs")
        failures: list[str] = []
        for path in sorted(config_root.rglob("*.yaml")):
            if path.parent.name == "_bases":
                continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and {"runs", "policy"}.intersection(set(raw.keys())):
                continue
            try:
                resolve_config(path)
            except Exception as exc:
                failures.append(f"{path.as_posix()}: {exc}")
        self.assertEqual(failures, [])

    def test_multiple_base_configs_merge_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_a = root / "base_a.yaml"
            base_b = root / "base_b.yaml"
            child = root / "child.yaml"

            cfg_a = _base_config()
            cfg_a["models"]["CALIBRATE"] = False
            cfg_a["transfer_weights"]["source"] = 0.15
            cfg_a["outputs"]["root"] = "outputs_a"

            cfg_b = {
                "models": {"CALIBRATE": True},
                "transfer_weights": {"target": 2.0},
                "outputs": {"root": "outputs_b"},
            }

            base_a.write_text(yaml.safe_dump(cfg_a, sort_keys=False), encoding="utf-8")
            base_b.write_text(yaml.safe_dump(cfg_b, sort_keys=False), encoding="utf-8")
            child.write_text(
                yaml.safe_dump(
                    {
                        "_base": ["base_a.yaml", "base_b.yaml"],
                        "models": {"FUSION_TOP_K": 9},
                        "outputs": {"root": "outputs_child"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            resolved = resolve_config(child)

            self.assertEqual(resolved.config.models["FUSION_TOP_K"], 9)
            self.assertTrue(resolved.config.models["CALIBRATE"])
            self.assertEqual(resolved.config.transfer_weights["source"], 0.15)
            self.assertEqual(resolved.config.transfer_weights["target"], 2.0)
            self.assertEqual(resolved.config.outputs["root"], "outputs_child")
            self.assertEqual(
                tuple(Path(x).name for x in resolved.base_chain),
                ("base_a.yaml", "base_b.yaml", "child.yaml"),
            )

    def test_base_inheritance_merges_nested_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.yaml"
            child_path = root / "child.yaml"

            base_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")
            child_path.write_text(
                yaml.safe_dump(
                    {
                        "_base": "base.yaml",
                        "models": {"FUSION_TOP_K": 8},
                        "transfer_weights": {"adaptive_source": True},
                        "outputs": {"root": "outputs_child"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            resolved = resolve_config(child_path)

            self.assertEqual(resolved.config.models["FUSION_TOP_K"], 8)
            self.assertFalse(resolved.config.models["CALIBRATE"])
            self.assertTrue(resolved.config.transfer_weights["adaptive_source"])
            self.assertEqual(resolved.config.outputs["root"], "outputs_child")
            self.assertEqual(resolved.config.datasets["external"], ["0028"])
            self.assertEqual(Path(resolved.base_chain[0]).name, "base.yaml")
            self.assertEqual(Path(resolved.base_chain[-1]).name, "child.yaml")

    def test_overrides_apply_after_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = root / "base.yaml"
            child_path = root / "child.yaml"

            base_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")
            child_path.write_text(
                yaml.safe_dump({"_base": "base.yaml", "outputs": {"root": "outputs_child"}}, sort_keys=False),
                encoding="utf-8",
            )

            resolved = resolve_config(
                child_path,
                overrides=[
                    "models.FUSION_TOP_K=8",
                    "transfer_weights.adaptive_source=true",
                    "seeds.default=0:9",
                    "models.HYPER_TL.n_models=5",
                    "models.LOCAL_TARGET_TRANSFORMS=[log1p, gradient_norm]",
                ],
            )

            self.assertEqual(resolved.config.models["FUSION_TOP_K"], 8)
            self.assertTrue(resolved.config.transfer_weights["adaptive_source"])
            self.assertEqual(resolved.config.seeds["default"], "0:9")
            self.assertEqual(resolved.raw["models"]["HYPER_TL"]["n_models"], 5)
            self.assertEqual(resolved.raw["models"]["LOCAL_TARGET_TRANSFORMS"], ["log1p", "gradient_norm"])
            self.assertEqual(
                resolved.overrides,
                (
                    "models.FUSION_TOP_K=8",
                    "transfer_weights.adaptive_source=true",
                    "seeds.default=0:9",
                    "models.HYPER_TL.n_models=5",
                    "models.LOCAL_TARGET_TRANSFORMS=[log1p, gradient_norm]",
                ),
            )

    def test_run_cli_applies_config_overrides_and_runtime_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "base.yaml"
            config_path.write_text(yaml.safe_dump(_base_config(), sort_keys=False), encoding="utf-8")

            expected_summary = pd.DataFrame(
                [
                    {
                        "dataset": "0052",
                        "paper_mae": 10.0,
                        "paper_r2": 0.5,
                        "our_mae_mean": 9.0,
                        "our_r2_mean": 0.6,
                        "delta_mae": 1.0,
                        "delta_r2": 0.1,
                        "p_mae": 0.5,
                        "p_r2": 0.5,
                        "p_adj_mae": 0.5,
                        "p_adj_r2": 0.5,
                        "win_both": False,
                    }
                ]
            )

            argv = [
                "rtbench.run",
                "--config",
                str(config_path),
                "--seeds",
                "2:4",
                "--datasets",
                "52,180",
                "--override",
                "models.FUSION_TOP_K=8",
                "transfer_weights.adaptive_source=true",
                "--override",
                "outputs.root=outputs_cli",
                "--no-download",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_entry, "configure_logging"),
                mock.patch.object(run_entry, "write_effective_config_snapshot") as write_snapshot,
                mock.patch.object(run_entry, "prepare", return_value=object()) as prepare_mock,
                mock.patch.object(run_entry, "record_experiment") as record_mock,
                mock.patch.object(
                    run_entry,
                    "run_trial",
                    return_value=SimpleNamespace(
                        out_root=root / "outputs_cli",
                        summary_df=expected_summary,
                        avg_mae=9.0,
                        avg_r2=0.6,
                        wins=1,
                        success=True,
                    ),
                ) as run_trial_mock,
            ):
                run_entry.main()

            resolved = resolve_config(
                config_path,
                overrides=[
                    "models.FUSION_TOP_K=8",
                    "transfer_weights.adaptive_source=true",
                    "outputs.root=outputs_cli",
                ],
            )
            self.assertEqual(prepare_mock.call_args.kwargs["external_ids"], ["0052", "0180"])
            self.assertTrue(prepare_mock.call_args.kwargs["no_download"])
            self.assertEqual(run_trial_mock.call_args.kwargs["seeds"], [2, 3, 4])
            self.assertEqual(run_trial_mock.call_args.kwargs["external_ids"], ["0052", "0180"])
            self.assertEqual(run_trial_mock.call_args.kwargs["config_sha1"], config_sha1_from_raw(resolved.raw))
            self.assertEqual(run_trial_mock.call_args.args[1].models["FUSION_TOP_K"], 8)
            self.assertTrue(run_trial_mock.call_args.args[1].transfer_weights["adaptive_source"])
            self.assertEqual(run_trial_mock.call_args.args[1].outputs["root"], "outputs_cli")
            self.assertEqual(write_snapshot.call_args.kwargs["config_raw"]["outputs"]["root"], "outputs_cli")
            self.assertTrue(record_mock.called)
            self.assertEqual(
                record_mock.call_args.kwargs["extra_hparams"],
                {
                    "config.overrides": [
                        "models.FUSION_TOP_K=8",
                        "transfer_weights.adaptive_source=true",
                        "outputs.root=outputs_cli",
                    ]
                },
            )


if __name__ == "__main__":
    unittest.main()
