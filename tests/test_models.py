from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import rtbench.models.ensemble as ensemble_module
from rtbench.metrics import compute_metrics
from rtbench.models import CandidateOutput, SplitData


class _DummyFeatureModel:
    def __init__(self, importances: list[float]) -> None:
        self.feature_importances_ = np.asarray(importances, dtype=np.float32)


def _base_train_inputs() -> dict[str, object]:
    split = SplitData(
        train_idx=np.array([0, 1]),
        val_idx=np.array([2, 3]),
        test_idx=np.array([4, 5]),
    )
    y_target = np.array([100.0, 120.0, 130.0, 140.0, 150.0, 160.0], dtype=np.float32)
    return {
        "model_cfg": {
            "ENABLE_HYPER_TL": False,
            "ONLY_HYPER_TL": False,
            "ENABLE_MLP": False,
            "ENABLE_MDL_SUBSET_CANDIDATES": False,
            "ENABLE_TRANSFER_TRANSFORM_CANDIDATES": False,
            "ENABLE_LOCAL_TRANSFORM_CANDIDATES": False,
            "ENABLE_ANCHOR_TL": False,
            "FUSION_TOP_K": 2,
            "FUSION_OBJECTIVE": "mae",
            "FUSION_RANK": "mae",
            "CALIBRATE": True,
            "CLIP_MULT": 1.5,
            "EARLY_STOPPING_ROUNDS": 0,
            "LOCAL_XGB": {
                "dummy_val_pred": [1000.0, 1000.0],
                "dummy_test_pred": [1000.0, 1000.0],
                "feature_importances": [1.0, 1.0],
            },
            "LOCAL_LGBM": {
                "dummy_val_pred": [1000.0, 1000.0],
                "dummy_test_pred": [1000.0, 1000.0],
                "feature_importances": [1.0, 1.0],
            },
        },
        "X_src": np.zeros((2, 2), dtype=np.float32),
        "X_src_mol": np.zeros((2, 2), dtype=np.float32),
        "X_src_cp": np.zeros((2, 1), dtype=np.float32),
        "y_src": np.array([90.0, 110.0], dtype=np.float32),
        "X_target": np.arange(12, dtype=np.float32).reshape(6, 2),
        "X_target_mol": np.arange(12, dtype=np.float32).reshape(6, 2),
        "X_target_cp": np.arange(6, dtype=np.float32).reshape(6, 1),
        "y_target": y_target,
        "y_target_sec": y_target,
        "split": split,
        "seed": 7,
        "source_weight": 0.2,
        "target_weight": 1.0,
        "group_sizes": {"mol": 1, "cp": 1},
    }


class TestEnsembleFusion(unittest.TestCase):
    def test_train_and_ensemble_uses_calibrated_top_k_candidates(self):
        inputs = _base_train_inputs()
        inputs["y_target"] = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float32)
        inputs["y_target_sec"] = np.array([400.0, 500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float32)
        inputs["model_cfg"] = dict(inputs["model_cfg"])
        inputs["model_cfg"]["CLIP_MULT"] = 2.0
        y_val = np.asarray(inputs["y_target_sec"])[inputs["split"].val_idx]

        def fake_tree_models(**kwargs):
            val_pred = np.array([500.0, 700.0], dtype=np.float32)
            test_pred = np.array([700.0, 900.0], dtype=np.float32)
            return [
                CandidateOutput(
                    name="TREE_A",
                    val_pred=val_pred,
                    test_pred=test_pred,
                    val_metrics=compute_metrics(kwargs["y_val_sec"], val_pred),
                    model=_DummyFeatureModel([3.0, 1.0]),
                )
            ]

        def fake_build_candidates(_ctx):
            val_pred = np.array([400.0, 700.0], dtype=np.float32)
            test_pred = np.array([600.0, 900.0], dtype=np.float32)
            return fake_tree_models(y_val_sec=y_val) + [
                CandidateOutput(
                    name="RIDGE_B",
                    val_pred=val_pred,
                    test_pred=test_pred,
                    val_metrics=compute_metrics(y_val, val_pred),
                    model={"type": "ridge"},
                )
            ]

        with mock.patch.object(ensemble_module, "build_candidates", side_effect=fake_build_candidates):
            result = ensemble_module.train_and_ensemble(**inputs)

        self.assertEqual(result.top_models, ["TREE_A", "RIDGE_B"])
        self.assertAlmostEqual(sum(result.weights), 1.0, places=6)
        np.testing.assert_allclose(result.pred_val, y_val, atol=1e-5)
        self.assertAlmostEqual(result.feature_group_importance["mol"], 0.75, places=6)
        self.assertAlmostEqual(result.feature_group_importance["cp"], 0.25, places=6)
        self.assertTrue(np.all(result.pred_test <= 1000.0))

    def test_train_and_ensemble_breaks_mae_ties_with_r2(self):
        inputs = _base_train_inputs()
        inputs["model_cfg"] = dict(inputs["model_cfg"])
        inputs["model_cfg"].update(
            {
                "FUSION_TOP_K": 1,
                "FUSION_RANK": "mae_then_r2",
                "CALIBRATE": False,
                "CLIP_MULT": 10.0,
                "LOCAL_XGB": {
                    "dummy_val_pred": [220.0, 220.0],
                    "dummy_test_pred": [220.0, 220.0],
                    "feature_importances": [1.0, 1.0],
                },
                "LOCAL_LGBM": {
                    "dummy_val_pred": [230.0, 230.0],
                    "dummy_test_pred": [230.0, 230.0],
                    "feature_importances": [1.0, 1.0],
                },
            }
        )
        inputs["y_target"] = np.array([80.0, 90.0, 100.0, 140.0, 150.0, 160.0], dtype=np.float32)
        inputs["y_target_sec"] = np.array([80.0, 90.0, 100.0, 140.0, 150.0, 160.0], dtype=np.float32)
        y_val = np.asarray(inputs["y_target_sec"])[inputs["split"].val_idx]

        def fake_tree_models(**kwargs):
            lower_r2 = np.array([100.0, 160.0], dtype=np.float32)
            higher_r2 = np.array([110.0, 130.0], dtype=np.float32)
            return [
                CandidateOutput(
                    name="LOWER_R2",
                    val_pred=lower_r2,
                    test_pred=np.array([160.0, 180.0], dtype=np.float32),
                    val_metrics=compute_metrics(kwargs["y_val_sec"], lower_r2),
                    model=_DummyFeatureModel([1.0, 1.0]),
                ),
                CandidateOutput(
                    name="HIGHER_R2",
                    val_pred=higher_r2,
                    test_pred=np.array([155.0, 165.0], dtype=np.float32),
                    val_metrics=compute_metrics(kwargs["y_val_sec"], higher_r2),
                    model=_DummyFeatureModel([1.0, 3.0]),
                ),
            ]

        with mock.patch.object(ensemble_module, "build_candidates", side_effect=lambda _ctx: fake_tree_models(y_val_sec=y_val)):
            result = ensemble_module.train_and_ensemble(**inputs)

        self.assertEqual(result.top_models, ["HIGHER_R2"])
        np.testing.assert_allclose(result.pred_val, np.array([110.0, 130.0], dtype=np.float32), atol=1e-6)
        self.assertEqual(len(result.weights), 1)
        self.assertAlmostEqual(result.weights[0], 1.0, places=6)
        self.assertAlmostEqual(result.feature_group_importance["mol"], 0.25, places=6)
        self.assertAlmostEqual(result.feature_group_importance["cp"], 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
