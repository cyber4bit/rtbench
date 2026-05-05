from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import rtbench.models.ensemble as ensemble_module
import rtbench.models.candidates.hyper_candidates as hyper_candidates_module
from rtbench.models.candidates.common import CandidateBuildContext
from rtbench.metrics import compute_metrics
from rtbench.models import CandidateOutput, SplitData


class _DummyFeatureModel:
    def __init__(self, importances: list[float]) -> None:
        self.feature_importances_ = np.asarray(importances, dtype=np.float32)


class _DummyHyperBundle:
    ridge_lambdas = [1.0]
    ridge_lambda_b = 0.0
    cp_mean = np.zeros(1, dtype=np.float32)
    use_conditioned_embeddings = False
    use_task_adapters = False


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
    def test_fusion_target_quantile_rule_supports_per_rule_blend(self):
        rules = ensemble_module._parse_fusion_target_quantile_rules(
            [
                {"pattern": "class=A", "quantile": 1.0, "blend": 0.25},
                "class=B=>0.0",
            ]
        )

        events: list[dict[str, object]] = []
        pred = ensemble_module._apply_fusion_target_quantile_rules(
            np.array([10.0, 10.0, 10.0], dtype=np.float32),
            [("class=A",), ("class=B",), ("class=C",)],
            np.array([100.0, 200.0, 300.0], dtype=np.float32),
            [("class=A",), ("class=B",), ("class=C",)],
            rules,
            blend=1.0,
            diagnostics=events,
        )

        np.testing.assert_allclose(pred, np.array([82.5, 100.0, 10.0], dtype=np.float32), atol=1e-6)
        self.assertEqual(len(events), 2)
        summary = ensemble_module._summarize_quantile_rule_events("test", events)
        self.assertEqual(summary["fusion_quantile_test_adjusted_count"], 2)
        self.assertEqual(summary["fusion_quantile_test_rule_count"], 2)
        self.assertGreater(summary["fusion_quantile_test_max_abs_shift"], 0.0)
        self.assertIn("class=A", summary["fusion_quantile_test_rules"])

        capped_events: list[dict[str, object]] = []
        capped = ensemble_module._apply_fusion_target_quantile_rules(
            np.array([10.0], dtype=np.float32),
            [("class=A",)],
            np.array([100.0, 200.0, 300.0], dtype=np.float32),
            [("class=A",)],
            rules[:1],
            blend=1.0,
            diagnostics=capped_events,
            max_abs_shift=5.0,
        )

        np.testing.assert_allclose(capped, np.array([15.0], dtype=np.float32), atol=1e-6)
        capped_summary = ensemble_module._summarize_quantile_rule_events("test", capped_events)
        self.assertEqual(capped_summary["fusion_quantile_test_capped_count"], 1)
        self.assertEqual(capped_summary["fusion_quantile_test_max_abs_shift"], 5.0)

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

    def test_priority_min_val_r2_demotes_bad_priority_candidate(self):
        inputs = _base_train_inputs()
        inputs["model_cfg"] = dict(inputs["model_cfg"])
        inputs["model_cfg"].update(
            {
                "FUSION_TOP_K": 1,
                "FUSION_PRIORITY_PATTERNS": [r"^PRIORITY_BAD$"],
                "CALIBRATE": False,
                "CLIP_MULT": 10.0,
                "LOCAL_XGB": {
                    "dummy_val_pred": [130.0, 140.0],
                    "dummy_test_pred": [150.0, 160.0],
                    "feature_importances": [1.0, 1.0],
                },
                "LOCAL_LGBM": {
                    "dummy_val_pred": [130.0, 140.0],
                    "dummy_test_pred": [150.0, 160.0],
                    "feature_importances": [1.0, 1.0],
                },
            }
        )
        y_val = np.asarray(inputs["y_target_sec"])[inputs["split"].val_idx]

        def fake_build_candidates(_ctx):
            priority_bad = np.array([170.0, 100.0], dtype=np.float32)
            regular_good = np.array([130.0, 140.0], dtype=np.float32)
            return [
                CandidateOutput(
                    name="PRIORITY_BAD",
                    val_pred=priority_bad,
                    test_pred=np.array([190.0, 120.0], dtype=np.float32),
                    val_metrics=compute_metrics(y_val, priority_bad),
                    model=_DummyFeatureModel([1.0, 1.0]),
                ),
                CandidateOutput(
                    name="REGULAR_GOOD",
                    val_pred=regular_good,
                    test_pred=np.array([150.0, 160.0], dtype=np.float32),
                    val_metrics=compute_metrics(y_val, regular_good),
                    model=_DummyFeatureModel([1.0, 3.0]),
                ),
            ]

        with mock.patch.object(ensemble_module, "build_candidates", side_effect=fake_build_candidates):
            result = ensemble_module.train_and_ensemble(**inputs)
        self.assertEqual(result.top_models, ["PRIORITY_BAD"])

        inputs["model_cfg"]["FUSION_PRIORITY_MIN_VAL_R2"] = 0.0
        with mock.patch.object(ensemble_module, "build_candidates", side_effect=fake_build_candidates):
            result = ensemble_module.train_and_ensemble(**inputs)

        self.assertEqual(result.top_models, ["REGULAR_GOOD"])
        np.testing.assert_allclose(result.pred_val, y_val, atol=1e-6)

    def test_candidate_diagnostics_selected_requires_nonzero_weight(self):
        inputs = _base_train_inputs()
        inputs["model_cfg"] = dict(inputs["model_cfg"])
        inputs["model_cfg"].update({"FUSION_TOP_K": 2, "CALIBRATE": False, "CLIP_MULT": 10.0})
        y_val = np.asarray(inputs["y_target_sec"])[inputs["split"].val_idx]

        def fake_build_candidates(_ctx):
            return [
                CandidateOutput(
                    name="USED",
                    val_pred=np.array([130.0, 140.0], dtype=np.float32),
                    test_pred=np.array([150.0, 160.0], dtype=np.float32),
                    val_metrics=compute_metrics(y_val, np.array([130.0, 140.0], dtype=np.float32)),
                    model=_DummyFeatureModel([1.0, 1.0]),
                ),
                CandidateOutput(
                    name="ZERO_WEIGHT",
                    val_pred=np.array([131.0, 141.0], dtype=np.float32),
                    test_pred=np.array([151.0, 161.0], dtype=np.float32),
                    val_metrics=compute_metrics(y_val, np.array([131.0, 141.0], dtype=np.float32)),
                    model={
                        "similarity_diagnostics": {
                            "n_test": 2,
                            "spaces": {
                                "descriptor": {
                                    "test_vs_train": {
                                        "test_distance_percentile_max": 0.99,
                                        "test_nn_distance_median": 1.5,
                                    }
                                }
                            },
                        }
                    },
                ),
            ]

        with (
            mock.patch.object(ensemble_module, "build_candidates", side_effect=fake_build_candidates),
            mock.patch.object(ensemble_module, "_optimize_weights", return_value=np.array([1.0, 0.0], dtype=np.float32)),
        ):
            result = ensemble_module.train_and_ensemble(**inputs)

        diagnostics = {row["name"]: row for row in result.candidate_diagnostics or []}
        self.assertTrue(diagnostics["USED"]["selected"])
        self.assertFalse(diagnostics["ZERO_WEIGHT"]["selected"])
        self.assertEqual(diagnostics["ZERO_WEIGHT"]["weight"], 0.0)
        self.assertIn("FUSION_PRE_QUANTILE", diagnostics)
        self.assertTrue(diagnostics["FUSION_PRE_QUANTILE"]["selected"])
        self.assertEqual(diagnostics["ZERO_WEIGHT"]["sim_n_test"], 2)
        self.assertAlmostEqual(
            diagnostics["ZERO_WEIGHT"]["sim_descriptor_test_vs_train_test_distance_percentile_max"],
            0.99,
            places=6,
        )

    def test_low_iqr_filter_removes_configured_candidate_family(self):
        candidates = [
            CandidateOutput(
                name="HYPER_TL_ENS(n=3)",
                val_pred=np.array([1.0], dtype=np.float32),
                test_pred=np.array([1.0], dtype=np.float32),
                val_metrics={"mae": 1.0},
                model={},
            ),
            CandidateOutput(
                name="HYPER_PRIOR_CAL_ISOTONIC(n=3)",
                val_pred=np.array([2.0], dtype=np.float32),
                test_pred=np.array([2.0], dtype=np.float32),
                val_metrics={"mae": 2.0},
                model={},
            ),
        ]
        cfg = {
            "FUSION_LOW_IQR_MAX_REF_IQR": 10.0,
            "FUSION_LOW_IQR_DENY_PATTERNS": [r"^HYPER_TL_ENS"],
        }

        filtered = ensemble_module._filter_low_iqr_candidates(
            candidates,
            cfg,
            np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float32),
        )

        self.assertEqual([candidate.name for candidate in filtered], ["HYPER_PRIOR_CAL_ISOTONIC(n=3)"])

    def test_low_iqr_filter_keeps_candidates_for_wide_reference_range(self):
        candidates = [
            CandidateOutput(
                name="HYPER_TL_ENS(n=3)",
                val_pred=np.array([1.0], dtype=np.float32),
                test_pred=np.array([1.0], dtype=np.float32),
                val_metrics={"mae": 1.0},
                model={},
            ),
            CandidateOutput(
                name="HYPER_PRIOR_CAL_ISOTONIC(n=3)",
                val_pred=np.array([2.0], dtype=np.float32),
                test_pred=np.array([2.0], dtype=np.float32),
                val_metrics={"mae": 2.0},
                model={},
            ),
        ]
        cfg = {
            "FUSION_LOW_IQR_MAX_REF_IQR": 10.0,
            "FUSION_LOW_IQR_DENY_PATTERNS": [r"^HYPER_TL_ENS"],
        }

        filtered = ensemble_module._filter_low_iqr_candidates(
            candidates,
            cfg,
            np.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype=np.float32),
        )

        self.assertEqual(filtered, candidates)

    def test_close_prior_calibration_filter_prefers_isotonic(self):
        linear = CandidateOutput(
            name="HYPER_PRIOR_CAL_LINEAR(n=3)",
            val_pred=np.array([100.0], dtype=np.float32),
            test_pred=np.array([110.0], dtype=np.float32),
            val_metrics={"mae": 10.0},
            model={},
        )
        isotonic = CandidateOutput(
            name="HYPER_PRIOR_CAL_ISOTONIC(n=3)",
            val_pred=np.array([101.0], dtype=np.float32),
            test_pred=np.array([109.0], dtype=np.float32),
            val_metrics={"mae": 10.4},
            model={},
        )
        other = CandidateOutput(
            name="HYPER_EMB_LGBM",
            val_pred=np.array([102.0], dtype=np.float32),
            test_pred=np.array([108.0], dtype=np.float32),
            val_metrics={"mae": 12.0},
            model={},
        )
        candidates = [linear, isotonic, other]

        self.assertEqual(
            ensemble_module._filter_close_linear_prior_calibration_candidates(candidates, {}),
            candidates,
        )
        self.assertEqual(
            ensemble_module._filter_close_linear_prior_calibration_candidates(
                candidates,
                {
                    "FUSION_PREFER_ISOTONIC_PRIOR_CAL_MAX_MAE_DELTA": 0.5,
                    "FUSION_PREFER_ISOTONIC_PRIOR_CAL_MIN_TARGET_ROWS": 100,
                },
                target_rows=40,
            ),
            candidates,
        )

        filtered = ensemble_module._filter_close_linear_prior_calibration_candidates(
            candidates,
            {"FUSION_PREFER_ISOTONIC_PRIOR_CAL_MAX_MAE_DELTA": 0.5},
            target_rows=120,
        )

        self.assertEqual([candidate.name for candidate in filtered], ["HYPER_PRIOR_CAL_ISOTONIC(n=3)", "HYPER_EMB_LGBM"])

    def test_prior_calibration_diagnostics_report_slope_and_extrapolation(self):
        fit = hyper_candidates_module._calibrate_prior_with_diagnostics(
            mode="linear",
            fit_pred=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            fit_y=np.array([2.0, 4.0, 6.0], dtype=np.float32),
            eval_pred=np.array([0.0, 2.0, 4.0], dtype=np.float32),
        )

        self.assertIsNotNone(fit)
        pred, diagnostics = fit
        np.testing.assert_allclose(pred, np.array([0.0, 4.0, 8.0], dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(diagnostics["slope"]), 2.0, places=6)
        self.assertAlmostEqual(float(diagnostics["fit_pred_range"]), 2.0, places=6)
        self.assertAlmostEqual(float(diagnostics["eval_extrapolation_frac"]), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(float(diagnostics["eval_extrapolation_ratio"]), 0.5, places=6)

    def test_prior_calibration_summary_exposes_fusion_metrics(self):
        summary = hyper_candidates_module._summarize_prior_calibration_diagnostics(
            mode="linear",
            val_diags=[
                {
                    "mode": "linear",
                    "slope": 1.5,
                    "fit_pred_range": 20.0,
                    "eval_extrapolation_frac": 0.0,
                    "eval_extrapolation_ratio": 0.0,
                    "eval_to_fit_range_ratio": 1.0,
                }
            ],
            test_diags=[
                {
                    "mode": "linear",
                    "slope": 3.0,
                    "fit_pred_range": 10.0,
                    "eval_extrapolation_frac": 0.5,
                    "eval_extrapolation_ratio": 1.2,
                    "eval_to_fit_range_ratio": 2.5,
                }
            ],
        )

        self.assertEqual(summary["mode"], "linear")
        self.assertEqual(summary["n"], 1)
        self.assertAlmostEqual(summary["test"]["slope_abs_max"], 3.0, places=6)
        self.assertAlmostEqual(summary["test"]["fit_pred_range_min"], 10.0, places=6)
        self.assertAlmostEqual(summary["test"]["eval_extrapolation_frac_max"], 0.5, places=6)

    def test_risky_linear_prior_calibration_filter_is_default_off(self):
        candidates = [
            CandidateOutput(
                name="HYPER_PRIOR_CAL_LINEAR(n=1)",
                val_pred=np.array([100.0], dtype=np.float32),
                test_pred=np.array([150.0], dtype=np.float32),
                val_metrics={"mae": 1.0},
                model={
                    "prior_calibration": {
                        "mode": "linear",
                        "val": {"eval_extrapolation_frac_max": 0.0},
                        "test": {
                            "slope_abs_max": 9.0,
                            "fit_pred_range_min": 1.0,
                            "eval_extrapolation_frac_max": 1.0,
                            "eval_extrapolation_ratio_max": 5.0,
                            "eval_to_fit_range_ratio_max": 8.0,
                        },
                    }
                },
            )
        ]

        filtered = ensemble_module._filter_risky_linear_prior_calibration_candidates(candidates, {})

        self.assertEqual(filtered, candidates)

    def test_risky_linear_prior_calibration_filter_suppresses_threshold_violations(self):
        linear = CandidateOutput(
            name="HYPER_PRIOR_CAL_LINEAR(n=1)",
            val_pred=np.array([100.0], dtype=np.float32),
            test_pred=np.array([150.0], dtype=np.float32),
            val_metrics={"mae": 1.0},
            model={
                "prior_calibration": {
                    "mode": "linear",
                    "val": {"eval_extrapolation_frac_max": 0.0},
                    "test": {
                        "slope_abs_max": 6.0,
                        "fit_pred_range_min": 8.0,
                        "eval_extrapolation_frac_max": 0.75,
                        "eval_extrapolation_ratio_max": 2.0,
                        "eval_to_fit_range_ratio_max": 3.0,
                    },
                }
            },
        )
        isotonic = CandidateOutput(
            name="HYPER_PRIOR_CAL_ISOTONIC(n=1)",
            val_pred=np.array([101.0], dtype=np.float32),
            test_pred=np.array([140.0], dtype=np.float32),
            val_metrics={"mae": 1.2},
            model={"prior_calibration": {"mode": "isotonic", "val": {}, "test": {}}},
        )
        other = CandidateOutput(
            name="HYPER_EMB_LGBM",
            val_pred=np.array([102.0], dtype=np.float32),
            test_pred=np.array([130.0], dtype=np.float32),
            val_metrics={"mae": 2.0},
            model={},
        )

        filtered = ensemble_module._filter_risky_linear_prior_calibration_candidates(
            [linear, isotonic, other],
            {
                "FUSION_PRIOR_CAL_LINEAR_MAX_ABS_SLOPE": 5.0,
                "FUSION_PRIOR_CAL_LINEAR_MAX_TEST_EXTRAP_FRAC": 0.5,
                "FUSION_PRIOR_CAL_LINEAR_FILTER_REQUIRE_ISOTONIC": True,
            },
        )

        self.assertEqual([candidate.name for candidate in filtered], ["HYPER_PRIOR_CAL_ISOTONIC(n=1)", "HYPER_EMB_LGBM"])

    def test_risky_linear_prior_calibration_filter_preserves_candidates_without_required_isotonic(self):
        linear = CandidateOutput(
            name="HYPER_PRIOR_CAL_LINEAR(n=1)",
            val_pred=np.array([100.0], dtype=np.float32),
            test_pred=np.array([150.0], dtype=np.float32),
            val_metrics={"mae": 1.0},
            model={
                "prior_calibration": {
                    "mode": "linear",
                    "val": {"eval_extrapolation_frac_max": 0.0},
                    "test": {
                        "slope_abs_max": 6.0,
                        "fit_pred_range_min": 8.0,
                        "eval_extrapolation_frac_max": 0.75,
                        "eval_extrapolation_ratio_max": 2.0,
                        "eval_to_fit_range_ratio_max": 3.0,
                    },
                }
            },
        )

        filtered = ensemble_module._filter_risky_linear_prior_calibration_candidates(
            [linear],
            {
                "FUSION_PRIOR_CAL_LINEAR_MAX_ABS_SLOPE": 5.0,
                "FUSION_PRIOR_CAL_LINEAR_FILTER_REQUIRE_ISOTONIC": True,
            },
        )

        self.assertEqual(filtered, [linear])


class TestCandidateSimilarityDiagnostics(unittest.TestCase):
    def test_distance_percentile_summary_flags_extrapolated_test_rows(self):
        summary = hyper_candidates_module._distance_percentile_summary(
            np.array([[0.1], [10.0]], dtype=np.float32),
            np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["n_ref"], 3.0)
        self.assertGreater(summary["test_distance_percentile_max"], 0.99)
        self.assertGreater(summary["test_nn_distance_max"], summary["test_nn_distance_median"])

    def test_similarity_diagnostics_are_default_off_and_attach_when_enabled(self):
        ctx = CandidateBuildContext(
            model_cfg={"ENABLE_HYPER_TL": True},
            X_src=np.zeros((3, 2), dtype=np.float32),
            X_src_mol=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            X_src_cp=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
            y_src=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            y_src_sec=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            X_train=np.zeros((2, 2), dtype=np.float32),
            X_train_mol=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            X_train_cp=np.array([[0.0], [0.0]], dtype=np.float32),
            y_train=np.array([10.0, 20.0], dtype=np.float32),
            y_train_sec=np.array([10.0, 20.0], dtype=np.float32),
            X_val=np.zeros((2, 2), dtype=np.float32),
            X_val_mol=np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            y_val_used=np.array([30.0, 40.0], dtype=np.float32),
            y_val_sec=np.array([30.0, 40.0], dtype=np.float32),
            X_test=np.zeros((2, 2), dtype=np.float32),
            X_test_mol=np.array([[0.5, 0.5], [8.0, 8.0]], dtype=np.float32),
            seed=1,
            target_transform="none",
            hyper_bundle=_DummyHyperBundle(),
            X_val_cp=np.array([[0.0], [0.0]], dtype=np.float32),
            X_test_cp=np.array([[0.0], [3.0]], dtype=np.float32),
        )

        def fake_ridge_prior_fit_predict(*, Z_eval, **_kwargs):
            return np.full(len(Z_eval), 25.0, dtype=np.float32)

        with (
            mock.patch.object(hyper_candidates_module, "head_prior", return_value=(np.ones(2, dtype=np.float32), 0.0)),
            mock.patch.object(
                hyper_candidates_module,
                "ridge_prior_fit_predict",
                side_effect=fake_ridge_prior_fit_predict,
            ),
            mock.patch.object(
                hyper_candidates_module,
                "mol_embeddings",
                side_effect=lambda _bundle, X_mol, *_args, **_kwargs: np.asarray(X_mol, dtype=np.float32),
            ),
        ):
            candidates = hyper_candidates_module.build_hyper_candidates(ctx)

        self.assertTrue(candidates)
        self.assertNotIn("similarity_diagnostics", candidates[0].model)

        enabled_ctx = CandidateBuildContext(
            **{
                **ctx.__dict__,
                "model_cfg": {"ENABLE_HYPER_TL": True, "ENABLE_CANDIDATE_SIMILARITY_DIAGNOSTICS": True},
            }
        )
        with (
            mock.patch.object(hyper_candidates_module, "head_prior", return_value=(np.ones(2, dtype=np.float32), 0.0)),
            mock.patch.object(
                hyper_candidates_module,
                "ridge_prior_fit_predict",
                side_effect=fake_ridge_prior_fit_predict,
            ),
            mock.patch.object(
                hyper_candidates_module,
                "mol_embeddings",
                side_effect=lambda _bundle, X_mol, *_args, **_kwargs: np.asarray(X_mol, dtype=np.float32),
            ),
        ):
            enabled_candidates = hyper_candidates_module.build_hyper_candidates(enabled_ctx)

        diagnostics = enabled_candidates[0].model["similarity_diagnostics"]
        self.assertEqual(diagnostics["n_test"], 2)
        self.assertIn("descriptor", diagnostics["spaces"])
        self.assertIn("embedding", diagnostics["spaces"])
        self.assertIn("cp", diagnostics["spaces"])
        self.assertIn("test_vs_train", diagnostics["spaces"]["descriptor"])


if __name__ == "__main__":
    unittest.main()
