import pandas as pd

from v7.scripts.summarize_phase import _failure_table


def test_failure_table_prioritizes_largest_shortfall() -> None:
    comp = pd.DataFrame(
        [
            {
                "phase": "V7_validate",
                "sheet": "S5",
                "dataset": "0283",
                "our_mae_mean": 14.0,
                "our_r2_mean": 0.12,
                "uni_rt_mae": 12.0,
                "uni_rt_r2": 0.67,
                "delta_mae": -2.0,
                "delta_r2": -0.55,
                "beat_mae": False,
                "beat_r2": False,
                "beat_both": False,
                "seed_count": 10,
            },
            {
                "phase": "V7_validate",
                "sheet": "S5",
                "dataset": "0376",
                "our_mae_mean": 43.0,
                "our_r2_mean": 0.87,
                "uni_rt_mae": 54.0,
                "uni_rt_r2": 0.88,
                "delta_mae": 11.0,
                "delta_r2": -0.01,
                "beat_mae": True,
                "beat_r2": False,
                "beat_both": False,
                "seed_count": 10,
            },
            {
                "phase": "V7_validate",
                "sheet": "S5",
                "dataset": "0377",
                "our_mae_mean": 33.0,
                "our_r2_mean": 0.91,
                "uni_rt_mae": 56.0,
                "uni_rt_r2": 0.87,
                "delta_mae": 23.0,
                "delta_r2": 0.04,
                "beat_mae": True,
                "beat_r2": True,
                "beat_both": True,
                "seed_count": 10,
            },
        ]
    )

    failures = _failure_table(comp)

    assert failures["dataset"].tolist() == ["0283", "0376"]
    assert failures.loc[failures["dataset"] == "0283", "failure_reason"].item() == "mae,r2"
    assert failures.loc[failures["dataset"] == "0376", "failure_reason"].item() == "r2"
    assert failures["priority_score"].iloc[0] > failures["priority_score"].iloc[1]
