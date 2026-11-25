import numpy as np
import pandas as pd

from cxrseg.stats import average_over_seeds, bootstrap_mean_diff_ci, holm, is_meaningful, paired_comparison


def test_holm_matches_hand_calculation():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_bootstrap_ci_contains_known_synthetic_difference():
    rng = np.random.default_rng(123)
    diff = rng.normal(loc=0.05, scale=0.01, size=200)
    lo, hi = bootstrap_mean_diff_ci(diff, n_boot=10000, seed=42)
    assert lo < 0.05 < hi
    assert np.isclose((lo + hi) / 2.0, diff.mean(), atol=0.002)


def test_trivial_difference_with_small_p_is_not_meaningful():
    assert is_meaningful(p_holm=1e-5, mean_diff=0.002) is False
    assert is_meaningful(p_holm=0.01, mean_diff=0.015) is True
    assert is_meaningful(p_holm=0.01, mean_diff=-0.012) is True
    assert is_meaningful(p_holm=0.06, mean_diff=0.05) is False


def test_average_over_seeds_and_paired_comparison():
    rows = []
    rng = np.random.default_rng(9)
    for idx in range(60):
        base = 0.88 + 0.05 * rng.random()
        for scen, delta in (("S0", 0.0), ("S2", 0.04)):
            for seed in (11, 22, 33):
                rows.append(
                    {
                        "fold": "ho_darwin",
                        "scenario": scen,
                        "seed": seed,
                        "id": f"darwin/img_{idx}",
                        "source": "darwin",
                        "group": "foreign",
                        "dice": float(base + delta + rng.normal(0, 0.002)),
                        "iou": 0.8,
                        "hd95": 5.0,
                        "coverage": 0.9,
                        "area_error_pct": 0.0,
                        "failed": False,
                    }
                )
    raw = pd.DataFrame(rows)
    avg = average_over_seeds(raw, failure_dice=0.90)
    assert len(avg) == 120  # 60 images * 2 scenarios

    res = paired_comparison(avg, baseline="S0", scenario="S2", source="darwin", n_boot=1000, seed=42)
    assert res["n_images"] == 60
    assert res["ci_low"] < 0.04 < res["ci_high"]
    assert res["p_value"] is not None and res["p_value"] < 1e-6

    res_cov = paired_comparison(
        avg, baseline="S0", scenario="S2", source="darwin", metric="coverage", n_boot=500, seed=42
    )
    assert res_cov["p_value"] is None
