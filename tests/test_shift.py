import numpy as np

from cxrseg.config import load_config
from cxrseg.shift import (
    alarm_rate,
    alarm_threshold,
    compute_alarm_evaluation,
    decision_from_alarm,
    intensity_distance,
    mean_entropy,
    pooled_hist,
    readiness_label,
)


def test_intensity_distance_zero_when_matched_and_increases_on_shift():
    rng = np.random.default_rng(42)
    ref_img = rng.integers(40, 120, size=(64, 64), dtype=np.uint8)
    ref_hist = pooled_hist(ref_img[None, ...])

    assert np.isclose(intensity_distance(ref_img, ref_hist), 0.0, atol=1e-12)

    brighter = np.clip(ref_img.astype(np.int16) + 50, 0, 255).astype(np.uint8)
    dist = intensity_distance(brighter, ref_hist)
    assert dist > 0.1
    even_brighter = np.clip(ref_img.astype(np.int16) + 100, 0, 255).astype(np.uint8)
    assert intensity_distance(even_brighter, ref_hist) > dist


def test_mean_entropy_bounds_and_maximum():
    zeros = np.zeros((32, 32), dtype=np.uint8)
    ones = np.full((32, 32), 255, dtype=np.uint8)
    half = np.full((32, 32), 0.5, dtype=np.float32)

    assert mean_entropy(zeros) == 0.0
    assert mean_entropy(ones) == 0.0
    assert np.isclose(mean_entropy(half), np.log(2.0), atol=1e-6)

    # uint8 near 0.5 (127 or 128) is close to ln(2)
    half_u8 = np.full((32, 32), 128, dtype=np.uint8)
    assert np.isclose(mean_entropy(half_u8), np.log(2.0), atol=1e-3)


def test_alarm_rate_on_validation_scores_is_around_five_percent():
    rng = np.random.default_rng(7)
    val_scores = rng.normal(0.1, 0.02, size=1000)
    tau = alarm_threshold(val_scores, percentile=95)
    rate = alarm_rate(val_scores, tau)
    assert 0.03 <= rate <= 0.07


def test_readiness_rules_and_auroc_guard():
    cfg = load_config()
    assert readiness_label(0.04, 0.95, 0.96, cfg) == "siap"
    assert readiness_label(0.06, 0.95, 0.96, cfg) == "belum siap"
    assert readiness_label(0.02, 0.92, 0.96, cfg) == "belum siap"

    assert decision_from_alarm(0.08, cfg) == "siap"
    assert decision_from_alarm(0.12, cfg) == "belum siap"

    # Fewer than 10 failures -> "tidak dapat dihitung"
    res_small = compute_alarm_evaluation(
        scores=np.linspace(0, 1, 50),
        dice=np.linspace(0.95, 0.85, 50),
        failed=[i >= 45 for i in range(50)],
    )
    assert res_small["auroc"] == "tidak dapat dihitung"

    # >= 10 failures -> numeric AUROC
    res_ok = compute_alarm_evaluation(
        scores=np.linspace(0, 1, 50),
        dice=np.linspace(0.95, 0.80, 50),
        failed=[i >= 35 for i in range(50)],
    )
    assert isinstance(res_ok["auroc"], float) and res_ok["auroc"] > 0.9
