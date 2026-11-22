import numpy as np

from cxrseg.metrics import area_error_pct, coverage, dice, evaluate_pair, hd95, iou


def square(top, left, size, shape=(256, 256)):
    m = np.zeros(shape, dtype=bool)
    m[top : top + size, left : left + size] = True
    return m


def test_identical_masks_are_perfect():
    g = square(50, 50, 100)
    r = evaluate_pair(g, g)
    assert r["dice"] == 1.0 and r["iou"] == 1.0 and r["hd95"] == 0.0
    assert r["coverage"] == 1.0 and r["area_error_pct"] == 0.0 and r["failed"] is False


def test_half_overlap_values():
    g = square(0, 0, 100)
    p = square(0, 50, 100)  # shifted right by half: overlap 50x100 of 100x100 each
    assert np.isclose(dice(p, g), 0.5)
    assert np.isclose(iou(p, g), 1 / 3)
    assert np.isclose(coverage(p, g), 0.5)
    assert area_error_pct(p, g) == 0.0


def test_under_segmentation_has_negative_area_error_and_full_precision():
    g = square(20, 20, 100)
    p = square(20, 20, 50)  # a quarter of the area, fully inside
    assert np.isclose(area_error_pct(p, g), -75.0)
    assert np.isclose(coverage(p, g), 0.25)


def test_hd95_of_translated_square_equals_translation():
    g = square(60, 60, 100)
    p = square(60, 70, 100)  # shifted 10 px horizontally
    # left/right edges are 10 px apart, top/bottom edges overlap in 90 of 100 columns
    assert 9.0 <= hd95(p, g) <= 10.5


def test_hd95_empty_prediction_uses_fallback():
    assert hd95(np.zeros((256, 256), bool), square(10, 10, 50), empty_value=362.0) == 362.0


def test_failure_flag_uses_threshold():
    g = square(0, 0, 100)
    p = square(0, 0, 80)  # dice = 2*6400/(6400+10000) = 0.78
    assert evaluate_pair(p, g, failure_dice=0.90)["failed"] is True
    assert evaluate_pair(p, g, failure_dice=0.70)["failed"] is False
