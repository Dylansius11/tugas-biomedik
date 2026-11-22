import numpy as np

from cxrseg.postprocess import postprocess_mask


def blob(prob, top, left, h, w, value=0.9):
    prob[top : top + h, left : left + w] = value
    return prob


def test_keeps_only_two_largest_components():
    p = np.zeros((256, 256), np.float32)
    blob(p, 20, 20, 80, 60)    # left lung
    blob(p, 20, 150, 80, 60)   # right lung
    blob(p, 200, 100, 8, 8)    # small false positive
    out = postprocess_mask(p)
    assert out[50, 50] and out[50, 180]
    assert not out[204, 104]


def test_fills_holes_and_closes_thin_gaps():
    p = np.zeros((256, 256), np.float32)
    blob(p, 20, 20, 100, 100)
    p[60:70, 60:70] = 0.0      # interior hole
    p[20:120, 69:71] = 0.0     # 2 px wide crack, narrower than the 5x5 closing kernel
    out = postprocess_mask(p)
    assert out[65, 65]         # hole filled
    assert out[50, 70]         # crack closed


def test_accepts_uint8_probabilities_and_threshold():
    p = np.zeros((256, 256), np.uint8)
    p[50:100, 50:100] = 140    # 0.549 >= 0.5
    p[150:200, 150:200] = 120  # 0.47 < 0.5
    out = postprocess_mask(p)
    assert out[75, 75] and not out[175, 175]
