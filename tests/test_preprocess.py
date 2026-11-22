import numpy as np
import pytest

from cxrseg.preprocess import (
    DEFAULT_CLAHE,
    ClaheParams,
    binarize_mask,
    enhance,
    preprocess,
    resize_image,
    resize_mask,
    to_gray_uint8,
)


@pytest.fixture
def low_contrast():
    rng = np.random.default_rng(0)
    base = np.linspace(90, 130, 256, dtype=np.float32)[None, :].repeat(256, 0)
    return np.clip(base + rng.normal(0, 3, (256, 256)), 0, 255).astype(np.uint8)


def test_s0_is_identity_and_range_is_unit(low_contrast):
    assert np.array_equal(enhance(low_contrast, "S0"), low_contrast)
    x = preprocess(low_contrast, "S0")
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0


def test_contrast_scenarios_widen_the_intensity_range(low_contrast):
    base = int(low_contrast.max()) - int(low_contrast.min())
    for scenario in ("S1", "S2"):
        out = enhance(low_contrast, scenario)
        assert int(out.max()) - int(out.min()) > base


def test_s2_equals_s3_with_default_parameters(low_contrast):
    assert np.array_equal(enhance(low_contrast, "S2"), enhance(low_contrast, "S3", DEFAULT_CLAHE))


def test_s3_requires_tuned_parameters(low_contrast):
    with pytest.raises(ValueError):
        enhance(low_contrast, "S3")
    a = enhance(low_contrast, "S3", ClaheParams(1.0, (4, 4)))
    b = enhance(low_contrast, "S3", ClaheParams(4.0, (16, 16)))
    assert not np.array_equal(a, b)


def test_mask_noise_is_binarized_before_resizing():
    m = np.zeros((512, 512), np.uint8)
    m[100:400, 100:400] = 255
    m[0, :5] = 3      # anti-aliasing noise like the one in the collection
    m[200, 200] = 200 # still above 127
    out = resize_mask(m, 256)
    assert set(np.unique(out)) <= {0, 255}
    assert out.shape == (256, 256)
    assert (binarize_mask(m) > 0).sum() == 300 * 300


def test_grayscale_conversion_handles_rgb_and_16bit():
    rgb = np.full((10, 10, 3), 100, np.uint8)
    assert to_gray_uint8(rgb).max() == 100
    g16 = np.full((10, 10), 257 * 50, np.uint16)
    assert to_gray_uint8(g16).max() == 50
    assert resize_image(np.zeros((1000, 800), np.uint8), 256).shape == (256, 256)
