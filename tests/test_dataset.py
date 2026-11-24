import numpy as np
import pandas as pd
import pytest

from cxrseg.config import load_config
from cxrseg.dataset import SegDataset, augment_pair, load_s3_params

AUG = load_config()["train"]["augmentation"]


def _toy():
    img = np.zeros((64, 64), np.float32)
    msk = np.zeros((64, 64), np.uint8)
    img[20:44, 10:30] = 1.0
    msk[20:44, 10:30] = 1
    return img, msk


def test_augment_keeps_mask_binary_and_aligned():
    img, msk = _toy()
    rng = np.random.default_rng(0)
    for _ in range(20):
        i2, m2 = augment_pair(img, msk, rng, AUG)
        assert set(np.unique(m2)) <= {0, 1}
        assert np.abs((i2 > 0.5).astype(int) - m2).mean() < 0.02  # image and mask moved together


def test_no_flip():
    img, msk = _toy()  # object sits on the left; flips would put its centre of mass on the right
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, m2 = augment_pair(img, msk, rng, AUG)
        if m2.any():
            assert np.nonzero(m2)[1].mean() < 32


def test_dataset_is_deterministic_per_epoch_and_changes_across_epochs():
    img, msk = _toy()
    ds = SegDataset(img[None], msk[None], AUG, seed=11)
    a, _ = ds[0]
    b, _ = ds[0]
    assert (a == b).all()
    ds.set_epoch(1)
    c, _ = ds[0]
    assert not (a == c).all()


def test_validation_not_augmented():
    img, msk = _toy()
    x, y = SegDataset(img[None], msk[None])[0]
    assert x.shape == (1, 64, 64) and (x[0].numpy() == img).all() and (y[0].numpy() == msk).all()


def test_s3_needs_params_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_s3_params("ho_darwin", tmp_path / "missing.csv")
    p = tmp_path / "s3.csv"
    pd.DataFrame([{"fold": "ho_darwin", "clip_limit": 3.0, "tile_grid": "8x8"}]).to_csv(p, index=False)
    got = load_s3_params("ho_darwin", p)
    assert got.clip_limit == 3.0 and got.tile_grid == (8, 8)
    with pytest.raises(ValueError):
        load_s3_params("ho_shenzhen", p)
