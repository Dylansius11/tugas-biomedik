import numpy as np
import torch

from cxrseg.config import load_config
from cxrseg.exp2 import ConvDataset, augment_strong, head_index, pick, target_head
from cxrseg.explain import Reference, features, hist, tta_probs


def _pair():
    img = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    msk = np.zeros((64, 64), np.uint8)
    msk[16:48, 10:30] = 1
    return img, msk


def test_strong_aug_keeps_mask_binary_and_image_in_range():
    cfg = load_config()
    img, msk = _pair()
    for s in range(5):
        i2, m2 = augment_strong(img, msk, np.random.default_rng(s), cfg["exp2"]["strong_aug"])
        assert set(np.unique(m2)) <= {0, 1} and 0 <= i2.min() and i2.max() <= 1 and i2.dtype == np.float32


def test_each_sample_uses_its_own_head():
    logits = torch.stack([torch.zeros(4, 4), torch.ones(4, 4)])[None].repeat(2, 1, 1, 1)  # (2,2,4,4): head0=0, head1=1
    out = pick(logits, torch.tensor([0, 1]))
    assert out.shape == (2, 1, 4, 4) and out[0].sum() == 0 and out[1].sum() == 16


def test_convention_heads_and_target_head():
    cfg = load_config()
    vis, ana = head_index(cfg, "montgomery"), head_index(cfg, "darwin")
    assert vis != ana and head_index(cfg, "shenzhen") == vis
    assert target_head(cfg, "ho_shenzhen", {vis, ana}) == vis
    assert target_head(cfg, "ho_darwin", {vis}) == vis  # anatomical head never trained: fall back to visible


def test_dataset_returns_head():
    img, msk = _pair()
    ds = ConvDataset(img[None], msk[None], [1])
    x, y, h = ds[0]
    assert x.shape == (1, 64, 64) and h == 1


def test_tta_identity_model_has_zero_spread_inside():
    x = np.random.default_rng(0).random((2, 64, 64)).astype(np.float32)
    mean, std = tta_probs(lambda z: np.full_like(z, 0.7), x, 4, [0.95, 1.05])
    assert np.allclose(std[:, 16:48, 16:48], 0, atol=1e-6) and np.allclose(mean[:, 16:48, 16:48], 0.7)


def test_reference_flags_outlier_with_reason():
    rng = np.random.default_rng(0)
    x8 = (rng.random((64, 64)) * 255).astype(np.uint8)
    ref_h = hist(x8)
    prob = np.zeros((64, 64), np.float32)
    prob[10:50, 5:28] = 0.95
    prob[10:50, 36:59] = 0.95
    base = [features(x8, prob * rng.uniform(0.98, 1.0), np.zeros_like(prob), ref_h, 0.15) for _ in range(40)]
    ref = Reference(base)
    small = prob.copy()
    small[30:, :] = 0
    f = features(x8, small, np.zeros_like(prob), ref_h, 0.15)
    assert ref.a3([f])[0] > ref.tau
    assert any("luas paru prediksi lebih kecil" in r for r in ref.reasons(f))
