import numpy as np
import torch

from cxrseg.config import load_config
from cxrseg.dataset import SegDataset
from cxrseg.model import UNet
from cxrseg.train import fit, hard_dice, soft_dice_loss

CFG = load_config()


def _toy_ds(n=8, aug=False):
    rng = np.random.default_rng(0)
    imgs, msks = [], []
    for _ in range(n):
        m = np.zeros((32, 32), np.uint8)
        x = int(rng.integers(4, 12))
        m[8:24, x : x + 10] = 1
        imgs.append((m * 0.7 + 0.1).astype(np.float32))
        msks.append(m)
    return SegDataset(np.stack(imgs), np.stack(msks), CFG["train"]["augmentation"] if aug else None, seed=1)


def test_dice_loss_zero_for_perfect_and_large_for_wrong():
    y = torch.zeros(2, 1, 8, 8)
    y[:, :, 2:6, 2:6] = 1
    perfect = (y * 2 - 1) * 20  # confident logits equal to the mask
    assert soft_dice_loss(perfect, y) < 0.05
    assert soft_dice_loss(-perfect, y) > 0.8


def test_hard_dice_threshold_and_empty():
    y = torch.zeros(1, 1, 4, 4)
    y[..., :2, :] = 1
    assert hard_dice(y.clone(), y).item() == 1.0
    assert hard_dice(torch.zeros_like(y), y).item() == 0.0
    assert hard_dice(torch.zeros_like(y), torch.zeros_like(y)).item() == 1.0


def test_fit_learns_saves_and_resumes(tmp_path):
    cfg = {**CFG, "train": {**CFG["train"], "batch_size": 4}}
    torch.manual_seed(0)
    model = UNet((4, 8), 16)
    res = fit(model, _toy_ds(aug=True), _toy_ds(), cfg, seed=1, max_epochs=6, ckpt_dir=tmp_path, device="cpu", verbose=False)
    assert res["epochs"] == 6 and (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert res["best_dice"] > 0.3 and res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    again = fit(UNet((4, 8), 16), _toy_ds(aug=True), _toy_ds(), cfg, seed=1, max_epochs=8, ckpt_dir=tmp_path, device="cpu", verbose=False)
    assert again["epochs"] == 8 and len(again["history"]) == 8  # continued from epoch 6, did not restart
