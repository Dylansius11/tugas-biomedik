"""Percobaan 2 (docs/PLAN-PERCOBAAN-2.md): convention-aware U-Net, strong augmentation and few-shot adaptation.

Darwin masks draw the *anatomical* lung (including the part behind the heart and the diaphragm); Montgomery and
Shenzhen draw the *visible* lung. One U-Net with one output head per convention learns both without mixing them.
Each training image only trains the head of its own convention. Choosing the head for a new source uses metadata
(which definition its labels follow), never its images. The held-out source is still read only at the end.

Usage:
  python -m cxrseg.exp2 train --arm E2 --fold ho_shenzhen --seed 11
  python -m cxrseg.exp2 fewshot --arm E2 --fold ho_darwin --seed 11 --k 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ROOT, load_config
from .data import fold_split, held_out_source, load_arrays, load_manifest
from .dataset import augment_pair
from .model import UNet
from .preprocess import preprocess
from .train import seed_everything

RESULTS = ROOT / "results" / "exp2"


def augment_strong(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator, a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Wider geometry plus intensity changes (gamma, brightness, contrast, blur, noise). Mask only moves geometrically."""
    img, msk = augment_pair(img, msk, rng, a)
    img = np.clip(img, 0, 1) ** rng.uniform(*a["gamma"])
    m = img.mean()
    img = (img - m) * rng.uniform(*a["contrast"]) + m + rng.uniform(-a["brightness"], a["brightness"])
    if rng.random() < a["blur_p"]:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*a["blur_sigma"]))
    if rng.random() < a["noise_p"]:
        img = img + rng.normal(0, a["noise_std"], img.shape)
    return np.clip(img, 0, 1).astype(np.float32), msk


class ConvDataset(Dataset):
    """Returns (image, mask, head index). aug: None, ('geo', params) or ('strong', params)."""

    def __init__(self, images, masks, heads, aug=None, seed: int = 0):
        self.images, self.masks, self.heads, self.aug, self.seed, self.epoch = images, masks, np.asarray(heads), aug, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img, msk = self.images[i], self.masks[i]
        if self.aug is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            kind, params = self.aug
            img, msk = (augment_strong if kind == "strong" else augment_pair)(img, msk, rng, params)
        return (torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(np.ascontiguousarray(msk).astype(np.float32))[None], int(self.heads[i]))


def head_index(cfg: dict, source: str) -> int:
    e = cfg["exp2"]
    return e["heads"].index(e["convention"][source])


def load_rows(cfg: dict, rows) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    imgs, msks = load_arrays(rows, ROOT / cfg["data"]["processed_dir"])
    x = np.stack([preprocess(i, cfg["exp2"]["scenario"]) for i in imgs])
    heads = np.array([head_index(cfg, s) for s in rows["source"]])
    return rows["id"].tolist(), x, (msks > 0).astype(np.uint8), heads


def pick(logits: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    """(N,H,W) logits of each sample's own head -> (N,1,H,W)."""
    return logits.gather(1, heads.view(-1, 1, 1, 1).expand(-1, 1, *logits.shape[2:]))


def seg_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return nn.functional.binary_cross_entropy_with_logits(logits, y) + dice.mean()


def build(cfg: dict) -> UNet:
    t = cfg["train"]
    return UNet(tuple(t["filters"]), t["bottleneck"], out_ch=len(cfg["exp2"]["heads"]))


@torch.no_grad()
def probs(model: nn.Module, x: np.ndarray, head: int | np.ndarray, device: str, bs: int = 16) -> np.ndarray:
    """float32 sigmoid probabilities (N,H,W) of the given head (one int, or one per image)."""
    model.eval()
    heads = np.broadcast_to(np.asarray(head), (len(x),))
    out = []
    for i in range(0, len(x), bs):
        lg = model(torch.from_numpy(x[i : i + bs])[:, None].to(device)).float()
        out.append(torch.sigmoid(pick(lg, torch.as_tensor(heads[i : i + bs].copy(), device=device)))[:, 0].cpu().numpy())
    return np.concatenate(out)


def val_dice(model, x, y, heads, device, thr) -> float:
    p = probs(model, x, heads, device) >= thr
    t = y > 0
    inter = (p & t).sum((1, 2))
    den = p.sum((1, 2)) + t.sum((1, 2))
    return float(np.where(den > 0, 2 * inter / np.maximum(den, 1), 1.0).mean())


def train_arm(cfg: dict, arm: str, fold: str, seed: int, device: str = "cuda", results: Path = RESULTS) -> None:
    """Full training of one arm/fold/seed with the same schedule as the main runs. Resumable."""
    run = results / arm / fold / f"seed{seed}"
    pred_path = run / "pred.npz"
    if pred_path.exists():
        print(f"skip {arm}/{fold}/seed{seed}")
        return
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    t, e = cfg["train"], cfg["exp2"]
    split = fold_split(load_manifest(ROOT / cfg["data"]["processed_dir"]), fold)
    _, xt, yt, ht = load_rows(cfg, split["train"])
    _, xv, yv, hv = load_rows(cfg, split["val"])
    aug = ("strong", e["strong_aug"]) if arm == "E2" else ("geo", t["augmentation"])
    train_ds = ConvDataset(xt, yt, ht, aug, seed)
    model = build(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=t["lr_patience"] - 1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, generator=g)
    st = {"epoch": 0, "best": -1.0, "bad": 0, "history": [], "finished": False}
    last = run / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); g.set_state(ck["gen"].cpu()); st = ck["state"]
    thr = cfg["postprocess"]["threshold"]
    print(f"{arm}/{fold}/seed{seed}: {len(xt)} train, heads used {sorted(set(ht.tolist()))}", flush=True)
    while not st["finished"] and st["epoch"] < t["max_epochs"]:
        ep = st["epoch"] + 1
        train_ds.set_epoch(ep)
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, h in loader:
            x, y, h = x.to(device), y.to(device), h.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                lg = model(x)
            loss = seg_loss(pick(lg.float(), h), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        v = val_dice(model, xv, yv, hv, device, thr)
        sched.step(v)
        if v > st["best"]:
            st["best"], st["bad"] = v, 0
            torch.save(model.state_dict(), run / "best.pt")
        else:
            st["bad"] += 1
        st["epoch"], st["finished"] = ep, st["bad"] >= t["early_stop_patience"]
        st["history"].append({"epoch": ep, "train_loss": tot / len(xt), "val_dice": v, "seconds": time.time() - t0})
        print(f"  epoch {ep:3d} loss {tot / len(xt):.4f} val {v:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "gen": g.get_state(), "state": st}, last)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    write_pred(cfg, model, fold, pred_path, device, split, trained_heads=set(ht.tolist()))
    (run / "val_dice.txt").write_text(f"{st['best']:.6f}\n")
    print(f"best val {st['best']:.4f} after {st['epoch']} epochs", flush=True)


def target_head(cfg: dict, fold: str, trained_heads: set[int]) -> int:
    """Head used on the held-out source: its own convention if some training source taught it, else 'visible'."""
    h = head_index(cfg, held_out_source(fold))
    return h if h in trained_heads else cfg["exp2"]["heads"].index("visible")


def write_pred(cfg, model, fold, pred_path, device, split, trained_heads) -> None:
    ids, groups, out, used = [], [], [], []
    for group, key in (("val", "val"), ("in_source", "in_source_test"), ("foreign", "foreign")):
        gid, x, _, h = load_rows(cfg, split[key])
        if group == "foreign":
            h = np.full(len(gid), target_head(cfg, fold, trained_heads))
        ids += gid; groups += [group] * len(gid); used.append(h)
        out.append(np.rint(probs(model, x, h, device) * 255).astype(np.uint8))
    np.savez_compressed(pred_path, ids=np.array(ids), group=np.array(groups), prob_u8=np.concatenate(out), head=np.concatenate(used))


def fewshot_split(cfg: dict, fold: str):
    """Adaptation pool = the held-out source's train split; evaluation = its val + test split (never adapted on)."""
    man = load_manifest(ROOT / cfg["data"]["processed_dir"])
    tgt = man[man["source"] == held_out_source(fold)]
    return tgt[tgt["split"] == "train"], tgt[tgt["split"] != "train"]


def fewshot(cfg: dict, arm: str, fold: str, seed: int, k: int, device: str = "cuda", results: Path = RESULTS) -> None:
    """Fine-tune the trained model on k labelled images of the new source, replaying the old training data.

    Fixed schedule (no early stopping), so no extra labelled target images are needed for validation.
    """
    out = results / "fewshot" / arm / fold / f"k{k}_seed{seed}.npz"
    if out.exists():
        return
    fs, e = cfg["exp2"]["fewshot"], cfg["exp2"]
    seed_everything(seed * 1000 + k)
    pool, evalrows = fewshot_split(cfg, fold)
    pick_rows = pool.sample(k, random_state=seed)  # nested by seed is not needed; each k is an independent draw
    split = fold_split(load_manifest(ROOT / cfg["data"]["processed_dir"]), fold)
    _, xa, ya, ha = load_rows(cfg, pick_rows)
    _, xr, yr, hr = load_rows(cfg, split["train"])
    model = build(cfg).to(device)
    model.load_state_dict(torch.load(results / arm / fold / f"seed{seed}" / "best.pt", map_location=device))
    trained = set(hr.tolist())
    th = int(ha[0])
    if th not in trained:  # the new convention's head never got gradients: start it from the visible head
        vis = e["heads"].index("visible")
        with torch.no_grad():
            model.head.weight[th].copy_(model.head.weight[vis]); model.head.bias[th].copy_(model.head.bias[vis])
    aug = ("strong", e["strong_aug"]) if arm == "E2" else ("geo", cfg["train"]["augmentation"])
    dsa, dsr = ConvDataset(xa, ya, ha, aug, seed), ConvDataset(xr, yr, hr, aug, seed + 1)
    rng = np.random.default_rng([seed, k])
    opt = torch.optim.Adam(model.parameters(), lr=fs["lr"])
    for it in range(fs["iters"]):
        dsa.set_epoch(it); dsr.set_epoch(it)
        ia = rng.integers(0, len(dsa), fs["batch_target"]); ir = rng.integers(0, len(dsr), fs["batch_replay"])
        b = [dsa[i] for i in ia] + [dsr[i] for i in ir]
        x = torch.stack([s[0] for s in b]).to(device); y = torch.stack([s[1] for s in b]).to(device)
        h = torch.tensor([s[2] for s in b], device=device)
        model.train()
        loss = seg_loss(pick(model(x), h), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    ids, x, _, h = load_rows(cfg, evalrows)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), prob_u8=np.rint(probs(model, x, th, device) * 255).astype(np.uint8), adapt_ids=np.array(pick_rows["id"].tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "fewshot"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int)
    a = ap.parse_args()
    cfg = load_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd == "train":
        train_arm(cfg, a.arm, a.fold, a.seed, dev)
    else:
        fewshot(cfg, a.arm, a.fold, a.seed, a.k, dev)


if __name__ == "__main__":
    main()
