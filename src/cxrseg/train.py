"""Train one run: python -m cxrseg.train --fold ho_shenzhen --scenario S0 --seed 11

Follows docs/PROTOCOL.md sections 5 and 6. All numbers come from configs/default.yaml.
Golden rule: the held-out source is only read when writing the 'foreign' predictions at the very end.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ROOT, load_config
from .dataset import SegDataset, load_group, make_train_val
from .model import build_model, count_params
from .preprocess import ClaheParams


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + smooth) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + smooth)).mean()


def hard_dice(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Per-image Dice of the thresholded probability, no post-processing."""
    p = (prob >= threshold).float()
    denom = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return torch.where(denom > 0, 2 * (p * target).sum((1, 2, 3)) / denom.clamp(min=1), torch.ones_like(denom))


@torch.no_grad()
def predict_probs(model: nn.Module, images: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """float32 (N,H,W) in [0,1] -> uint8 (N,H,W) sigmoid probability x 255, rounded."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i : i + batch_size])[:, None].to(device)
        out.append(torch.sigmoid(model(x).float())[:, 0].cpu().numpy())
    return np.rint(np.concatenate(out) * 255).astype(np.uint8)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str, threshold: float) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        scores.append(hard_dice(torch.sigmoid(model(x).float()), y, threshold).cpu())
    return float(torch.cat(scores).mean())


def fit(
    model: nn.Module,
    train_ds: SegDataset,
    val_ds: SegDataset,
    cfg: dict,
    seed: int,
    max_epochs: int,
    ckpt_dir: Path | None = None,
    device: str = "cuda",
    amp: bool = True,
    verbose: bool = True,
) -> dict:
    """Train with BCE + Dice loss, ReduceLROnPlateau and early stopping on validation Dice.

    Saves best.pt (only when validation Dice improves) and last.pt (every epoch, for resuming).
    Returns {'best_dice', 'epochs', 'history'}.
    """
    t = cfg["train"]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    # patience=lr_patience-1: halve after exactly lr_patience epochs without improvement
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=t["lr_patience"] - 1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device == "cuda")
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False)
    bce = nn.BCEWithLogitsLoss()
    thr = cfg["postprocess"]["threshold"]

    state = {"epoch": 0, "best": -1.0, "bad": 0, "history": [], "finished": False}
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        last = ckpt_dir / "last.pt"
        if last.exists():
            ck = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            g.set_state(ck["gen"].cpu())
            state = ck["state"]
            if verbose:
                print(f"resuming from epoch {state['epoch']} (best val dice {state['best']:.4f})")

    while not state["finished"] and state["epoch"] < max_epochs:
        epoch = state["epoch"] + 1
        train_ds.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = bce(logits, y) + soft_dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            n += len(x)
        val = validate(model, val_loader, device, thr)
        sched.step(val)
        improved = val > state["best"]
        if improved:
            state["best"], state["bad"] = val, 0
            if ckpt_dir is not None:
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            state["bad"] += 1
        state["epoch"] = epoch
        state["finished"] = state["bad"] >= t["early_stop_patience"]
        secs = time.time() - t0
        state["history"].append({"epoch": epoch, "train_loss": total / n, "val_dice": val, "lr": opt.param_groups[0]["lr"], "seconds": secs})
        if verbose:
            mem = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
            print(f"epoch {epoch:3d}  loss {total / n:.4f}  val dice {val:.4f}{' *' if improved else '  '}  lr {opt.param_groups[0]['lr']:.2e}  {secs:5.1f}s  gpu {mem:.2f} GiB", flush=True)
        if ckpt_dir is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "gen": g.get_state(), "state": state}, ckpt_dir / "last.pt")
    return {"best_dice": state["best"], "epochs": state["epoch"], "history": state["history"]}


def write_predictions(model: nn.Module, cfg: dict, fold: str, scenario: str, pred_path: Path, device: str) -> None:
    """The only place where the held-out ('foreign') source is read. Contract: docs/PROTOCOL.md section 6."""
    ids, groups, probs = [], [], []
    for group, key in (("val", "val"), ("in_source", "in_source_test"), ("foreign", "foreign")):
        gid, x, _ = load_group(cfg, fold, scenario, key)
        ids += gid
        groups += [group] * len(gid)
        probs.append(predict_probs(model, x, device, cfg["train"]["batch_size"]))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, ids=np.array(ids), group=np.array(groups), prob_u8=np.concatenate(probs))


def run_dirs(results: Path, fold: str, scenario: str, seed: int) -> tuple[Path, Path]:
    tail = Path(fold) / scenario / f"seed{seed}"
    return results / "checkpoints" / tail, results / "predictions" / tail / "pred.npz"


def train_run(cfg: dict, fold: str, scenario: str, seed: int, results: Path, max_epochs: int | None = None, device: str | None = None, amp: bool = True) -> dict | None:
    ckpt_dir, pred_path = run_dirs(results, fold, scenario, seed)
    if pred_path.exists():
        print(f"skip {fold}/{scenario}/seed{seed}: pred.npz exists")
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    train_ds, val_ds = make_train_val(cfg, fold, scenario, seed)
    model = build_model(cfg)
    print(f"{fold}/{scenario}/seed{seed}: {len(train_ds)} train, {len(val_ds)} val images, {count_params(model):,} parameters, device {device}")
    res = fit(model, train_ds, val_ds, cfg, seed, max_epochs or cfg["train"]["max_epochs"], ckpt_dir, device, amp)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    write_predictions(model, cfg, fold, scenario, pred_path, device)
    print(f"best val dice {res['best_dice']:.4f} after {res['epochs']} epochs; wrote {pred_path}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, help="override train.max_epochs (smoke test only)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    train_run(load_config(), a.fold, a.scenario, a.seed, ROOT / a.results, a.max_epochs, amp=not a.no_amp)


if __name__ == "__main__":
    main()
