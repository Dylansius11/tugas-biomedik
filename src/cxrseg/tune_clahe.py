"""Tune CLAHE per fold: python -m cxrseg.tune_clahe --fold ho_darwin   (docs/PROTOCOL.md section 5)

12 combinations (clip_limit x tile_grid), each trained for tuning.epochs with tuning.seed, scored by the best
validation Dice on the two training sources. The held-out source is never read here.
Writes results/tables/tuning_{fold}.csv and updates the fold's row in results/tables/s3_params.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import ROOT, load_config
from .dataset import make_train_val
from .model import build_model
from .preprocess import ClaheParams
from .train import fit, seed_everything


def grid_label(tile: tuple[int, int]) -> str:
    return f"{tile[0]}x{tile[1]}"


def pick_winner(df: pd.DataFrame) -> pd.Series:
    """Highest val_dice; ties go to the earliest row (grid order)."""
    return df.loc[df["val_dice"].idxmax()]


def update_s3_params(path: Path, fold: str, clip_limit: float, tile_grid: str) -> pd.DataFrame:
    cols = ["fold", "clip_limit", "tile_grid"]
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    new = pd.concat([old[old["fold"] != fold], pd.DataFrame([{"fold": fold, "clip_limit": clip_limit, "tile_grid": tile_grid}])])
    new = new.sort_values("fold")[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def tune_fold(cfg: dict, fold: str, results: Path, device: str | None = None, amp: bool = True) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tables = results / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    final, partial = tables / f"tuning_{fold}.csv", tables / f"tuning_{fold}.partial.csv"
    grid = [(float(c), tuple(t)) for c in cfg["clahe_grid"]["clip_limit"] for t in cfg["clahe_grid"]["tile_grid"]]
    done = pd.read_csv(partial) if partial.exists() else pd.DataFrame(columns=["clip_limit", "tile_grid", "val_dice"])
    seen = {(float(r.clip_limit), r.tile_grid) for r in done.itertuples()}
    tune = cfg["tuning"]
    for clip, tile in grid:
        if (clip, grid_label(tile)) in seen:
            continue
        seed_everything(tune["seed"])
        train_ds, val_ds = make_train_val(cfg, fold, "S3", tune["seed"], clahe=ClaheParams(clip, tile))
        res = fit(build_model(cfg), train_ds, val_ds, cfg, tune["seed"], tune["epochs"], None, device, amp, verbose=False)
        done = pd.concat([done, pd.DataFrame([{"clip_limit": clip, "tile_grid": grid_label(tile), "val_dice": res["best_dice"]}])], ignore_index=True)
        done.to_csv(partial, index=False)  # safe to resume after an interruption
        print(f"{fold}  clip {clip:.0f}  tile {grid_label(tile):>5}  val dice {res['best_dice']:.4f}", flush=True)
    # keep grid order, independent of the order combinations were finished
    order = {(c, grid_label(t)): i for i, (c, t) in enumerate(grid)}
    done = done.assign(_o=[order[(float(r.clip_limit), r.tile_grid)] for r in done.itertuples()]).sort_values("_o").drop(columns="_o")
    done.to_csv(final, index=False)
    partial.unlink(missing_ok=True)
    win = pick_winner(done)
    update_s3_params(tables / "s3_params.csv", fold, float(win["clip_limit"]), win["tile_grid"])
    print(f"{fold}: winner clip {win['clip_limit']:.0f} tile {win['tile_grid']} (val dice {win['val_dice']:.4f})")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    results = ROOT / a.results
    if (results / "tables" / f"tuning_{a.fold}.csv").exists():
        print(f"skip {a.fold}: tuning_{a.fold}.csv exists")
        return
    tune_fold(cfg, a.fold, results, amp=not a.no_amp)


if __name__ == "__main__":
    main()
