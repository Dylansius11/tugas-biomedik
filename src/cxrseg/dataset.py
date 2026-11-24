"""Datasets for one LOSO fold (docs/PROTOCOL.md sections 3 and 4).

The contrast scenario is applied once per image and kept in memory. Augmentation is geometric only,
applied to image and mask together. Only the 'foreign' group is the held-out source: load it
only when writing predictions, never for training, validation or tuning.
"""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ROOT
from .data import fold_split, load_arrays
from .preprocess import ClaheParams, preprocess

S3_PARAMS_PATH = ROOT / "results" / "tables" / "s3_params.csv"


def load_s3_params(fold: str, path: str | Path = S3_PARAMS_PATH) -> ClaheParams:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run tune_clahe for all folds before using scenario S3")
    df = pd.read_csv(path)
    row = df[df["fold"] == fold]
    if row.empty:
        raise ValueError(f"fold {fold!r} missing in {path}: run tune_clahe --fold {fold}")
    r = row.iloc[0]
    grid = r["tile_grid"]
    if isinstance(grid, str):
        grid = ast.literal_eval(grid) if grid.strip().startswith(("(", "[")) else tuple(int(x) for x in grid.lower().split("x"))
    return ClaheParams(float(r["clip_limit"]), tuple(int(g) for g in grid))


def scenario_params(fold: str, scenario: str, s3_path: str | Path = S3_PARAMS_PATH) -> ClaheParams | None:
    return load_s3_params(fold, s3_path) if scenario == "S3" else None


def augment_pair(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator, aug: dict) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation, scale and translation (no flip, no intensity change). Border filled with 0."""
    h, w = img.shape
    angle = rng.uniform(-aug["rotation_deg"], aug["rotation_deg"])
    scale = rng.uniform(*aug["scale"])
    tx, ty = rng.uniform(-aug["translate"], aug["translate"], 2) * (w, h)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[:, 2] += (tx, ty)
    img2 = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    msk2 = cv2.warpAffine(msk, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, msk2


class SegDataset(Dataset):
    """images: float32 (N,H,W) in [0,1]; masks: uint8 (N,H,W) with 0/1. Augments only if aug is given."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, aug: dict | None = None, seed: int = 0):
        assert len(images) == len(masks)
        self.images, self.masks, self.aug, self.seed, self.epoch = images, masks, aug, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img, msk = self.images[i], self.masks[i]
        if self.aug is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])  # same result regardless of worker
            img, msk = augment_pair(img, msk, rng, self.aug)
        return torch.from_numpy(np.ascontiguousarray(img))[None], torch.from_numpy(np.ascontiguousarray(msk).astype(np.float32))[None]


def load_group(cfg: dict, fold: str, scenario: str, group: str, s3_path: str | Path = S3_PARAMS_PATH, clahe: ClaheParams | None = None):
    """Preprocessed arrays for one group: train, val, in_source_test or foreign.

    `clahe` overrides the S3 parameters (used by tuning, which tries a grid before s3_params.csv exists).
    Returns (ids, images float32 (N,H,W), masks uint8 0/1 (N,H,W)).
    """
    from .data import load_manifest

    processed = ROOT / cfg["data"]["processed_dir"]
    rows = fold_split(load_manifest(processed), fold)[group]
    if clahe is None:
        clahe = scenario_params(fold, scenario, s3_path)
    imgs, msks = load_arrays(rows, processed)
    x = np.stack([preprocess(i, scenario, clahe) for i in imgs])
    return rows["id"].tolist(), x, (msks > 0).astype(np.uint8)


def make_train_val(cfg: dict, fold: str, scenario: str, seed: int, s3_path: str | Path = S3_PARAMS_PATH, clahe: ClaheParams | None = None):
    """Training (augmented) and validation datasets. Never touches the held-out source."""
    _, xt, yt = load_group(cfg, fold, scenario, "train", s3_path, clahe)
    _, xv, yv = load_group(cfg, fold, scenario, "val", s3_path, clahe)
    return SegDataset(xt, yt, cfg["train"]["augmentation"], seed), SegDataset(xv, yv)
