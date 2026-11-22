"""Manifest and fold logic shared by training, evaluation and analysis."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SOURCES = ("darwin", "montgomery", "shenzhen")
FOLDS = tuple(f"ho_{s}" for s in SOURCES)


def held_out_source(fold: str) -> str:
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}, expected one of {FOLDS}")
    return fold.removeprefix("ho_")


def load_manifest(processed_dir: str | Path = "data/processed") -> pd.DataFrame:
    return pd.read_csv(Path(processed_dir) / "manifest.csv")


def fold_split(manifest: pd.DataFrame, fold: str) -> dict[str, pd.DataFrame]:
    """Rows for one leave-one-source-out fold.

    train / val / in_source_test come from the two training sources.
    foreign is the whole held-out source and must never be used to fit or tune anything.
    """
    ho = held_out_source(fold)
    seen = manifest[manifest["source"] != ho]
    return {
        "train": seen[seen["split"] == "train"],
        "val": seen[seen["split"] == "val"],
        "in_source_test": seen[seen["split"] == "test"],
        "foreign": manifest[manifest["source"] == ho],
    }


def load_arrays(rows: pd.DataFrame, processed_dir: str | Path = "data/processed") -> tuple[np.ndarray, np.ndarray]:
    """Return images (N,256,256) uint8 and masks (N,256,256) uint8 (0 or 255) for manifest rows."""
    root = Path(processed_dir)
    images, masks = [], []
    for r in rows.itertuples():
        img = cv2.imread(str(root / r.image_path), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(f"missing files for {r.id}")
        images.append(img)
        masks.append(msk)
    return np.stack(images), np.stack(masks)
