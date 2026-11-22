"""Image and mask preprocessing. Order is fixed by docs/PROTOCOL.md section 4:
grayscale uint8 -> resize -> contrast scenario -> divide by 255."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

SCENARIOS = ("S0", "S1", "S2", "S3")


@dataclass(frozen=True)
class ClaheParams:
    clip_limit: float
    tile_grid: tuple[int, int]


DEFAULT_CLAHE = ClaheParams(clip_limit=2.0, tile_grid=(8, 8))


def to_gray_uint8(arr: np.ndarray) -> np.ndarray:
    """HxW, HxWx3 or HxWx4 array (uint8 or uint16) to HxW uint8."""
    if arr.ndim == 3:
        arr = arr[..., :3]
        if arr.dtype == np.uint8:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr.mean(axis=2)
    if arr.dtype == np.uint16:
        arr = arr / 257.0
    elif arr.dtype != np.uint8:
        raise ValueError(f"unsupported dtype {arr.dtype}")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def resize_image(gray: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Masks in the collection carry ~0.1% anti-aliasing noise (values 1..254). Threshold at 127."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(binarize_mask(mask), (size, size), interpolation=cv2.INTER_NEAREST)


@lru_cache(maxsize=64)
def _clahe(clip_limit: float, tile_grid: tuple[int, int]) -> "cv2.CLAHE":
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))


def enhance(img: np.ndarray, scenario: str, clahe: ClaheParams | None = None) -> np.ndarray:
    """Apply the contrast scenario to a uint8 image. S3 needs the tuned parameters."""
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("enhance expects a 2-D uint8 image")
    if scenario == "S0":
        return img
    if scenario == "S1":
        return cv2.equalizeHist(img)
    if scenario == "S2":
        p = DEFAULT_CLAHE
        return _clahe(p.clip_limit, p.tile_grid).apply(img)
    if scenario == "S3":
        if clahe is None:
            raise ValueError("S3 requires tuned ClaheParams (see results/tables/s3_params.csv)")
        return _clahe(clahe.clip_limit, tuple(clahe.tile_grid)).apply(img)
    raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")


def to_unit_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def preprocess(img_u8: np.ndarray, scenario: str, clahe: ClaheParams | None = None) -> np.ndarray:
    """uint8 256x256 -> float32 in [0, 1] after the contrast scenario."""
    return to_unit_float(enhance(img_u8, scenario, clahe))
