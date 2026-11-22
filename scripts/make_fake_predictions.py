"""Write synthetic pred.npz files that follow the real contract (docs/PROTOCOL.md section 6).

Purpose: lets Nabil build and test evaluate / shift / stats / figures before Huda's real runs exist.
The fake model is the ground-truth mask blurred and randomly shifted, harder on foreign images.
NEVER put these numbers in a report. Output goes to results/fake_predictions (gitignored).

Usage: python scripts/make_fake_predictions.py [--seeds 11 22]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.data import fold_split, load_arrays, load_manifest  # noqa: E402

SEVERITY = {"val": 0.6, "in_source": 0.6, "foreign": 2.0}  # pixels of shift and noise scale


def fake_prob(mask: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    dx, dy = rng.normal(0, severity * 2, 2)
    m = cv2.warpAffine((mask > 0).astype(np.float32), np.float32([[1, 0, dx], [0, 1, dy]]), mask.shape[::-1])
    p = cv2.GaussianBlur(m, (0, 0), 1.5 + severity)
    p = np.clip(p + rng.normal(0, 0.05 * severity, p.shape), 0, 1)
    return np.rint(p * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--out", default="results/fake_predictions")
    args = ap.parse_args()
    cfg = load_config()
    manifest = load_manifest(ROOT / cfg["data"]["processed_dir"])
    for fold in cfg["folds"]:
        parts = fold_split(manifest, fold)
        for scenario in cfg["scenarios"]:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                ids, groups, probs = [], [], []
                for group, key in (("val", "val"), ("in_source", "in_source_test"), ("foreign", "foreign")):
                    _, masks = load_arrays(parts[key], ROOT / cfg["data"]["processed_dir"])
                    ids += parts[key]["id"].tolist()
                    groups += [group] * len(masks)
                    probs += [fake_prob(m, SEVERITY[group], rng) for m in masks]
                dest = ROOT / args.out / fold / scenario / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest / "pred.npz", ids=np.array(ids), group=np.array(groups), prob_u8=np.stack(probs))
        print(f"{fold}: done")


if __name__ == "__main__":
    main()
