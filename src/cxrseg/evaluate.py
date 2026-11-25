"""Evaluate saved model probability maps against ground truth masks (docs/PROTOCOL.md sections 6 & 7).

Usage:
    python -m cxrseg.evaluate --pred-dir results/fake_predictions --out results/fake_metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data import load_manifest
from .metrics import evaluate_pair
from .postprocess import postprocess_mask

COLUMNS = [
    "fold",
    "scenario",
    "seed",
    "id",
    "source",
    "group",
    "dice",
    "iou",
    "hd95",
    "coverage",
    "area_error_pct",
    "failed",
]


def _parse_run_metadata(
    pred_npz_path: Path,
    fold: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> tuple[str, str, int]:
    """Extract (fold, scenario, seed) from `{pred_dir}/{fold}/{scenario}/seed{seed}/pred.npz`."""
    seed_dir = pred_npz_path.parent
    scenario_dir = seed_dir.parent
    fold_dir = scenario_dir.parent
    if seed is None:
        name = seed_dir.name
        if not name.startswith("seed"):
            raise ValueError(f"Cannot parse seed from path {pred_npz_path}; expected .../seed<N>/pred.npz")
        seed = int(name.removeprefix("seed"))
    if scenario is None:
        scenario = scenario_dir.name
    if fold is None:
        fold = fold_dir.name
    return fold, scenario, int(seed)


def evaluate_run(
    pred_npz_path: str | Path,
    manifest: pd.DataFrame,
    processed_dir: str | Path,
    cfg: dict,
    *,
    fold: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
    gt_cache: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Evaluate one `pred.npz` run against ground-truth masks.

    Applies post-processing (`postprocess_mask`) before computing metrics (`metrics.evaluate_pair`).
    """
    pred_path = Path(pred_npz_path)
    processed_root = Path(processed_dir)
    run_fold, run_scenario, run_seed = _parse_run_metadata(pred_path, fold, scenario, seed)

    pp = cfg["postprocess"]
    mcfg = cfg["metrics"]
    failure_dice = float(mcfg["failure_dice"])
    hd95_empty = float(mcfg["hd95_empty_value"])

    man_idx = manifest.set_index("id", drop=False)
    rows: list[dict] = []

    with np.load(pred_path, allow_pickle=True) as data:
        ids = data["ids"]
        groups = data["group"]
        probs_u8 = data["prob_u8"]

        for img_id_raw, group_raw, prob_u8 in zip(ids, groups, probs_u8):
            img_id = str(img_id_raw)
            group = str(group_raw)
            if img_id not in man_idx.index:
                raise KeyError(f"Image id {img_id!r} not found in manifest")
            mrow = man_idx.loc[img_id]
            if gt_cache is not None and img_id in gt_cache:
                gt_mask = gt_cache[img_id]
            else:
                mask_file = processed_root / str(mrow["mask_path"])
                gt_u8 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if gt_u8 is None:
                    raise FileNotFoundError(f"Missing ground-truth mask: {mask_file}")
                gt_mask = gt_u8 > 0
                if gt_cache is not None:
                    gt_cache[img_id] = gt_mask

            pred_mask = postprocess_mask(
                prob_u8,
                threshold=float(pp["threshold"]),
                keep_components=int(pp["keep_components"]),
                closing_kernel=int(pp["closing_kernel"]),
            )
            m = evaluate_pair(
                pred_mask,
                gt_mask,
                failure_dice=failure_dice,
                hd95_empty=hd95_empty,
            )
            source = str(mrow["source"]) if "source" in mrow else img_id.split("/", 1)[0]
            rows.append(
                {
                    "fold": run_fold,
                    "scenario": run_scenario,
                    "seed": run_seed,
                    "id": img_id,
                    "source": source,
                    "group": group,
                    **m,
                }
            )

    return pd.DataFrame(rows, columns=COLUMNS)


def evaluate_all(
    pred_dir: str | Path,
    out_dir: str | Path,
    cfg: dict,
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate all runs under `pred_dir` and write `{out_dir}/{fold}/{scenario}/seed{seed}.csv`."""
    pred_root = Path(pred_dir)
    out_root = Path(out_dir)
    proc_root = Path(processed_dir) if processed_dir is not None else ROOT / cfg["data"]["processed_dir"]
    manifest = load_manifest(proc_root)
    gt_cache: dict[str, np.ndarray] = {}

    frames: list[pd.DataFrame] = []
    for fold in cfg["folds"]:
        for scenario in cfg["scenarios"]:
            scen_dir = pred_root / fold / scenario
            if not scen_dir.exists():
                continue
            seed_dirs = sorted(
                (p for p in scen_dir.iterdir() if p.is_dir() and p.name.startswith("seed")),
                key=lambda p: int(p.name.removeprefix("seed")),
            )
            for sdir in seed_dirs:
                pred_npz = sdir / "pred.npz"
                if not pred_npz.exists():
                    continue
                seed = int(sdir.name.removeprefix("seed"))
                df = evaluate_run(
                    pred_npz,
                    manifest,
                    proc_root,
                    cfg,
                    fold=fold,
                    scenario=scenario,
                    seed=seed,
                    gt_cache=gt_cache,
                )
                dest = out_root / fold / scenario / f"seed{seed}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(dest, index=False)
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    ap.add_argument("--pred-dir", default="results/predictions", help="Directory containing {fold}/{scenario}/seed{seed}/pred.npz")
    ap.add_argument("--out", default="results/metrics", help="Output directory for CSV metrics")
    ap.add_argument("--processed-dir", default=None, help="Override processed data directory")
    ap.add_argument("--config", default=None, help="Override config YAML path")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    pred_root = Path(args.pred_dir)
    if not pred_root.is_absolute():
        pred_root = ROOT / pred_root
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    proc_root = Path(args.processed_dir) if args.processed_dir else ROOT / cfg["data"]["processed_dir"]
    if not proc_root.is_absolute():
        proc_root = ROOT / proc_root

    df = evaluate_all(pred_root, out_root, cfg, proc_root)
    n_runs = df[["fold", "scenario", "seed"]].drop_duplicates().shape[0] if not df.empty else 0
    print(f"Evaluated {n_runs} run(s), {len(df)} image predictions -> {out_root}")


if __name__ == "__main__":
    main()
