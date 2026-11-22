"""Build data/processed from the raw Mendeley zips.

- reads images and masks straight from the zips (no extraction needed)
- Darwin: random sample of 600 with seed 42; Montgomery and Shenzhen: everything
- grayscale uint8, resize to 256x256 (INTER_AREA for images, nearest for binarized masks)
- 80/10/10 split inside every source (seed 42)
- writes data/processed/{source}/images|masks/*.png and data/processed/manifest.csv

Usage: python scripts/prepare_data.py [--zip]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.preprocess import binarize_mask, resize_image, resize_mask, to_gray_uint8  # noqa: E402

SOURCE_ZIP = {"darwin": "Darwin.zip", "montgomery": "Montgomery.zip", "shenzhen": "Shenzhen.zip"}


def list_pairs(z: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    """Map image stem -> (image member, mask member) using the /img/ and /mask/ folders."""
    imgs = {Path(n).name: n for n in z.namelist() if "/img/" in n and not n.endswith("/")}
    masks = {Path(n).name: n for n in z.namelist() if "/mask/" in n and not n.endswith("/")}
    common = sorted(set(imgs) & set(masks))
    missing = (set(imgs) ^ set(masks))
    if missing:
        print(f"  WARNING: {len(missing)} files without a partner image or mask, skipped", file=sys.stderr)
    return {Path(n).stem: (imgs[n], masks[n]) for n in common}


def decode(z: zipfile.ZipFile, member: str) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"cannot decode {member}")
    return arr


def assign_splits(n: int, fractions: dict, rng: np.random.Generator) -> list[str]:
    order = rng.permutation(n)
    n_val, n_test = int(round(n * fractions["val"])), int(round(n * fractions["test"]))
    labels = np.array(["train"] * n, dtype=object)
    labels[order[:n_val]] = "val"
    labels[order[n_val : n_val + n_test]] = "test"
    return list(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also write data/processed.zip for sharing (Colab, teammates)")
    ap.add_argument("--sources", nargs="+", choices=["darwin", "montgomery", "shenzhen"], help="partial dry run on some sources (manifest will be incomplete)")
    args = ap.parse_args()

    cfg = load_config()
    raw, out = ROOT / cfg["data"]["raw_dir"], ROOT / cfg["data"]["processed_dir"]
    size = cfg["data"]["image_size"]
    rows = []
    for source in (args.sources or cfg["sources"]):
        z = zipfile.ZipFile(raw / SOURCE_ZIP[source])
        pairs = list_pairs(z)
        stems = sorted(pairs)
        if source == "darwin":  # fixed random order; take the first N usable pairs
            rng = np.random.default_rng(cfg["data"]["sample_seed"])
            stems = rng.permutation(stems).tolist()
        target = cfg["data"]["darwin_sample"] if source == "darwin" else len(stems)
        kept, skipped, resized_masks = [], 0, 0
        for stem in stems:
            if len(kept) == target:
                break
            im_member, mk_member = pairs[stem]
            img, msk = decode(z, im_member), decode(z, mk_member)
            gray = to_gray_uint8(img)
            if msk.shape[:2] != gray.shape[:2]:
                msk = cv2.resize(binarize_mask(msk), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                resized_masks += 1
            img256, msk256 = resize_image(gray, size), resize_mask(msk, size)
            frac = float((msk256 > 0).mean())
            if frac < cfg["data"]["min_mask_frac"]:
                skipped += 1
                continue
            kept.append((stem, img256, msk256, gray.shape, frac))
        print(f"{source}: {len(pairs)} pairs in zip, using {len(kept)}, skipped {skipped} with empty or tiny masks")
        kept.sort(key=lambda k: k[0])
        split_rng = np.random.default_rng(cfg["data"]["split_seed"])
        splits = assign_splits(len(kept), cfg["data"]["split"], split_rng)
        for sub in ("images", "masks"):
            shutil.rmtree(out / source / sub, ignore_errors=True)  # drop files from earlier runs
            (out / source / sub).mkdir(parents=True, exist_ok=True)
        for (stem, img256, msk256, shape, frac), split in zip(kept, splits):
            cv2.imwrite(str(out / source / "images" / f"{stem}.png"), img256)
            cv2.imwrite(str(out / source / "masks" / f"{stem}.png"), msk256)
            rows.append(
                {
                    "id": f"{source}/{stem}",
                    "source": source,
                    "name": stem,
                    "image_path": f"{source}/images/{stem}.png",
                    "mask_path": f"{source}/masks/{stem}.png",
                    "orig_h": shape[0],
                    "orig_w": shape[1],
                    "mask_area_frac": round(frac, 5),
                    "split": split,
                }
            )
        if resized_masks:
            print(f"  note: {resized_masks} masks had a different size than their image and were resized")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    print("\nsplit sizes per source")
    print(manifest.pivot_table(index="source", columns="split", values="id", aggfunc="count"))
    print("\noriginal size range per source (h x w)")
    print(manifest.groupby("source")[["orig_h", "orig_w"]].agg(["min", "median", "max"]))
    print("\nmask area fraction per source")
    print(manifest.groupby("source")["mask_area_frac"].describe()[["min", "50%", "max"]])
    if args.zip:
        archive = shutil.make_archive(str(out.parent / "processed"), "zip", out)
        print(f"\nwrote {archive}")


if __name__ == "__main__":
    main()
