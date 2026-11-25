import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qc_data import compute_dhash, find_cross_source_duplicates, run_qc  # noqa: E402

from cxrseg.config import load_config  # noqa: E402


def _make_qc_fixture(tmp_path):
    proc = tmp_path / "processed"
    rows = []
    rng = np.random.default_rng(7)

    # Create a gradient image shared across darwin and montgomery (exact cross-source duplicate)
    dup_img = np.tile(np.linspace(20, 230, 256, dtype=np.uint8), (256, 1))
    dup_img = cv2.GaussianBlur(dup_img, (5, 5), 1.0)

    for src in ("darwin", "montgomery", "shenzhen"):
        (proc / src / "images").mkdir(parents=True, exist_ok=True)
        (proc / src / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(4):
            if i == 0 and src in ("darwin", "montgomery"):
                img = dup_img.copy()
            else:
                # Distinct random smooth pattern per image so dHashes differ
                low = rng.integers(20, 230, size=(8, 8), dtype=np.uint8)
                img = cv2.resize(low, (256, 256), interpolation=cv2.INTER_CUBIC)
            msk = np.zeros((256, 256), dtype=np.uint8)
            msk[64:192, 64:192] = 255
            name = f"s_{i}"
            img_rel = f"{src}/images/{name}.png"
            msk_rel = f"{src}/masks/{name}.png"
            cv2.imwrite(str(proc / img_rel), img)
            cv2.imwrite(str(proc / msk_rel), msk)
            rows.append(
                {
                    "id": f"{src}/{name}",
                    "source": src,
                    "name": name,
                    "image_path": img_rel,
                    "mask_path": msk_rel,
                    "orig_h": 512,
                    "orig_w": 512,
                    "mask_area_frac": 0.25,
                    "split": "train",
                }
            )
    pd.DataFrame(rows).to_csv(proc / "manifest.csv", index=False)
    return proc


def test_dhash_and_cross_source_duplicate_detection(tmp_path):
    proc = _make_qc_fixture(tmp_path)
    out = tmp_path / "qc_out"
    cfg = load_config()

    summary, dups, _ = run_qc(proc, out, cfg, n_samples=4, seed=42, max_hamming=4)

    for src in cfg["sources"]:
        assert (out / "figures" / f"qc_{src}.png").exists()
    assert (out / "tables" / "qc_summary.csv").exists()
    assert (out / "tables" / "qc_duplicates.csv").exists()

    assert len(summary) == 3
    assert (summary["non_binary_masks"] == 0).all()

    # Cross-source duplicate between darwin/s_0 and montgomery/s_0 must be detected
    pairs = set(zip(dups["id_a"], dups["id_b"]))
    assert ("darwin/s_0", "montgomery/s_0") in pairs
    assert dups.loc[dups["id_a"] == "darwin/s_0", "hamming"].iloc[0] == 0


def test_dhash_ignores_same_source_pairs():
    img = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    h = compute_dhash(img)
    df = find_cross_source_duplicates(
        ["darwin/a", "darwin/b"],
        ["darwin", "darwin"],
        np.stack([h, h]),
        max_hamming=4,
    )
    assert df.empty
