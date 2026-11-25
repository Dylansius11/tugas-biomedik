"""Data quality control (QC): visual overlays, summary statistics, and cross-source duplicate detection.

Follows docs/plans/BUILD-PLAN-NABIL.md (N1) and docs/PROTOCOL.md section 2.
Outputs:
  - {out}/figures/qc_{source}.png
  - {out}/tables/qc_summary.csv
  - {out}/tables/qc_duplicates.csv

Usage:
    python scripts/qc_data.py [--processed-dir data/processed] [--out results]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.data import load_manifest  # noqa: E402


def compute_dhash(img_u8: np.ndarray) -> np.ndarray:
    """Compute 64-bit difference hash (8x8): resize to 9x8 and compare horizontal neighbors."""
    small = cv2.resize(img_u8, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).reshape(-1)


def find_cross_source_duplicates(
    ids: list[str],
    sources: list[str],
    hashes: np.ndarray,
    max_hamming: int = 4,
) -> pd.DataFrame:
    """Find near-duplicate pairs across different sources with Hamming distance <= max_hamming."""
    cols = ["id_a", "id_b", "hamming"]
    if len(ids) < 2:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    n = len(ids)
    # Vectorized row-vs-remaining comparison for speed and deterministic ordering
    for i in range(n - 1):
        diff_source = np.array([sources[j] != sources[i] for j in range(i + 1, n)], dtype=bool)
        if not diff_source.any():
            continue
        dists = np.count_nonzero(hashes[i + 1 :] != hashes[i], axis=1)
        hits = np.where(diff_source & (dists <= max_hamming))[0]
        for offset in hits:
            j = i + 1 + int(offset)
            rows.append(
                {
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "hamming": int(dists[offset]),
                }
            )
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df = df.sort_values(["hamming", "id_a", "id_b"]).reset_index(drop=True)
    return df


def save_qc_figure(
    source: str,
    rows: pd.DataFrame,
    processed_dir: Path,
    out_path: Path,
    *,
    n_samples: int = 20,
    seed: int = 42,
) -> None:
    """Save a grid of random images with ground-truth mask contour overlays."""
    rng = np.random.default_rng(seed)
    k = min(n_samples, len(rows))
    chosen_idx = np.sort(rng.choice(len(rows), size=k, replace=False))
    sample = rows.iloc[chosen_idx]

    ncols = 5
    nrows = max(1, int(np.ceil(k / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.1))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat:
        ax.axis("off")

    for ax, r in zip(axes_flat, sample.itertuples()):
        img = cv2.imread(str(processed_dir / r.image_path), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(processed_dir / r.mask_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.contour(msk > 0, levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(f"{r.name}\nluas={r.mask_area_frac:.2f}", fontsize=8)

    fig.suptitle(f"QC Visual: {source.capitalize()} ({k} sampel acak, seed={seed}, hijau=tepi mask)", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_qc(
    processed_dir: Path,
    out_dir: Path,
    cfg: dict,
    *,
    n_samples: int = 20,
    seed: int = 42,
    max_hamming: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run full QC: summary stats, visual figures, cross-source dHash duplicates, and polarity warnings."""
    manifest = load_manifest(processed_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    all_ids: list[str] = []
    all_sources: list[str] = []
    all_hashes: list[np.ndarray] = []
    suspects: list[str] = []

    for source in cfg["sources"]:
        sub = manifest[manifest["source"] == source].reset_index(drop=True)
        if sub.empty:
            continue

        save_qc_figure(
            source,
            sub,
            processed_dir,
            figures_dir / f"qc_{source}.png",
            n_samples=n_samples,
            seed=seed,
        )

        px_sum = 0.0
        px_sq_sum = 0.0
        px_count = 0
        non_binary = 0
        mask_fracs: list[float] = []

        for r in sub.itertuples():
            img = cv2.imread(str(processed_dir / r.image_path), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(processed_dir / r.mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                raise FileNotFoundError(f"Missing processed image or mask for {r.id}")

            uniq = set(np.unique(msk).tolist())
            if not uniq.issubset({0, 255}):
                non_binary += 1
                suspects.append(f"{r.id}: non-binary mask values {sorted(uniq)[:6]}")

            img_f = img.astype(np.float64)
            px_sum += float(img_f.sum())
            px_sq_sum += float((img_f * img_f).sum())
            px_count += int(img_f.size)
            mask_fracs.append(float((msk > 0).mean()))

            # Heuristic flag: inverted polarity (lung area darker than corners/border or very bright corners)
            # Check border mean vs lung mask mean — normally X-ray background is darker or bone is brighter;
            # flag extreme low/high mean intensity or extreme mask fraction
            if img_f.mean() < 35 or img_f.mean() > 220:
                suspects.append(f"{r.id}: extreme mean intensity ({img_f.mean():.1f})")
            if mask_fracs[-1] < 0.05 or mask_fracs[-1] > 0.65:
                suspects.append(f"{r.id}: unusual mask_area_frac ({mask_fracs[-1]:.3f})")

            all_ids.append(str(r.id))
            all_sources.append(str(source))
            all_hashes.append(compute_dhash(img))

        mean_int = px_sum / max(px_count, 1)
        var_int = max(0.0, px_sq_sum / max(px_count, 1) - mean_int * mean_int)
        std_int = float(np.sqrt(var_int))

        row = {
            "source": source,
            "n_images": int(len(sub)),
            "orig_h_min": int(sub["orig_h"].min()) if "orig_h" in sub.columns else np.nan,
            "orig_h_max": int(sub["orig_h"].max()) if "orig_h" in sub.columns else np.nan,
            "orig_w_min": int(sub["orig_w"].min()) if "orig_w" in sub.columns else np.nan,
            "orig_w_max": int(sub["orig_w"].max()) if "orig_w" in sub.columns else np.nan,
            "mask_area_frac_mean": float(np.mean(mask_fracs)),
            "mask_area_frac_min": float(np.min(mask_fracs)),
            "mask_area_frac_max": float(np.max(mask_fracs)),
            "intensity_mean": float(mean_int),
            "intensity_std": std_int,
            "non_binary_masks": int(non_binary),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "qc_summary.csv", index=False)

    hashes_arr = np.stack(all_hashes) if all_hashes else np.empty((0, 64), dtype=bool)
    dup_df = find_cross_source_duplicates(all_ids, all_sources, hashes_arr, max_hamming=max_hamming)
    dup_df.to_csv(tables_dir / "qc_duplicates.csv", index=False)

    return summary_df, dup_df, suspects


def main() -> None:
    ap = argparse.ArgumentParser(description="Run data quality control (QC) on processed CXR images.")
    ap.add_argument("--processed-dir", default="data/processed", help="Processed data directory")
    ap.add_argument("--out", default="results", help="Output directory (writes figures/ and tables/)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for QC sample grids")
    ap.add_argument("--n-samples", type=int, default=20, help="Samples per source in qc_{source}.png")
    ap.add_argument("--max-hamming", type=int, default=4, help="Maximum dHash Hamming distance for duplicate flag")
    ap.add_argument("--config", default=None, help="Override config YAML path")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    proc_dir = Path(args.processed_dir)
    if not proc_dir.is_absolute():
        proc_dir = ROOT / proc_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    summary_df, dup_df, suspects = run_qc(
        proc_dir,
        out_dir,
        cfg,
        n_samples=args.n_samples,
        seed=args.seed,
        max_hamming=args.max_hamming,
    )
    print("== QC Summary ==")
    print(summary_df.to_string(index=False))
    print(f"\nCross-source duplicate pairs (hamming <= {args.max_hamming}): {len(dup_df)}")
    if not dup_df.empty:
        print(dup_df.to_string(index=False))
    if suspects:
        print(f"\nHeuristic warnings ({len(suspects)}):")
        for s in suspects[:30]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
