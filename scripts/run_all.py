"""Run all 36 full trainings (4 scenarios x 3 folds x 3 seeds) and write results/env.txt.

Safe to re-run after an interruption (local or Colab): finished runs are skipped, unfinished ones resume from last.pt.
S3 needs results/tables/s3_params.csv, so it is skipped for a fold until tuning has produced that fold's row.

Usage: python scripts/run_all.py [--scenarios S0 S1] [--folds ho_darwin] [--seeds 11] [--results results]
"""
from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.dataset import load_s3_params  # noqa: E402
from cxrseg.train import train_run  # noqa: E402


def write_env(results: Path) -> None:
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}", f"torch {torch.__version__}", f"cuda {torch.version.cuda}"]
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        lines.append(f"gpu {p.name} ({p.total_memory / 2**30:.1f} GiB)")
    else:
        lines.append("gpu none")
    for pkg in ("numpy", "scipy", "pandas", "opencv-python-headless", "pyyaml", "matplotlib", "scikit-learn", "tqdm"):
        try:
            lines.append(f"{pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{pkg} not installed")
    results.mkdir(parents=True, exist_ok=True)
    (results / "env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=cfg["scenarios"])
    ap.add_argument("--folds", nargs="+", default=cfg["folds"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["train"]["seeds"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()
    results = ROOT / a.results
    write_env(results)
    skipped_s3 = []
    for scenario in a.scenarios:
        for fold in a.folds:
            if scenario == "S3":
                try:
                    load_s3_params(fold, results / "tables" / "s3_params.csv")
                except (FileNotFoundError, ValueError):
                    skipped_s3.append(fold)
                    continue
            for seed in a.seeds:
                train_run(cfg, fold, scenario, seed, results, amp=not a.no_amp)
    if skipped_s3:
        print(f"S3 skipped for {skipped_s3}: run python -m cxrseg.tune_clahe --fold <fold> first, then re-run this script")
    print("done. Check progress with: python scripts/check_runs.py")


if __name__ == "__main__":
    main()
