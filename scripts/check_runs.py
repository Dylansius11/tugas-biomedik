"""Show which of the 36 full training runs (and tuning files) exist.

Usage: python scripts/check_runs.py [--pred-dir results/predictions]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="results/predictions")
    args = ap.parse_args()
    cfg = load_config()
    pred = ROOT / args.pred_dir
    total = done = 0
    print(f"{'fold':<15}{'scenario':<10}" + "".join(f"seed{s:<5}" for s in cfg["train"]["seeds"]))
    for fold in cfg["folds"]:
        for scenario in cfg["scenarios"]:
            cells = []
            for seed in cfg["train"]["seeds"]:
                ok = (pred / fold / scenario / f"seed{seed}" / "pred.npz").exists()
                total += 1
                done += ok
                cells.append(f"{'ok' if ok else '--':<9}")
            print(f"{fold:<15}{scenario:<10}" + "".join(cells))
    print(f"\npredictions: {done}/{total}")
    tables = ROOT / "results" / "tables"
    for fold in cfg["folds"]:
        f = tables / f"tuning_{fold}.csv"
        print(f"tuning_{fold}.csv: {'ok' if f.exists() else 'missing'}")
    print(f"s3_params.csv: {'ok' if (tables / 's3_params.csv').exists() else 'missing'}")
    print(f"env.txt: {'ok' if (ROOT / 'results' / 'env.txt').exists() else 'missing'}")


if __name__ == "__main__":
    main()
