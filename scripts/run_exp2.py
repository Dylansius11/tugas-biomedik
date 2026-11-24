"""Percobaan 2 end to end (docs/PLAN-PERCOBAAN-2.md). Resumable: finished pieces are skipped.

1. train arms E1 and E2 (3 folds x 3 seeds each)
2. pick the final arm on in-source validation Dice only (tie rule in configs/default.yaml exp2.pick_tie)
3. few-shot adaptation of the final arm for every fold, seed and k
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import load_config  # noqa: E402
from cxrseg.exp2 import RESULTS, fewshot, train_arm  # noqa: E402


def val_mean(cfg, arm) -> float:
    return float(np.mean([float((RESULTS / arm / f / f"seed{s}" / "val_dice.txt").read_text()) for f in cfg["folds"] for s in cfg["train"]["seeds"]]))


def main() -> None:
    cfg = load_config()
    e = cfg["exp2"]
    for arm in e["arms"]:
        for fold in cfg["folds"]:
            for seed in cfg["train"]["seeds"]:
                train_arm(cfg, arm, fold, seed)
    v = {arm: val_mean(cfg, arm) for arm in e["arms"]}
    final = "E2" if v["E2"] >= v["E1"] - e["pick_tie"] else "E1"
    (RESULTS / "final_arm.txt").write_text(f"{final}\n" + "".join(f"{a} mean val dice {d:.5f}\n" for a, d in v.items()))
    print(f"validation: {v} -> final arm {final}", flush=True)
    for fold in cfg["folds"]:
        for k in e["fewshot"]["k"]:
            for seed in cfg["train"]["seeds"]:
                fewshot(cfg, final, fold, seed, k)
                print(f"fewshot {final}/{fold}/k{k}/seed{seed} done", flush=True)
    print("done")


if __name__ == "__main__":
    main()
