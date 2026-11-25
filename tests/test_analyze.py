import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze import run_analysis  # noqa: E402

from cxrseg.config import load_config  # noqa: E402
from cxrseg.data import fold_split  # noqa: E402


def _make_full_synthetic_workspace(tmp_path):
    cfg = load_config()
    cfg["stats"]["bootstrap"] = 200
    cfg["stats"]["sample_size_check"] = 10
    cfg["stats"]["sample_repeats"] = 50
    cfg["train"]["seeds"] = [11, 22]

    proc = tmp_path / "processed"
    rng = np.random.default_rng(42)
    rows = []

    # 12 images per source: 8 train, 2 val, 2 test (all 36 images)
    for src, base_int in (("darwin", 150), ("montgomery", 80), ("shenzhen", 95)):
        (proc / src / "images").mkdir(parents=True, exist_ok=True)
        (proc / src / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(12):
            split = "train" if i < 8 else ("val" if i < 10 else "test")
            img = np.clip(rng.normal(base_int, 20, size=(256, 256)), 0, 255).astype(np.uint8)
            msk = np.zeros((256, 256), dtype=np.uint8)
            msk[60:190, 40:110] = 255
            msk[60:190, 146:216] = 255
            name = f"cxr_{i:02d}"
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
                    "mask_area_frac": float((msk > 0).mean()),
                    "split": split,
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(proc / "manifest.csv", index=False)

    # Create synthetic tuning tables and s3_params
    tables_ref = tmp_path / "ref_tables"
    tables_ref.mkdir(parents=True, exist_ok=True)
    s3_rows = []
    for fold in cfg["folds"]:
        t_rows = []
        for clip in cfg["clahe_grid"]["clip_limit"]:
            for tile in cfg["clahe_grid"]["tile_grid"]:
                t_rows.append(
                    {
                        "clip_limit": float(clip),
                        "tile_grid": f"{tile[0]}x{tile[1]}",
                        "val_dice": float(0.95 + 0.001 * clip),
                    }
                )
        pd.DataFrame(t_rows).to_csv(tables_ref / f"tuning_{fold}.csv", index=False)
        s3_rows.append({"fold": fold, "clip_limit": 4.0, "tile_grid": "4x4"})
    pd.DataFrame(s3_rows).to_csv(tables_ref / "s3_params.csv", index=False)

    # Create synthetic pred.npz for all 3 folds x 4 scenarios x 2 seeds
    pred_dir = tmp_path / "preds"
    for fold in cfg["folds"]:
        parts = fold_split(manifest, fold)
        for scen in cfg["scenarios"]:
            for seed in cfg["train"]["seeds"]:
                ids, groups, probs = [], [], []
                for grp, key in (("val", "val"), ("in_source", "in_source_test"), ("foreign", "foreign")):
                    for r in parts[key].itertuples():
                        msk = cv2.imread(str(proc / r.mask_path), cv2.IMREAD_GRAYSCALE)
                        prob = msk.copy()
                        if grp == "foreign" and fold == "ho_darwin" and scen == "S0":
                            prob[140:, :] = 0  # degrade Darwin S0 so failure rate is non-trivial
                        ids.append(r.id)
                        groups.append(grp)
                        probs.append(prob)
                dest = pred_dir / fold / scen / f"seed{seed}"
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dest / "pred.npz",
                    ids=np.array(ids),
                    group=np.array(groups),
                    prob_u8=np.stack(probs),
                )

    return cfg, proc, tables_ref, pred_dir


def test_analyze_generates_all_tables_and_figures_deterministically(tmp_path):
    cfg, proc, tables_ref, pred_dir = _make_full_synthetic_workspace(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    t_run1 = run_analysis(pred_dir, out1, proc, tables_ref, cfg)
    t_run2 = run_analysis(pred_dir, out2, proc, tables_ref, cfg)

    expected_tables = [
        "T1_gap",
        "T2_scenarios",
        "T3_clinical",
        "T4_tuning",
        "T5_alarm",
        "T6_readiness",
        "T7_sample50",
    ]
    expected_figures = [
        "F1_dice_by_scenario.png",
        "F2_gap.png",
        "F3_alarm_scatter.png",
        "F4_clahe_grid.png",
        "F5_worst_cases.png",
        "F6_histograms.png",
    ]

    for tname in expected_tables:
        p1 = out1 / "tables" / f"{tname}.csv"
        p2 = out2 / "tables" / f"{tname}.csv"
        assert p1.exists() and p2.exists()
        assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")

    for fname in expected_figures:
        assert (out1 / "figures" / fname).exists()

    # Check key table invariants
    assert len(t_run1["T1_gap"]) == 12          # 3 folds x 4 scenarios
    assert len(t_run1["T2_scenarios"]) == 9     # 3 sources x (S1, S2, S3)
    assert len(t_run1["T3_clinical"]) == 12     # 3 folds x 4 scenarios
    assert len(t_run1["T4_tuning"]) == 36       # 3 folds x 12 CLAHE combos
    assert len(t_run1["T5_alarm"]) == 24        # 3 folds x 4 scenarios x 2 alarms
    assert len(t_run1["T6_readiness"]) == 48    # 2 alarms x (12 foreign + 12 in_source)
    assert len(t_run1["T7_sample50"]) == 24     # 3 folds x 4 scenarios x 2 alarms
