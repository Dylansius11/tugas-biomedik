import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_exp2 as AE  # noqa: E402

from cxrseg.config import load_config  # noqa: E402
from cxrseg.data import fold_split  # noqa: E402


def _prob(mask: np.ndarray, quality: float, rng) -> np.ndarray:
    """Probability map that recovers `quality` of the mask (lower quality shrinks the right lung)."""
    m = mask.astype(np.float32) / 255.0
    cut = int(40 + (1 - quality) * 150)
    m[:, 216 - cut : 216] = 0
    noise = rng.normal(0, 0.05, m.shape)
    return np.clip(np.rint((m * 0.9 + 0.05 + noise) * 255), 0, 255).astype(np.uint8)


def _workspace(tmp_path, bad_head: bool = False):
    cfg = load_config()
    cfg["stats"].update(bootstrap=200, sample_size_check=10, sample_repeats=50)
    cfg["train"]["seeds"] = [11, 22]
    cfg["exp2"]["fewshot"]["k"] = [2, 4]
    rng = np.random.default_rng(3)

    proc = tmp_path / "processed"
    rows = []
    for src, base in (("darwin", 150), ("montgomery", 80), ("shenzhen", 95)):
        for sub in ("images", "masks"):
            (proc / src / sub).mkdir(parents=True, exist_ok=True)
        for i in range(14):
            split = "train" if i < 8 else ("val" if i < 11 else "test")
            img = np.clip(rng.normal(base, 20, (256, 256)), 0, 255).astype(np.uint8)
            msk = np.zeros((256, 256), np.uint8)
            msk[60:190, 40:110] = 255
            msk[60:190, 146:216] = 255
            name = f"cxr_{i:02d}"
            cv2.imwrite(str(proc / src / "images" / f"{name}.png"), img)
            cv2.imwrite(str(proc / src / "masks" / f"{name}.png"), msk)
            rows.append(dict(id=f"{src}/{name}", source=src, name=name, image_path=f"{src}/images/{name}.png",
                             mask_path=f"{src}/masks/{name}.png", orig_h=512, orig_w=512, mask_area_frac=0.27, split=split))
    man = pd.DataFrame(rows)
    man.to_csv(proc / "manifest.csv", index=False)
    msk = cv2.imread(str(proc / "darwin/masks/cxr_00.png"), cv2.IMREAD_GRAYSCALE)

    pred, exp2 = tmp_path / "preds", tmp_path / "exp2"
    quality = {"E0": 0.80, "E1": 0.93, "E2": 0.97}
    for fold in cfg["folds"]:
        parts = fold_split(man, fold)
        ids, groups = [], []
        for g, key in (("val", "val"), ("in_source", "in_source_test"), ("foreign", "foreign")):
            ids += parts[key]["id"].tolist()
            groups += [g] * len(parts[key])
        for arm, q in quality.items():
            for seed in cfg["train"]["seeds"]:
                probs = np.stack([_prob(msk, q if g == "foreign" else 0.99, rng) for g in groups])
                if arm == "E0":
                    d = pred / fold / "S0" / f"seed{seed}"
                    d.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(d / "pred.npz", ids=np.array(ids), group=np.array(groups), prob_u8=probs)
                    continue
                d = exp2 / arm / fold / f"seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                head = np.zeros(len(ids), int)
                if bad_head and arm == "E2" and fold == "ho_darwin":
                    head[np.array(groups) == "foreign"] = 1
                np.savez_compressed(d / "pred.npz", ids=np.array(ids), group=np.array(groups), prob_u8=probs, head=head)
                (d / "val_dice.txt").write_text("0.958\n")
        pool, ev = AE.fewshot_eval_rows(man, fold)
        for k in cfg["exp2"]["fewshot"]["k"]:
            for seed in cfg["train"]["seeds"]:
                p = AE.fewshot_path(exp2, fold, k, seed)
                p.parent.mkdir(parents=True, exist_ok=True)
                probs = np.stack([_prob(msk, 0.97 + 0.005 * k, rng) for _ in range(len(ev))])
                np.savez_compressed(p, ids=np.array(ev["id"].tolist()), prob_u8=probs,
                                    adapt_ids=np.array(pool.sample(k, random_state=seed)["id"].tolist()))
    return cfg, proc, pred, exp2


def _run(tmp_path, cfg, proc, pred, exp2, out_name="out", huda=None):
    out = tmp_path / out_name
    out.mkdir()
    huda = huda or tmp_path / "no_huda"
    return out, AE.run_mode_a(pred, exp2, out, proc, huda, tmp_path / "no_main", cfg, AE.Progress("A", enabled=False))


def test_mode_a_writes_all_tables_and_figures(tmp_path):
    cfg, proc, pred, exp2 = _workspace(tmp_path)
    out, t = _run(tmp_path, cfg, proc, pred, exp2)

    for name in ("X0_validation", "X1_verify", "X2_e2_vs_e0", "X3_clinical", "X4_alarm", "X5_readiness", "X6_fewshot", "X7_sample50"):
        assert (out / f"{name}.csv").exists(), name
    for name in ("XF1_dice_box", "XF2_fewshot", "XF3_alarm_e2", "XF4_worst_e2"):
        assert (out / "figures" / f"{name}.png").exists(), name

    x0 = t["X0_validation"]
    assert x0["valid"].all()
    assert len(x0) == 3 * 2 * 3 + 3 * 2 * 2  # (E0, E1, E2) x folds x seeds + fewshot

    x2 = t["X2_e2_vs_e0"]
    tested = x2[x2["role"] == "diuji"]
    assert set(tested["arm"]) == {"E2"} and len(tested) == 3
    assert x2.loc[x2["role"] == "deskriptif", "p_holm"].isna().all()
    assert (tested["p_holm"] >= tested["p_value"]).all()
    assert (tested["mean_diff_dice"] > 0).all()  # E2 is built to be better than E0

    x3 = t["X3_clinical"]
    e0 = x3[x3["arm"] == "E0"]
    assert (e0["failure_diff_vs_e0_pct"] == 0).all()
    assert (x3[x3["arm"] == "E2"]["failure_rate_090_pct"].values <= e0["failure_rate_090_pct"].values).all()

    x4 = t["X4_alarm"]
    assert set(x4["scenario"]) == {"E0", "E2"} and set(x4["alarm"]) == {"A1", "A2"}
    # A1 depends only on the image, so tau and rate are identical for E0 and E2
    a1 = x4[x4["alarm"] == "A1"].pivot(index="fold", columns="scenario", values="tau")
    assert np.allclose(a1["E0"], a1["E2"])

    x6 = t["X6_fewshot"]
    assert set(x6["k"]) == {0, 2, 4}
    lo, mid, hi = x6["fail_pct_seed_min"], x6["fail_pct_seed_mean"], x6["fail_pct_seed_max"]
    assert (lo <= mid + 1e-9).all() and (mid <= hi + 1e-9).all()


def test_mode_a_is_deterministic_and_verifies_against_reference(tmp_path):
    cfg, proc, pred, exp2 = _workspace(tmp_path)
    out1, t1 = _run(tmp_path, cfg, proc, pred, exp2, "out1")

    # A reference table built from our own numbers must verify 100%
    huda = tmp_path / "huda"
    huda.mkdir()
    x3 = t1["X3_clinical"]
    avg = pd.concat(pd.read_csv(p) for p in sorted((out1 / "metrics").rglob("*.csv")))
    fo = avg[avg["group"] == "foreign"].groupby(["scenario", "fold", "id"])[["dice", "coverage"]].mean().reset_index()
    ref = pd.DataFrame(dict(
        arm=x3["arm"], fold=x3["fold"], in_median=x3["in_source_median_dice"], foreign_median=x3["foreign_median_dice"],
        foreign_mean=[fo[(fo.scenario == a) & (fo.fold == f)]["dice"].mean() for a, f in zip(x3["arm"], x3["fold"])],
        foreign_fail_pct=x3["failure_rate_090_pct"], coverage=x3["coverage_median"]))
    ref.to_csv(huda / "main.csv", index=False)

    out2, t2 = _run(tmp_path, cfg, proc, pred, exp2, "out2", huda=huda)
    assert len(t2["X1_verify"]) == len(ref) * 5 and t2["X1_verify"]["match"].all()
    for name in ("X2_e2_vs_e0", "X3_clinical", "X4_alarm", "X5_readiness", "X6_fewshot", "X7_sample50"):
        assert (out1 / f"{name}.csv").read_bytes() == (out2 / f"{name}.csv").read_bytes(), name


def test_wrong_head_on_darwin_stops_the_analysis(tmp_path):
    cfg, proc, pred, exp2 = _workspace(tmp_path, bad_head=True)
    with pytest.raises(SystemExit):
        _run(tmp_path, cfg, proc, pred, exp2)
    x0 = pd.read_csv(tmp_path / "out" / "X0_validation.csv")
    bad = x0[~x0["valid"]]
    assert set(bad["arm"]) == {"E2"} and set(bad["fold"]) == {"ho_darwin"}
    assert (bad["foreign_head"] == "anatomical").all()


def test_mode_main_tests_both_arms_with_holm_over_six(tmp_path):
    cfg, proc, pred, exp2 = _workspace(tmp_path)
    out = tmp_path / "main"
    t = AE.run_mode_main(pred, exp2, out, proc, tmp_path / "no_huda", tmp_path / "no_main", cfg,
                         AE.Progress("B", enabled=False))
    for name in ("T0_validation", "T1_gap", "T2_arms", "T3_clinical", "T5_alarm", "T6_readiness", "T7_sample50",
                 "T8_fewshot", "T9_verify"):
        assert (out / "tables" / f"{name}.csv").exists(), name
    for name in ("F1_dice_box", "F2_gap", "F3_alarm_e2", "F4_fewshot", "F5_worst_e2"):
        assert (out / "figures" / f"{name}.png").exists(), name
    assert not (out / "tables" / "T4_tuning.csv").exists()

    t2 = t["T2_arms"]
    assert (t2["role"] == "diuji").all() and len(t2) == 6 and (t2["holm_family_size"] == 6).all()
    assert set(t["T5_alarm"]["scenario"]) == {"E0", "E1", "E2"}
    assert set(t["T1_gap"]["scenario"]) == {"E0", "E1", "E2"}
