"""Analisis Percobaan 2 (post-hoc) dengan alat yang sama dengan hasil utama.

Follows docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2.md (N7-N12) and docs/plans/BRANCHING-PERCOBAAN-2.md.
E0 = S0 of the main experiment (single head). E1/E2 = two-head U-Net from Percobaan 2.
scripts/analyze.py is not modified; its table builders for T5-T7 are reused unchanged.

Mode "a" (Percobaan 2 sebagai analisis tambahan):
  X0_validation.csv   N7  54 berkas Percobaan 2 + 9 berkas E0
  X1_verify.csv       N8  angka kita vs tabel Huda (results/exp2/tables) dan vs T5 S0 hasil utama
  X2_e2_vs_e0.csv     N9  Wilcoxon + bootstrap, Holm atas 3 uji (E2 vs E0); E1 deskriptif
  X3_clinical.csv     N9  celah, tingkat gagal 0,85/0,90/0,93, cakupan, galat luas
  X4_alarm.csv        N10 A1/A2 pada E0 dan E2 (+ A3 Huda)
  X5_readiness.csv    N10 siap vs keputusan alarm
  X6_fewshot.csv      N11 kurva few-shot dengan rentang antar seed
  X7_sample50.csv     N10 sebaran tingkat alarm, 1.000 sampel 50 citra
  figures/XF1..XF4    N12

Mode "main" (skenario B, Percobaan 2 sebagai hasil utama), {out}/tables and {out}/figures:
  T0_validation, T1_gap, T2_arms (Holm atas 6 uji: E1 dan E2 vs E0), T3_clinical, T5_alarm, T6_readiness,
  T7_sample50, T8_fewshot, T9_verify; F1_dice_box, F2_gap, F3_alarm_e2, F4_fewshot, F5_worst_e2.
  Tidak ada T4: Percobaan 2 tidak menguji CLAHE.

Usage:
    python scripts/analyze_exp2.py --mode main --pred-dir <predictions> --exp2-dir <percobaan2> --out results/exp2/main
    python scripts/analyze_exp2.py --mode a --pred-dir <predictions> --exp2-dir <percobaan2> --out results/exp2/nabil
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import analyze as AN  # noqa: E402
import progress as PG  # noqa: E402

from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.data import fold_split, held_out_source, load_arrays, load_manifest  # noqa: E402
from cxrseg.evaluate import evaluate_run  # noqa: E402
from cxrseg.metrics import evaluate_pair  # noqa: E402
from cxrseg.postprocess import postprocess_mask  # noqa: E402
from cxrseg.preprocess import enhance  # noqa: E402
from cxrseg.shift import alarm_threshold, intensity_distance, mean_entropy, pooled_hist  # noqa: E402
from cxrseg.stats import average_over_seeds, holm, is_meaningful, paired_comparison  # noqa: E402

ARMS = ["E0", "E1", "E2"]
ARM_COLORS = {"E0": "#257de0", "E1": "#f47230", "E2": "#00b894"}
SRC_COLORS = {"darwin": "#257de0", "montgomery": "#f47230", "shenzhen": "#00b894"}
ARM_LABEL = {"E0": "E0 satu head", "E1": "E1 dua head", "E2": "E2 dua head + augmentasi kuat"}
SRC = AN.SOURCE_TITLE

MODES = {
    # tested: arms that enter the Holm family. alarm_arms: arms whose A1/A2 alarm is evaluated.
    "a": {"tested": ["E2"], "alarm_arms": ["E0", "E2"]},
    # Skenario B: Percobaan 2 sebagai hasil utama. E1 and E2 vs E0 like T2 tests S1-S3 vs S0.
    "main": {"tested": ["E1", "E2"], "alarm_arms": ["E0", "E1", "E2"]},
}


# ----------------------------------------------------------------------------- progress
class Progress:
    """Writes progress/running.txt and progress/log.txt so `python scripts/progress.py` can show the stage."""

    def __init__(self, track: str, enabled: bool = True):
        self.track, self.enabled = track, enabled

    def __call__(self, step: str, msg: str) -> None:
        print(f"[{step}] {msg}", flush=True)
        if not self.enabled:
            return
        PG.RUNNING.parent.mkdir(parents=True, exist_ok=True)
        PG.RUNNING.write_text(f"{os.getpid()} | {self.track} {step} | {msg} | sejak {datetime.now():%H:%M:%S}\n", encoding="utf-8")
        PG.write_log(self.track, step, msg)

    def done(self, msg: str) -> None:
        print(msg, flush=True)
        if self.enabled:
            PG.write_log(self.track, "done", msg)
            PG.RUNNING.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- paths
def pred_path(arm: str, fold: str, seed: int, pred_dir: Path, exp2_dir: Path) -> Path:
    if arm == "E0":
        return pred_dir / fold / "S0" / f"seed{seed}" / "pred.npz"
    return exp2_dir / arm / fold / f"seed{seed}" / "pred.npz"


def fewshot_path(exp2_dir: Path, fold: str, k: int, seed: int, arm: str = "E2") -> Path:
    return exp2_dir / "fewshot" / arm / fold / f"k{k}_seed{seed}.npz"


def fewshot_eval_rows(manifest: pd.DataFrame, fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same rule as cxrseg.exp2.fewshot_split: adapt pool = held-out train split, eval = its val + test."""
    tgt = manifest[manifest["source"] == held_out_source(fold)]
    return tgt[tgt["split"] == "train"], tgt[tgt["split"] != "train"]


def expected_foreign_head(cfg: dict, fold: str) -> str:
    """Head that must be used on the held-out source: its own convention if a training source taught it, else visible."""
    e = cfg["exp2"]
    ho = held_out_source(fold)
    trained = {e["convention"][s] for s in cfg["sources"] if s != ho}
    own = e["convention"][ho]
    return own if own in trained else "visible"


# ----------------------------------------------------------------------------- N7 validation
def validate_inputs(pred_dir: Path, exp2_dir: Path, manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seeds, folds, heads = cfg["train"]["seeds"], cfg["folds"], cfg["exp2"]["heads"]
    rows: list[dict] = []
    for fold in folds:
        for seed in seeds:
            p0 = pred_path("E0", fold, seed, pred_dir, exp2_dir)
            ref = None
            r0 = {"kind": "pred", "arm": "E0", "fold": fold, "seed": seed, "k": np.nan, "path": str(p0), "exists": p0.exists()}
            if p0.exists():
                with np.load(p0) as z:
                    ref = (z["ids"].astype(str), z["group"].astype(str), z["prob_u8"].shape)
                    r0.update(keys_ok={"ids", "group", "prob_u8"} <= set(z.files), n=len(ref[0]))
            r0["valid"] = bool(r0["exists"] and r0.get("keys_ok", False))
            rows.append(r0)
            for arm in ("E1", "E2"):
                p = pred_path(arm, fold, seed, pred_dir, exp2_dir)
                r = {"kind": "pred", "arm": arm, "fold": fold, "seed": seed, "k": np.nan, "path": str(p), "exists": p.exists()}
                if p.exists():
                    with np.load(p) as z:
                        ids, grp = z["ids"].astype(str), z["group"].astype(str)
                        r["keys_ok"] = {"ids", "group", "prob_u8", "head"} <= set(z.files)
                        r["n"] = len(ids)
                        r["same_ids_as_e0"] = bool(ref is not None and np.array_equal(ids, ref[0]))
                        r["same_group_as_e0"] = bool(ref is not None and np.array_equal(grp, ref[1]))
                        r["same_shape_as_e0"] = bool(ref is not None and z["prob_u8"].shape == ref[2])
                        fh = sorted({heads[int(h)] for h in z["head"][grp == "foreign"]}) if "head" in z.files else []
                        r["foreign_head"] = "+".join(fh)
                        r["expected_head"] = expected_foreign_head(cfg, fold)
                        r["head_ok"] = fh == [r["expected_head"]]
                vd = p.parent / "val_dice.txt"
                r["val_dice"] = float(vd.read_text().split()[0]) if vd.exists() else np.nan
                r["valid"] = bool(
                    r["exists"] and r.get("keys_ok") and r.get("same_ids_as_e0") and r.get("same_group_as_e0")
                    and r.get("same_shape_as_e0") and r.get("head_ok") and 0 < r["val_dice"] <= 1
                )
                rows.append(r)
    for fold in folds:
        pool, ev = fewshot_eval_rows(manifest, fold)
        pool_ids, ev_ids = set(pool["id"].astype(str)), ev["id"].astype(str).tolist()
        for k in cfg["exp2"]["fewshot"]["k"]:
            for seed in seeds:
                p = fewshot_path(exp2_dir, fold, k, seed)
                r = {"kind": "fewshot", "arm": "E2", "fold": fold, "seed": seed, "k": k, "path": str(p), "exists": p.exists()}
                if p.exists():
                    with np.load(p) as z:
                        r["keys_ok"] = {"ids", "prob_u8", "adapt_ids"} <= set(z.files)
                        ids, ad = z["ids"].astype(str).tolist(), set(z["adapt_ids"].astype(str))
                        r["n"] = len(ids)
                        r["eval_ids_ok"] = sorted(ids) == sorted(ev_ids)
                        r["adapt_ok"] = len(ad) == k and ad <= pool_ids and not (ad & set(ids))
                r["valid"] = bool(r["exists"] and r.get("keys_ok") and r.get("eval_ids_ok") and r.get("adapt_ok"))
                rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- N8 evaluation
def evaluate_arms(pred_dir: Path, exp2_dir: Path, manifest: pd.DataFrame, processed_dir: Path, cfg: dict,
                  metrics_dir: Path, gt_cache: dict) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for fold in cfg["folds"]:
            for seed in cfg["train"]["seeds"]:
                df = evaluate_run(pred_path(arm, fold, seed, pred_dir, exp2_dir), manifest, processed_dir, cfg,
                                  fold=fold, scenario=arm, seed=seed, gt_cache=gt_cache)
                out = metrics_dir / arm / fold / f"seed{seed}.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
                frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _close(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(abs(float(a) - float(b)) <= tol)


def verify_against_huda(avg: pd.DataFrame, x2: pd.DataFrame, x6: pd.DataFrame, x4: pd.DataFrame,
                        huda_dir: Path, main_tables: Path, cfg: dict) -> pd.DataFrame:
    """X1: every number we can recompute from Huda's tables, with tolerance (3 decimals for Dice, 0.05 pp for %)."""
    fail = float(cfg["metrics"]["failure_dice"])
    rows: list[dict] = []

    def add(check, arm, fold, metric, ours, ref, tol):
        rows.append(dict(check=check, arm=arm, fold=fold, metric=metric, ours=ours, reference=ref,
                         diff=(np.nan if pd.isna(ref) else float(ours) - float(ref)), match=_close(ours, ref, tol)))

    main_csv = huda_dir / "main.csv"
    if main_csv.exists():
        h = pd.read_csv(main_csv)
        for r in h.itertuples():
            g = avg[(avg["scenario"] == r.arm) & (avg["fold"] == r.fold)]
            ins, fo = g[g["group"] == "in_source"], g[g["group"] == "foreign"]
            add("huda main.csv", r.arm, r.fold, "in_median", ins["dice"].median(), r.in_median, 5e-4)
            add("huda main.csv", r.arm, r.fold, "foreign_median", fo["dice"].median(), r.foreign_median, 5e-4)
            add("huda main.csv", r.arm, r.fold, "foreign_mean", fo["dice"].mean(), r.foreign_mean, 5e-4)
            add("huda main.csv", r.arm, r.fold, "foreign_fail_pct", (fo["dice"] < fail).mean() * 100, r.foreign_fail_pct, 0.05)
            add("huda main.csv", r.arm, r.fold, "coverage_median", fo["coverage"].median(), r.coverage, 5e-4)
    vs_csv = huda_dir / "vs_E0.csv"
    if vs_csv.exists():
        h = pd.read_csv(vs_csv)
        for r in h.itertuples():
            m = x2[(x2["arm"] == r.arm) & (x2["fold"] == r.fold)]
            if m.empty:
                continue
            add("huda vs_E0.csv", r.arm, r.fold, "mean_diff", m["mean_diff_dice"].iloc[0], r.mean_diff, 5e-4)
            add("huda vs_E0.csv", r.arm, r.fold, "p_value (sebelum Holm)", m["p_value"].iloc[0], r.p, max(1e-6, abs(r.p) * 1e-3))
    fs_csv = huda_dir / "fewshot.csv"
    if fs_csv.exists():
        h = pd.read_csv(fs_csv)
        for r in h.itertuples():
            m = x6[(x6["fold"] == r.fold) & (x6["k"] == r.k) & (x6["model"] == r.model)]
            if m.empty:
                continue
            add("huda fewshot.csv", r.model, r.fold, f"k{r.k} median", m["median_dice"].iloc[0], r.median, 5e-4)
            add("huda fewshot.csv", r.model, r.fold, f"k{r.k} fail_pct", m["fail_pct"].iloc[0], r.fail_pct, 0.05)
    t5 = main_tables / "T5_alarm.csv"
    if t5.exists():
        ref = pd.read_csv(t5)
        ref = ref[ref["scenario"] == "S0"]
        for r in ref.itertuples():
            m = x4[(x4["scenario"] == "E0") & (x4["fold"] == r.fold) & (x4["alarm"] == r.alarm)]
            if m.empty:
                continue
            add("hasil utama T5 (S0)", "E0", r.fold, f"{r.alarm} tau", m["tau"].iloc[0], r.tau, 1e-9)
            add("hasil utama T5 (S0)", "E0", r.fold, f"{r.alarm} alarm_rate_pct", m["alarm_rate_pct"].iloc[0], r.alarm_rate_pct, 1e-9)
    return pd.DataFrame(rows, columns=["check", "arm", "fold", "metric", "ours", "reference", "diff", "match"])


# ----------------------------------------------------------------------------- N9 statistics
def build_comparisons(avg: pd.DataFrame, cfg: dict, tested: list[str]) -> pd.DataFrame:
    s = cfg["stats"]
    n_boot, alpha, min_eff, bseed = int(s["bootstrap"]), float(s["alpha"]), float(s["min_effect_dice"]), int(cfg["data"]["sample_seed"])
    rows = []
    for arm in ("E1", "E2"):
        for fold in cfg["folds"]:
            ho = held_out_source(fold)
            res = paired_comparison(avg, baseline="E0", scenario=arm, source=ho, metric="dice",
                                    n_boot=n_boot, seed=bseed, alpha=alpha, run_wilcoxon=True)
            rows.append(dict(fold=fold, foreign_source=ho, arm=arm, baseline="E0", role="diuji" if arm in tested else "deskriptif",
                             n_images=res["n_images"], mean_diff_dice=res["mean_diff"], ci_low=res["ci_low"],
                             ci_high=res["ci_high"], p_value=res["p_value"]))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    df["meaningful"] = pd.Series([pd.NA] * len(df), dtype="object")
    fam = df["role"] == "diuji"
    df.loc[fam, "p_holm"] = holm(df.loc[fam, "p_value"].tolist())
    df.loc[fam, "meaningful"] = [is_meaningful(p, d, alpha, min_eff) for p, d in zip(df.loc[fam, "p_holm"], df.loc[fam, "mean_diff_dice"])]
    df["holm_family_size"] = int(fam.sum())
    return df


def build_clinical(avg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["stats"]
    n_boot, alpha, bseed = int(s["bootstrap"]), float(s["alpha"]), int(cfg["data"]["sample_seed"])
    thr = [0.85, float(cfg["metrics"]["failure_dice"]), 0.93]
    thr = sorted(set(thr + [float(t) for t in cfg["metrics"]["failure_sensitivity"]]))
    rows = []
    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        for arm in ARMS:
            g = avg[(avg["fold"] == fold) & (avg["scenario"] == arm)]
            ins, fo = g[g["group"] == "in_source"], g[g["group"] == "foreign"]
            r = dict(fold=fold, foreign_source=ho, arm=arm, n_images=len(fo),
                     in_source_median_dice=ins["dice"].median(), foreign_median_dice=fo["dice"].median(),
                     generalization_gap=ins["dice"].median() - fo["dice"].median())
            for t in thr:
                r[f"failure_rate_{int(round(t * 100)):03d}_pct"] = (fo["dice"] < t).mean() * 100
            r.update(coverage_mean=fo["coverage"].mean(), coverage_median=fo["coverage"].median(),
                     median_area_error_pct=fo["area_error_pct"].median(),
                     median_abs_area_error_pct=fo["area_error_pct"].abs().median(),
                     median_hd95=fo["hd95"].median())
            if arm == "E0":
                r.update(failure_diff_vs_e0_pct=0.0, failure_diff_ci_low_pct=0.0, failure_diff_ci_high_pct=0.0,
                         coverage_diff_vs_e0=0.0, coverage_diff_ci_low=0.0, coverage_diff_ci_high=0.0)
            else:
                f = paired_comparison(avg, "E0", arm, ho, metric="failed", n_boot=n_boot, seed=bseed, alpha=alpha, run_wilcoxon=False)
                c = paired_comparison(avg, "E0", arm, ho, metric="coverage", n_boot=n_boot, seed=bseed, alpha=alpha, run_wilcoxon=False)
                r.update(failure_diff_vs_e0_pct=f["mean_diff"] * 100, failure_diff_ci_low_pct=f["ci_low"] * 100,
                         failure_diff_ci_high_pct=f["ci_high"] * 100, coverage_diff_vs_e0=c["mean_diff"],
                         coverage_diff_ci_low=c["ci_low"], coverage_diff_ci_high=c["ci_high"])
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- N10 alarm
def compute_shift_records(pred_dir: Path, exp2_dir: Path, manifest: pd.DataFrame, processed_dir: Path,
                          avg: pd.DataFrame, cfg: dict, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same rules as analyze.compute_shift_records for S0: A1 reference = train images of the 2 training sources,
    tau = 95th percentile of their val scores, A2 = per-image mean entropy averaged over the 3 seeds."""
    perc = float(cfg["readiness"]["alarm_percentile"])
    score_rows, tau_rows = [], []
    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        split = fold_split(manifest, fold)
        groups = {"val": "val", "in_source": "in_source_test", "foreign": "foreign"}
        train_imgs, _ = load_arrays(split["train"], processed_dir)
        ref_hist = pooled_hist(np.stack([enhance(im, "S0", None) for im in train_imgs]))
        a1: dict[tuple[str, str], float] = {}
        ids_by_group: dict[str, list[str]] = {}
        for g, key in groups.items():
            imgs, _ = load_arrays(split[key], processed_dir)
            ids_by_group[g] = split[key]["id"].astype(str).tolist()
            for i, im in zip(ids_by_group[g], imgs):
                a1[(g, i)] = intensity_distance(enhance(im, "S0", None), ref_hist)
        for arm in arms:
            a2: dict[tuple[str, str], list[float]] = {}
            for seed in cfg["train"]["seeds"]:
                with np.load(pred_path(arm, fold, seed, pred_dir, exp2_dir)) as z:
                    for i, g, p in zip(z["ids"], z["group"], z["prob_u8"]):
                        a2.setdefault((str(g), str(i)), []).append(mean_entropy(p))
            tau1 = alarm_threshold([a1[("val", i)] for i in ids_by_group["val"]], perc)
            tau2 = alarm_threshold([float(np.mean(a2[("val", i)])) for i in ids_by_group["val"]], perc)
            tau_rows += [dict(fold=fold, foreign_source=ho, scenario=arm, alarm="A1", tau=tau1),
                         dict(fold=fold, foreign_source=ho, scenario=arm, alarm="A2", tau=tau2)]
            ev = avg[(avg["fold"] == fold) & (avg["scenario"] == arm)].set_index(["group", "id"])
            for g, ids in ids_by_group.items():
                for i in ids:
                    d, f = (float(ev.loc[(g, i), "dice"]), bool(ev.loc[(g, i), "failed"])) if (g, i) in ev.index else (np.nan, False)
                    score_rows.append(dict(fold=fold, foreign_source=ho, scenario=arm, group=g, id=i,
                                           A1=a1[(g, i)], A2=float(np.mean(a2[(g, i)])), dice=d, failed=f))
    return pd.DataFrame(score_rows), pd.DataFrame(tau_rows)


def build_alarm_table(scores: pd.DataFrame, taus: pd.DataFrame, huda_dir: Path, arms: list[str]) -> pd.DataFrame:
    t5 = AN.build_t5_alarm(scores, taus)
    caught = []
    for r in t5.itertuples():
        fo = scores[(scores["fold"] == r.fold) & (scores["scenario"] == r.scenario) & (scores["group"] == "foreign")]
        f = fo[fo["failed"]]
        caught.append((f[r.alarm] > r.tau).mean() * 100 if len(f) else np.nan)
    t5["caught_fail_pct"] = caught
    t5["source_of_numbers"] = "Nabil (rata-rata 3 seed per citra)"
    xa = huda_dir / "xai_alarm.csv"
    if xa.exists():
        h = pd.read_csv(xa)
        h = h[h["arm"].isin(arms)].groupby(["arm", "fold"], as_index=False).mean(numeric_only=True)
        for r in h.itertuples():
            base = t5[(t5["scenario"] == r.arm) & (t5["fold"] == r.fold)].iloc[0]
            t5.loc[len(t5)] = {
                "fold": r.fold, "foreign_source": base["foreign_source"], "scenario": r.arm, "alarm": "A3", "tau": r.tau,
                "n_foreign": base["n_foreign"], "n_failed": base["n_failed"], "alarm_rate_pct": r.foreign_alarm_pct,
                "in_source_alarm_rate_pct": r.in_alarm_pct, "spearman_rho": np.nan, "spearman_p": np.nan,
                "auroc": r.auroc_A3 if not pd.isna(r.auroc_A3) else "tidak dapat dihitung",
                "caught_fail_pct": r.caught_fail_pct, "source_of_numbers": "Huda xai_alarm.csv (rata-rata per seed)",
            }
    order = {a: i for i, a in enumerate(["A1", "A2", "A3"])}
    return t5.sort_values(["fold", "scenario", "alarm"], key=lambda c: c.map(order) if c.name == "alarm" else c).reset_index(drop=True)


def build_readiness_table(avg: pd.DataFrame, scores: pd.DataFrame, taus: pd.DataFrame, huda_dir: Path,
                          cfg: dict, arms: list[str]) -> pd.DataFrame:
    t6 = AN.build_t6_readiness(avg, scores, taus, cfg)
    t6["source_of_numbers"] = "Nabil"
    xv = huda_dir / "xai_verdicts.csv"
    if xv.exists():
        h = pd.read_csv(xv)
        for r in h[h["arm"].isin(arms)].itertuples():
            base = t6[(t6["scenario"] == r.arm) & (t6["fold"] == r.fold) & (t6["group"] == "foreign")].iloc[0]
            dec = "siap" if bool(r.says_ready) else "belum siap"
            row = base.to_dict()
            row.update(alarm="A3", alarm_rate_pct=r.alarm_rate * 100, alarm_decision=dec,
                       match=bool(dec == base["truth_label"]), source_of_numbers="Huda xai_verdicts.csv (seed 11)")
            t6.loc[len(t6)] = row
    return t6.sort_values(["alarm", "fold", "scenario", "group"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- N11 few-shot
def build_fewshot(raw: pd.DataFrame, avg: pd.DataFrame, exp2_dir: Path, manifest: pd.DataFrame,
                  processed_dir: Path, cfg: dict, gt_cache: dict) -> pd.DataFrame:
    fail = float(cfg["metrics"]["failure_dice"])
    pp = cfg["postprocess"]
    man = manifest.set_index("id")
    rows = []

    def summarise(fold, k, model, per_seed: dict[int, pd.Series]):
        mean = pd.concat(per_seed.values(), axis=1).mean(axis=1)
        f_seed = [(s < fail).mean() * 100 for s in per_seed.values()]
        m_seed = [s.median() for s in per_seed.values()]
        rows.append(dict(fold=fold, foreign_source=held_out_source(fold), k=k, model=model, n_images=len(mean),
                         median_dice=mean.median(), fail_pct=(mean < fail).mean() * 100,
                         fail_pct_seed_mean=float(np.mean(f_seed)), fail_pct_seed_min=min(f_seed), fail_pct_seed_max=max(f_seed),
                         median_dice_seed_mean=float(np.mean(m_seed)), median_dice_seed_min=min(m_seed),
                         median_dice_seed_max=max(m_seed)))

    for fold in cfg["folds"]:
        _, ev = fewshot_eval_rows(manifest, fold)
        ev_ids = set(ev["id"].astype(str))
        for arm in ("E0", "E2"):
            sub = raw[(raw["scenario"] == arm) & (raw["fold"] == fold) & (raw["group"] == "foreign") & raw["id"].isin(ev_ids)]
            summarise(fold, 0, arm, {s: g.set_index("id")["dice"] for s, g in sub.groupby("seed")})
        for k in cfg["exp2"]["fewshot"]["k"]:
            per_seed = {}
            for seed in cfg["train"]["seeds"]:
                with np.load(fewshot_path(exp2_dir, fold, k, seed)) as z:
                    d = {}
                    for i, p in zip(z["ids"].astype(str), z["prob_u8"]):
                        if i not in gt_cache:
                            gt_cache[i] = cv2.imread(str(processed_dir / str(man.loc[i, "mask_path"])), cv2.IMREAD_GRAYSCALE) > 0
                        pm = postprocess_mask(p, threshold=float(pp["threshold"]), keep_components=int(pp["keep_components"]),
                                              closing_kernel=int(pp["closing_kernel"]))
                        d[i] = evaluate_pair(pm, gt_cache[i], failure_dice=fail, hd95_empty=float(cfg["metrics"]["hd95_empty_value"]))["dice"]
                    per_seed[seed] = pd.Series(d)
            summarise(fold, k, "E2+k", per_seed)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- N12 figures
def plot_dice_box(avg: pd.DataFrame, cfg: dict, path: Path, title: str) -> None:
    fail = float(cfg["metrics"]["failure_dice"])
    fig, axes = plt.subplots(1, len(cfg["folds"]), figsize=(4.3 * len(cfg["folds"]), 4.0), sharey=True)
    for ax, fold in zip(np.atleast_1d(axes), cfg["folds"]):
        sub = avg[(avg["fold"] == fold) & (avg["group"] == "foreign")]
        data = [sub[sub["scenario"] == a]["dice"].values for a in ARMS]
        bp = ax.boxplot(data, patch_artist=True, flierprops={"markersize": 3, "alpha": 0.4}, medianprops={"color": "black"})
        for patch, a in zip(bp["boxes"], ARMS):
            patch.set_facecolor(ARM_COLORS[a])
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(ARMS) + 1), ARMS)
        for j, a in enumerate(ARMS, start=1):
            fr = (sub[sub["scenario"] == a]["dice"] < fail).mean() * 100
            ax.text(j, 0.02, f"gagal\n{fr:.1f}%", ha="center", va="bottom", fontsize=7)
        ax.axhline(fail, color="crimson", linestyle="--", linewidth=1.0, label=f"Ambang gagal ({fail:.2f})")
        ax.set_title(f"Sumber asing: {SRC[held_out_source(fold)]}")
        ax.set_xlabel("Lengan (kode)")
        ax.set_ylim(0.0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Koefisien Dice (tanpa satuan)")
    axes[0].legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.12))
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fewshot(x6: pd.DataFrame, cfg: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for fold in cfg["folds"]:
        src = held_out_source(fold)
        g = x6[(x6["fold"] == fold) & (x6["model"] != "E0")].sort_values("k")
        ax.fill_between(g["k"], g["fail_pct_seed_min"], g["fail_pct_seed_max"], color=SRC_COLORS[src], alpha=0.15, linewidth=0)
        ax.plot(g["k"], g["fail_pct_seed_mean"], marker="o", markersize=5, linewidth=2, color=SRC_COLORS[src], label=SRC[src])
        e0 = x6[(x6["fold"] == fold) & (x6["model"] == "E0")]
        ax.plot([0], e0["fail_pct_seed_mean"], marker="x", markersize=8, color=SRC_COLORS[src], linestyle="none")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks([0, 5, 10, 20, 50], ["0", "5", "10", "20", "50"])
    ax.set_xlabel("Jumlah citra berlabel dari sumber baru, k (citra)")
    ax.set_ylabel("Citra gagal, Dice < 0,90 (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.plot([], [], marker="x", color="gray", linestyle="none", label="E0 pada k = 0")
    ax.legend(frameon=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alarm_scatter(scores: pd.DataFrame, taus: pd.DataFrame, cfg: dict, arm: str, path: Path, title: str) -> None:
    fail = float(cfg["metrics"]["failure_dice"])
    folds = cfg["folds"]
    fig, axes = plt.subplots(2, len(folds), figsize=(4.3 * len(folds), 7.2), sharey=True)
    for c, fold in enumerate(folds):
        fo = scores[(scores["fold"] == fold) & (scores["scenario"] == arm) & (scores["group"] == "foreign")]
        for r, (alarm, xlabel) in enumerate((("A1", "Skor A1: jarak histogram (tanpa satuan)"),
                                             ("A2", "Skor A2: entropi rata-rata (nat)"))):
            ax = axes[r, c]
            tau = float(taus[(taus["fold"] == fold) & (taus["scenario"] == arm) & (taus["alarm"] == alarm)]["tau"].iloc[0])
            col = np.where(fo["failed"], "#c0392b", ARM_COLORS[arm])
            ax.scatter(fo[alarm], fo["dice"], s=9, c=col, alpha=0.6, linewidths=0)
            ax.axvline(tau, color="black", linestyle=":", linewidth=1.0, label=f"tau = {tau:.3g}")
            ax.axhline(fail, color="crimson", linestyle="--", linewidth=1.0)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{alarm}, sumber asing {SRC[held_out_source(fold)]}")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left", frameon=False)
            if c == 0:
                ax.set_ylabel("Koefisien Dice (tanpa satuan)")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_worst(avg: pd.DataFrame, pred_dir: Path, exp2_dir: Path, manifest: pd.DataFrame, processed_dir: Path,
               cfg: dict, arm: str, path: Path, title: str) -> None:
    seed0 = int(cfg["train"]["seeds"][0])
    pp = cfg["postprocess"]
    man = manifest.set_index("id")
    dice = avg.set_index(["fold", "scenario", "group", "id"])["dice"]
    chosen = []
    for fold in cfg["folds"]:
        sub = avg[(avg["fold"] == fold) & (avg["scenario"] == arm) & (avg["group"] == "foreign")]
        chosen += [(fold, str(i)) for i in sub.sort_values("dice").head(3)["id"]]
    probs: dict[tuple[str, str, str], np.ndarray] = {}
    for fold in cfg["folds"]:
        for a in ("E0", arm):
            with np.load(pred_path(a, fold, seed0, pred_dir, exp2_dir)) as z:
                want = {i for f, i in chosen if f == fold}
                for i, p in zip(z["ids"].astype(str), z["prob_u8"]):
                    if i in want:
                        probs[(fold, a, i)] = p
    cols = [("GT", "Mask asli"), ("E0", f"Prediksi E0 (seed {seed0})"), (arm, f"Prediksi {arm} (seed {seed0})")]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9.3, 2.9 * len(chosen)))
    for r, (fold, i) in enumerate(chosen):
        img = cv2.imread(str(processed_dir / str(man.loc[i, "image_path"])), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(processed_dir / str(man.loc[i, "mask_path"])), cv2.IMREAD_GRAYSCALE) > 0
        for c, (key, name) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.1)
            if key == "GT":
                ax.set_title(f"{i[:38]}\n{name}", fontsize=7)
            else:
                pm = postprocess_mask(probs[(fold, key, i)], threshold=float(pp["threshold"]),
                                      keep_components=int(pp["keep_components"]), closing_kernel=int(pp["closing_kernel"]))
                ax.contour(pm, levels=[0.5], colors="red", linewidths=1.1)
                ax.set_title(f"{name}\nDice rata-rata 3 seed = {dice.get((fold, key, 'foreign', i), np.nan):.3f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(SRC[held_out_source(fold)], fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- driver
def run_mode_a(pred_dir: Path, exp2_dir: Path, out: Path, processed_dir: Path, huda_dir: Path, main_tables: Path,
               cfg: dict, prog: Progress) -> dict[str, pd.DataFrame]:
    mode = MODES["a"]
    AN._configure_matplotlib()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(processed_dir)
    gt_cache: dict[str, np.ndarray] = {}

    prog("N7", "validasi berkas Percobaan 2")
    x0 = validate_inputs(pred_dir, exp2_dir, manifest, cfg)
    x0.to_csv(out / "X0_validation.csv", index=False)
    bad = x0[~x0["valid"]]
    if not bad.empty:
        raise SystemExit(f"{len(bad)} berkas tidak valid, lihat {out / 'X0_validation.csv'}")
    prog("N7", f"{int((x0['arm'] != 'E0').sum())} berkas Percobaan 2 + {int((x0['arm'] == 'E0').sum())} berkas E0 valid")

    prog("N8", "evaluasi 27 run (E0, E1, E2)")
    raw = evaluate_arms(pred_dir, exp2_dir, manifest, processed_dir, cfg, out / "metrics", gt_cache)
    avg = average_over_seeds(raw, failure_dice=float(cfg["metrics"]["failure_dice"]))

    prog("N9", "statistik E2 vs E0 dan ringkasan klinis")
    x2 = build_comparisons(avg, cfg, mode["tested"])
    x3 = build_clinical(avg, cfg)

    prog("N10", "alarm A1/A2 pada E0 dan E2")
    scores, taus = compute_shift_records(pred_dir, exp2_dir, manifest, processed_dir, avg, cfg, mode["alarm_arms"])
    x4 = build_alarm_table(scores, taus, huda_dir, mode["alarm_arms"])
    x5 = build_readiness_table(avg, scores, taus, huda_dir, cfg, mode["alarm_arms"])
    x7 = AN.build_t7_sample50(scores, taus, cfg)

    prog("N11", "few-shot")
    x6 = build_fewshot(raw, avg, exp2_dir, manifest, processed_dir, cfg, gt_cache)

    prog("N8", "cocokkan dengan angka Huda dan hasil utama")
    x1 = verify_against_huda(avg, x2, x6, x4, huda_dir, main_tables, cfg)

    tables = {"X1_verify": x1, "X2_e2_vs_e0": x2, "X3_clinical": x3, "X4_alarm": x4, "X5_readiness": x5,
              "X6_fewshot": x6, "X7_sample50": x7}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    scores.to_csv(out / "alarm_scores.csv", index=False)

    prog("N12", "gambar XF1-XF4")
    plot_dice_box(avg, cfg, fig_dir / "XF1_dice_box.png", "Sebaran Dice pada Sumber Asing: E0, E1, E2 (post-hoc)")
    plot_fewshot(x6, cfg, fig_dir / "XF2_fewshot.png", "Adaptasi Few-shot E2 (garis: rata-rata tingkat gagal 3 seed, pita: rentang seed)")
    plot_alarm_scatter(scores, taus, cfg, "E2", fig_dir / "XF3_alarm_e2.png",
                       "Skor Alarm vs Dice pada E2, Sumber Asing (merah = gagal)")
    plot_worst(avg, pred_dir, exp2_dir, manifest, processed_dir, cfg, "E2", fig_dir / "XF4_worst_e2.png",
               "Tiga Kasus Terburuk E2 per Sumber Asing (hijau = mask asli, merah = prediksi)")
    n_bad = int((~x1["match"]).sum()) if not x1.empty else 0
    prog.done(f"mode a selesai: {len(tables)} tabel, 4 gambar, verifikasi {len(x1) - n_bad}/{len(x1)} cocok")
    return {"X0_validation": x0, **tables}


def plot_gap(t1: pd.DataFrame, cfg: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(cfg["folds"]), figsize=(4.5 * len(cfg["folds"]), 4.2), sharey=True)
    for ax, fold in zip(np.atleast_1d(axes), cfg["folds"]):
        sub = t1[t1["fold"] == fold].set_index("scenario").reindex(ARMS)
        x, w = np.arange(len(ARMS)), 0.36
        ax.bar(x - w / 2, sub["in_source_median_dice"], w - 0.02, label="Dalam sumber", color="#257de0")
        ax.bar(x + w / 2, sub["foreign_median_dice"], w - 0.02, label="Sumber asing", color="#f47230")
        for xi, (iv, fv) in enumerate(zip(sub["in_source_median_dice"], sub["foreign_median_dice"])):
            ax.annotate(f"Δ={iv - fv:.3f}", xy=(xi, max(iv, fv) + 0.012), ha="center", fontsize=8)
        ax.set_xticks(x, ARMS)
        ax.set_title(f"Lipatan {SRC[held_out_source(fold)]}")
        ax.set_xlabel("Lengan (kode)")
        ax.set_ylim(0.0, 1.12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Median koefisien Dice (tanpa satuan)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_mode_main(pred_dir: Path, exp2_dir: Path, out: Path, processed_dir: Path, huda_dir: Path, main_tables: Path,
                  cfg: dict, prog: Progress) -> dict[str, pd.DataFrame]:
    """Skenario B: Percobaan 2 sebagai hasil utama. Same rules as the main analysis, arms instead of scenarios."""
    mode = MODES["main"]
    AN._configure_matplotlib()
    tab_dir, fig_dir = out / "tables", out / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(processed_dir)
    gt_cache: dict[str, np.ndarray] = {}

    prog("B2", "validasi berkas")
    t0 = validate_inputs(pred_dir, exp2_dir, manifest, cfg)
    t0.to_csv(tab_dir / "T0_validation.csv", index=False)
    if not t0["valid"].all():
        raise SystemExit(f"{int((~t0['valid']).sum())} berkas tidak valid, lihat {tab_dir / 'T0_validation.csv'}")

    prog("B2", "evaluasi 27 run (E0, E1, E2)")
    raw = evaluate_arms(pred_dir, exp2_dir, manifest, processed_dir, cfg, out / "metrics", gt_cache)
    avg = average_over_seeds(raw, failure_dice=float(cfg["metrics"]["failure_dice"]))

    prog("B2", "T1 celah, T2 lengan vs E0 (Holm 6 uji), T3 klinis")
    t1 = AN.build_t1_gap(avg, {**cfg, "scenarios": ARMS})
    t2 = build_comparisons(avg, cfg, mode["tested"])
    t3 = build_clinical(avg, cfg)

    prog("B2", "T5-T7 alarm A1/A2 pada E0, E1, E2")
    scores, taus = compute_shift_records(pred_dir, exp2_dir, manifest, processed_dir, avg, cfg, mode["alarm_arms"])
    t5 = build_alarm_table(scores, taus, huda_dir, mode["alarm_arms"])
    t6 = build_readiness_table(avg, scores, taus, huda_dir, cfg, mode["alarm_arms"])
    t7 = AN.build_t7_sample50(scores, taus, cfg)

    prog("B2", "T8 few-shot, T9 verifikasi")
    t8 = build_fewshot(raw, avg, exp2_dir, manifest, processed_dir, cfg, gt_cache)
    t9 = verify_against_huda(avg, t2, t8, t5, huda_dir, main_tables, cfg)

    tables = {"T1_gap": t1, "T2_arms": t2, "T3_clinical": t3, "T5_alarm": t5, "T6_readiness": t6,
              "T7_sample50": t7, "T8_fewshot": t8, "T9_verify": t9}
    for name, df in tables.items():
        df.to_csv(tab_dir / f"{name}.csv", index=False)
    scores.to_csv(tab_dir / "alarm_scores.csv", index=False)

    prog("B3", "gambar F1-F5")
    plot_dice_box(avg, cfg, fig_dir / "F1_dice_box.png", "Sebaran Dice pada Sumber Asing per Lengan")
    plot_gap(t1, cfg, fig_dir / "F2_gap.png", "Celah Generalisasi per Lengan: Median Dice Dalam Sumber vs Sumber Asing")
    plot_alarm_scatter(scores, taus, cfg, "E2", fig_dir / "F3_alarm_e2.png",
                       "Skor Alarm vs Dice pada E2, Sumber Asing (merah = gagal)")
    plot_fewshot(t8, cfg, fig_dir / "F4_fewshot.png", "Adaptasi Few-shot E2 (garis: rata-rata 3 seed, pita: rentang seed)")
    plot_worst(avg, pred_dir, exp2_dir, manifest, processed_dir, cfg, "E2", fig_dir / "F5_worst_e2.png",
               "Tiga Kasus Terburuk E2 per Sumber Asing (hijau = mask asli, merah = prediksi)")
    n_bad = int((~t9["match"]).sum()) if not t9.empty else 0
    prog.done(f"mode main selesai: {len(tables)} tabel, 5 gambar, verifikasi {len(t9) - n_bad}/{len(t9)} cocok")
    return {"T0_validation": t0, **tables}


def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis Percobaan 2 (post-hoc).")
    ap.add_argument("--mode", choices=sorted(MODES), default="a")
    ap.add_argument("--pred-dir", required=True, help="Prediksi percobaan utama ({fold}/S0/seed{s}/pred.npz) = E0")
    ap.add_argument("--exp2-dir", required=True, help="Folder Percobaan 2 (E1/, E2/, fewshot/)")
    ap.add_argument("--out", default="results/exp2/nabil")
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--huda-tables", default="results/exp2/tables", help="Tabel Huda untuk verifikasi")
    ap.add_argument("--main-tables", default="results/tables", help="Tabel hasil utama (T5) untuk verifikasi E0")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-progress", action="store_true", help="Jangan tulis progress/log.txt")
    a = ap.parse_args()

    cfg = load_config(a.config) if a.config else load_config()

    def absolute(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    processed = absolute(a.processed_dir or cfg["data"]["processed_dir"])
    out = absolute(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prog = Progress("B" if a.mode == "main" else a.mode.upper(), enabled=not a.no_progress)
    try:
        runner = run_mode_main if a.mode == "main" else run_mode_a
        runner(absolute(a.pred_dir), absolute(a.exp2_dir), out, processed, absolute(a.huda_tables),
                   absolute(a.main_tables), cfg, prog)
    except BaseException as e:
        if prog.enabled:
            PG.write_log(prog.track, "GAGAL", f"{type(e).__name__}: {e}")
            PG.RUNNING.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
