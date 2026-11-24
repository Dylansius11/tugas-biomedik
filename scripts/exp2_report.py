"""Tables, explainable-alarm results and figures for Percobaan 2. Exploratory: not the official analysis tables.

Writes results/exp2/tables/*.csv, results/exp2/figures/*.png and results/exp2/summary.txt.
Usage: python scripts/exp2_report.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg import exp2  # noqa: E402
from cxrseg.config import ROOT, load_config  # noqa: E402
from cxrseg.data import fold_split, held_out_source, load_arrays, load_manifest  # noqa: E402
from cxrseg.explain import FACTORS, Reference, features, hist, source_verdict, tta_probs  # noqa: E402
from cxrseg.model import build_model  # noqa: E402
from cxrseg.postprocess import postprocess_mask  # noqa: E402

cfg = load_config()
R = exp2.RESULTS
TAB, FIG = R / "tables", R / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEEDS, FOLDS = cfg["train"]["seeds"], cfg["folds"]
FAIL = cfg["metrics"]["failure_dice"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FINAL = (R / "final_arm.txt").read_text().split()[0]
NAME = {"ho_darwin": "Darwin", "ho_montgomery": "Montgomery", "ho_shenzhen": "Shenzhen"}
man = load_manifest(ROOT / cfg["data"]["processed_dir"])
_, M = load_arrays(man, ROOT / cfg["data"]["processed_dir"])
GT = {i: m > 0 for i, m in zip(man["id"], M)}
out_lines: list[str] = []


def say(s: str = "") -> None:
    print(s)
    out_lines.append(s)


def pred_path(arm: str, fold: str, seed: int) -> Path:
    return ROOT / "results" / "predictions" / fold / "S0" / f"seed{seed}" / "pred.npz" if arm == "E0" else R / arm / fold / f"seed{seed}" / "pred.npz"


def scores(prob_u8: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    p = postprocess_mask(prob_u8)
    inter = (p & gt).sum()
    return 2 * inter / max(p.sum() + gt.sum(), 1), inter / max(gt.sum(), 1), inter / max(p.sum(), 1)


# ---------------------------------------------------------------- 1. per-image metrics
rows = []
for arm in ["E0", "E1", "E2"]:
    for fold in FOLDS:
        for seed in SEEDS:
            z = np.load(pred_path(arm, fold, seed))
            for i, g, p in zip(z["ids"], z["group"], z["prob_u8"]):
                d, c, pr = scores(p, GT[str(i)])
                rows.append((arm, fold, seed, str(i), str(g), d, c, pr))
df = pd.DataFrame(rows, columns="arm fold seed id group dice coverage precision".split())
a = df.groupby(["arm", "fold", "group", "id"], as_index=False)[["dice", "coverage", "precision"]].mean()
a.to_csv(TAB / "per_image_seedmean.csv", index=False)

t = []
for (arm, fold), g in a.groupby(["arm", "fold"]):
    ins, fo = g[g.group == "in_source"], g[g.group == "foreign"]
    t.append(dict(arm=arm, fold=fold, in_median=ins.dice.median(), foreign_median=fo.dice.median(), foreign_mean=fo.dice.mean(),
                  foreign_fail_pct=(fo.dice < FAIL).mean() * 100, coverage=fo.coverage.median(), precision=fo.precision.median()))
T = pd.DataFrame(t)
T.to_csv(TAB / "main.csv", index=False)
val = {arm: np.mean([float((R / arm / f / f"seed{s}" / "val_dice.txt").read_text()) for f in FOLDS for s in SEEDS]) for arm in ["E1", "E2"]}
say(f"final arm (picked on validation only): {FINAL}   mean val dice {val}")
say("\n== Foreign (held-out source) results, per-image Dice averaged over 3 seeds ==")
say(T.round(4).to_string(index=False))

say("\n== Paired comparison vs E0 on the foreign source (Wilcoxon, bootstrap 95% CI of mean difference) ==")
rng = np.random.default_rng(0)
cmp_rows = []
for arm in ["E1", "E2"]:
    for fold in FOLDS:
        p = a[(a.fold == fold) & (a.group == "foreign")].pivot(index="id", columns="arm", values="dice")
        d = (p[arm] - p["E0"]).values
        try:
            pv = wilcoxon(d).pvalue
        except ValueError:
            pv = 1.0
        lo, hi = np.percentile(rng.choice(d, (cfg["stats"]["bootstrap"], len(d))).mean(1), [2.5, 97.5])
        cmp_rows.append(dict(arm=arm, fold=fold, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, p=pv))
C = pd.DataFrame(cmp_rows)
C.to_csv(TAB / "vs_E0.csv", index=False)
say(C.round(4).to_string(index=False))

# ---------------------------------------------------------------- 2. few-shot
fs_rows = []
for fold in FOLDS:
    _, evalrows = exp2.fewshot_split(cfg, fold)
    ev = set(evalrows["id"])
    for arm in ["E0", FINAL]:
        g = a[(a.arm == arm) & (a.fold == fold) & (a.group == "foreign") & a.id.isin(ev)]
        fs_rows.append(dict(fold=fold, k=0, model=arm, median=g.dice.median(), fail_pct=(g.dice < FAIL).mean() * 100))
    for k in cfg["exp2"]["fewshot"]["k"]:
        per = []
        for seed in SEEDS:
            z = np.load(R / "fewshot" / FINAL / fold / f"k{k}_seed{seed}.npz")
            per.append(pd.Series([scores(p, GT[str(i)])[0] for i, p in zip(z["ids"], z["prob_u8"])], index=z["ids"]))
        d = pd.concat(per, axis=1).mean(1)
        fs_rows.append(dict(fold=fold, k=k, model=f"{FINAL}+k", median=d.median(), fail_pct=(d < FAIL).mean() * 100))
FS = pd.DataFrame(fs_rows)
FS.to_csv(TAB / "fewshot.csv", index=False)
say("\n== Few-shot: k labelled images of the new source (evaluated on its val+test split, never adapted on) ==")
say(FS.round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. explainable alarm (A3) with uncertainty maps
ex = cfg["exp2"]["explain"]


def predictor(arm: str, fold: str, seed: int):
    if arm == "E0":
        m = build_model(cfg).to(DEV)
        m.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / fold / "S0" / f"seed{seed}" / "best.pt", map_location=DEV))
        m.eval()

        def f(x, heads=None):
            with torch.no_grad():
                return np.concatenate([torch.sigmoid(m(torch.from_numpy(x[i:i + 16])[:, None].to(DEV)).float())[:, 0].cpu().numpy() for i in range(0, len(x), 16)])
        return f
    m = exp2.build(cfg).to(DEV)
    m.load_state_dict(torch.load(R / arm / fold / f"seed{seed}" / "best.pt", map_location=DEV))
    return lambda x, heads: exp2.probs(m, x, heads, DEV)


xai_rows, verdicts, factor_rows, cards = [], [], [], {}
dice_map = a.set_index(["arm", "fold", "group", "id"]).dice
for arm in ["E0", FINAL]:
    for fold in FOLDS:
        split = fold_split(man, fold)
        data = {}
        for key in ("train", "val", "in_source_test", "foreign"):
            ids, x, _, h = exp2.load_rows(cfg, split[key])
            data[key] = (ids, x, h)
        ref_hist = hist(np.rint(data["train"][1] * 255).astype(np.uint8))
        for seed in SEEDS:
            f = predictor(arm, fold, seed)
            th = int(np.load(pred_path(arm, fold, seed))["head"][-1]) if arm != "E0" else 0
            feats = {}
            for key in ("val", "in_source_test", "foreign"):
                ids, x, h = data[key]
                heads = np.full(len(ids), th) if key == "foreign" else h
                plain = f(x, heads)
                _, std = tta_probs(lambda xi: f(xi, heads), x, ex["tta_shift_px"], ex["tta_scales"])
                x8 = np.rint(x * 255).astype(np.uint8)
                feats[key] = [features(x8[i], plain[i], std[i], ref_hist, ex["uncertain_std"]) for i in range(len(ids))]
                if seed == SEEDS[0] and key == "foreign":
                    cards[(arm, fold)] = (ids, x, plain, std, feats[key])
            ref = Reference(feats["val"], ex["ref_percentile"])
            ids_f = data["foreign"][0]
            d = dice_map.loc[(arm, fold, "foreign")].reindex(ids_f).values
            fail = d < FAIL
            zf = np.abs(ref.z(feats["foreign"]))
            row = dict(arm=arm, fold=fold, seed=seed, tau=ref.tau, in_alarm_pct=(ref.a3(feats["in_source_test"]) > ref.tau).mean() * 100,
                       foreign_alarm_pct=(zf.max(1) > ref.tau).mean() * 100, foreign_fail_pct=fail.mean() * 100,
                       caught_fail_pct=((zf.max(1) > ref.tau) & fail).sum() / max(fail.sum(), 1) * 100)
            for name, s in (("A1", zf[:, 2]), ("A2", zf[:, 7]), ("A3", zf.max(1))):
                row[f"auroc_{name}"] = roc_auc_score(fail, s) if 10 <= fail.sum() <= len(fail) - 10 else np.nan
            xai_rows.append(row)
            for j, (k, *_rest) in enumerate(FACTORS):
                factor_rows.append(dict(arm=arm, fold=fold, seed=seed, factor=k, flagged_pct=(zf[:, j] > ref.tau).mean() * 100,
                                        flagged_in_failed_pct=(zf[fail, j] > ref.tau).mean() * 100 if fail.any() else np.nan,
                                        auroc=roc_auc_score(fail, zf[:, j]) if 10 <= fail.sum() <= len(fail) - 10 else np.nan))
            if seed == SEEDS[0]:
                v = source_verdict(ref, feats["foreign"], cfg["readiness"]["max_alarm_rate"])
                truth = fail.mean() <= cfg["readiness"]["max_failure_rate"]
                verdicts.append(dict(arm=arm, fold=fold, alarm_rate=v["alarm_rate"], says_ready=v["ready"], truly_ready=bool(truth), reasons=" | ".join(v["reasons"])))
                cards[(arm, fold)] += (ref,)
X = pd.DataFrame(xai_rows)
X.to_csv(TAB / "xai_alarm.csv", index=False)
F = pd.DataFrame(factor_rows)
F.to_csv(TAB / "xai_factors.csv", index=False)
V = pd.DataFrame(verdicts)
V.to_csv(TAB / "xai_verdicts.csv", index=False)
say("\n== Explainable alarm A3 vs A1/A2 (mean over seeds; tau from validation only) ==")
say(X.groupby(["arm", "fold"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
say("\n== Which factors fire on the foreign source (mean over seeds) ==")
say(F.groupby(["arm", "fold", "factor"]).mean(numeric_only=True).drop(columns="seed").round(2).to_string())
say("\n== Source-level verdict with reasons (seed 11) ==")
for r in V.itertuples():
    say(f"{r.arm} {NAME[r.fold]:10s} alarm {r.alarm_rate * 100:5.1f}%  says {'SIAP' if r.says_ready else 'BELUM SIAP'}  truth {'SIAP' if r.truly_ready else 'BELUM SIAP'}")
    for reason in r.reasons.split(" | "):
        if reason:
            say(f"      - {reason}")

# ---------------------------------------------------------------- 4. figures
# 4a convention: one example per source, ground truth only
fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
for c, src in enumerate(["darwin", "montgomery", "shenzhen"]):
    r = man[(man.source == src) & (man.split == "test")].iloc[3:4]
    img, msk = load_arrays(r, ROOT / cfg["data"]["processed_dir"])
    ax[c].imshow(img[0], cmap="gray")
    ax[c].contour(msk[0] > 0, [0.5], colors="lime", linewidths=1.5)
    ax[c].set_title(f"{src.capitalize()}: paru {cfg['exp2']['convention'][src]}", fontsize=10)
    ax[c].axis("off")
fig.suptitle("Tanda paru (hijau) dari pembuat dataset: Darwin ikut menandai paru di belakang jantung dan diafragma", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "convention.png", dpi=120)
plt.close()

# 4b foreign failure rate per arm
fig, ax = plt.subplots(figsize=(7, 3.8))
arms = ["E0", "E1", "E2"]
for j, arm in enumerate(arms):
    vals = [T[(T.arm == arm) & (T.fold == f)].foreign_fail_pct.iloc[0] for f in FOLDS]
    b = ax.bar(np.arange(3) + (j - 1) * 0.27, vals, 0.27, label={"E0": "E0 baseline", "E1": "E1 head per konvensi", "E2": "E2 + augmentasi kuat"}[arm])
    ax.bar_label(b, fmt="%.0f%%", fontsize=8)
ax.set_xticks(range(3), [NAME[f] for f in FOLDS])
ax.set_ylabel("% foto gagal (Dice < 0.90)")
ax.set_title("Sumber asing: persentase foto gagal")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(FIG / "foreign_fail.png", dpi=120)
plt.close()

# 4c few-shot curve
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for fold in FOLDS:
    g = FS[(FS.fold == fold) & (FS.model != "E0")].sort_values("k")
    ax.plot(g.k, g.fail_pct, marker="o", label=NAME[fold])
ax.set_xscale("symlog", linthresh=5)
ax.set_xticks([0, 5, 10, 20, 50], ["0", "5", "10", "20", "50"])
ax.set_xlabel("jumlah foto berlabel dari sumber baru (k)")
ax.set_ylabel("% foto gagal")
ax.set_title(f"Adaptasi few-shot ({FINAL})")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG / "fewshot.png", dpi=120)
plt.close()

# 4d explanation cards: worst and typical foreign image for the baseline on Darwin and Shenzhen, and the final arm
for arm in ["E0", FINAL]:
    fig, ax = plt.subplots(3, 4, figsize=(15, 11), gridspec_kw={"height_ratios": [1, 1, 0.55]})
    for c, fold in enumerate(["ho_darwin", "ho_shenzhen"]):
        ids, x, plain, std, feats, ref = cards[(arm, fold)]
        d = dice_map.loc[(arm, fold, "foreign")].reindex(ids).values
        for cc, i in enumerate([int(np.argsort(d)[len(d) // 10]), int(np.argsort(d)[len(d) // 2])]):
            col = 2 * c + cc
            ax[0, col].imshow(x[i], cmap="gray")
            ax[0, col].contour(GT[ids[i]], [0.5], colors="lime", linewidths=1)
            ax[0, col].contour(postprocess_mask(plain[i]), [0.5], colors="red", linewidths=1)
            ax[0, col].set_title(f"{NAME[fold]} {ids[i]}\nDice {d[i]:.3f}  (hijau=tanda, merah=model)", fontsize=9)
            ax[1, col].imshow(x[i], cmap="gray")
            im = ax[1, col].imshow(std[i], cmap="inferno", alpha=0.65, vmin=0, vmax=0.35)
            ax[1, col].set_title("peta ketidakpastian (goyah saat digeser/zoom)", fontsize=9)
            reasons = ref.reasons(feats[i])
            verdict = "ALARM: BELUM SIAP" if reasons else "tidak ada alarm"
            txt = verdict + "\n" + "\n".join(textwrap.fill("- " + r, 48) for r in reasons[:4])
            ax[2, col].text(0, 1, txt, va="top", fontsize=8, family="monospace")
            for r_ in range(3):
                ax[r_, col].axis("off")
    fig.colorbar(im, ax=ax[1, :], shrink=0.6, label="std probabilitas")
    fig.suptitle(f"Kartu penjelasan alarm ({arm}): foto persentil-10 dan median per sumber", fontsize=11)
    plt.savefig(FIG / f"xai_cards_{arm}.png", dpi=100, bbox_inches="tight")
    plt.close()

(R / "summary.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"\nwrote {TAB}, {FIG}, {R / 'summary.txt'}")
