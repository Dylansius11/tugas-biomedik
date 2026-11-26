"""End-to-end official analysis script producing tables T1..T7 and figures F1..F6.

Follows docs/plans/BUILD-PLAN-NABIL.md (N5) and docs/PROTOCOL.md sections 7, 8, 9.

Usage:
    python scripts/analyze.py --pred-dir results/fake_predictions --out results/fake_analysis
    python scripts/analyze.py --pred-dir results/predictions --out results
"""
from __future__ import annotations

import argparse
import ast
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
from cxrseg.data import fold_split, held_out_source, load_arrays, load_manifest  # noqa: E402
from cxrseg.evaluate import evaluate_all  # noqa: E402
from cxrseg.postprocess import postprocess_mask  # noqa: E402
from cxrseg.preprocess import DEFAULT_CLAHE, ClaheParams, enhance  # noqa: E402
from cxrseg.shift import (  # noqa: E402
    alarm_rate,
    alarm_threshold,
    compute_alarm_evaluation,
    decision_from_alarm,
    intensity_distance,
    mean_entropy,
    pooled_hist,
    readiness_label,
)
from cxrseg.stats import (  # noqa: E402
    average_over_seeds,
    bootstrap_mean_diff_ci,
    holm,
    is_meaningful,
    paired_comparison,
)

SOURCE_TITLE = {"darwin": "Darwin", "montgomery": "Montgomery", "shenzhen": "Shenzhen"}
SCEN_COLORS = {"S0": "#4C72B0", "S1": "#DD8452", "S2": "#55A868", "S3": "#C44E52"}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )


def _parse_tile_grid(value: str) -> tuple[int, int]:
    s = str(value).strip()
    if s.startswith(("(", "[")):
        tup = ast.literal_eval(s)
        return (int(tup[0]), int(tup[1]))
    parts = s.lower().split("x")
    return (int(parts[0]), int(parts[1]))


def _resolve_s3_params_path(tables_dir: Path) -> Path:
    candidate = tables_dir / "s3_params.csv"
    if candidate.exists():
        return candidate
    return ROOT / "results" / "tables" / "s3_params.csv"


def _load_s3_params(fold: str, path: Path) -> ClaheParams:
    df = pd.read_csv(path)
    row = df[df["fold"] == fold]
    if row.empty:
        raise ValueError(f"fold {fold!r} missing in {path}")
    r = row.iloc[0]
    return ClaheParams(float(r["clip_limit"]), _parse_tile_grid(str(r["tile_grid"])))


def _get_clahe_for_scenario(fold: str, scenario: str, s3_path: Path) -> ClaheParams | None:
    if scenario != "S3":
        return None
    if s3_path.exists():
        try:
            return _load_s3_params(fold, s3_path)
        except ValueError:
            return DEFAULT_CLAHE
    return DEFAULT_CLAHE


def build_t1_gap(avg_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T1_gap.csv: Per fold x scenario generalization gap (answers RM1)."""
    rows: list[dict] = []
    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        for scenario in cfg["scenarios"]:
            sub = avg_df[(avg_df["fold"] == fold) & (avg_df["scenario"] == scenario)]
            if sub.empty:
                continue
            ins = sub[sub["group"] == "in_source"]
            fo = sub[sub["group"] == "foreign"]
            if ins.empty or fo.empty:
                continue
            in_med = float(ins["dice"].median())
            fo_med = float(fo["dice"].median())
            rows.append(
                {
                    "fold": fold,
                    "scenario": scenario,
                    "foreign_source": ho,
                    "in_source_median_dice": in_med,
                    "foreign_median_dice": fo_med,
                    "generalization_gap": in_med - fo_med,
                    "in_source_mean_dice": float(ins["dice"].mean()),
                    "foreign_mean_dice": float(fo["dice"].mean()),
                    "foreign_failure_rate_pct": float(fo["failed"].mean() * 100.0),
                    "foreign_median_coverage": float(fo["coverage"].median()),
                    "foreign_median_area_error_pct": float(fo["area_error_pct"].median()),
                    "foreign_median_hd95": float(fo["hd95"].median()),
                }
            )
    return pd.DataFrame(rows)


def build_t2_scenarios(avg_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T2_scenarios.csv: Paired comparison S1/S2/S3 vs S0 on foreign sources with Holm correction (RM2)."""
    scfg = cfg["stats"]
    n_boot = int(scfg["bootstrap"])
    alpha = float(scfg["alpha"])
    min_eff = float(scfg["min_effect_dice"])
    boot_seed = int(cfg["data"]["sample_seed"])

    rows: list[dict] = []
    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        for scenario in [s for s in cfg["scenarios"] if s != "S0"]:
            sub = avg_df[(avg_df["source"] == ho) & (avg_df["group"] == "foreign")]
            if sub[sub["scenario"] == "S0"].empty or sub[sub["scenario"] == scenario].empty:
                continue
            res = paired_comparison(
                avg_df,
                baseline="S0",
                scenario=scenario,
                source=ho,
                metric="dice",
                n_boot=n_boot,
                seed=boot_seed,
                alpha=alpha,
                run_wilcoxon=True,
            )
            rows.append(
                {
                    "fold": fold,
                    "foreign_source": ho,
                    "scenario": scenario,
                    "baseline": "S0",
                    "n_images": res["n_images"],
                    "mean_diff_dice": res["mean_diff"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "p_value": res["p_value"],
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_holm"] = holm(df["p_value"].tolist())
        df["meaningful"] = [
            is_meaningful(ph, md, alpha=alpha, min_effect=min_eff)
            for ph, md in zip(df["p_holm"], df["mean_diff_dice"])
        ]
    return df


def build_t3_clinical(avg_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T3_clinical.csv: Failure rate (0.90 and sensitivity 0.85, 0.93), coverage, area error (RM2)."""
    scfg = cfg["stats"]
    n_boot = int(scfg["bootstrap"])
    alpha = float(scfg["alpha"])
    boot_seed = int(cfg["data"]["sample_seed"])
    sens_thresholds = [float(t) for t in cfg["metrics"]["failure_sensitivity"]]
    main_thr = float(cfg["metrics"]["failure_dice"])

    rows: list[dict] = []
    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        for scenario in cfg["scenarios"]:
            fo = avg_df[
                (avg_df["fold"] == fold)
                & (avg_df["scenario"] == scenario)
                & (avg_df["group"] == "foreign")
            ]
            if fo.empty:
                continue

            row = {
                "fold": fold,
                "foreign_source": ho,
                "scenario": scenario,
                "n_images": int(len(fo)),
                f"failure_rate_{int(round(main_thr * 100)):03d}_pct": float((fo["dice"] < main_thr).mean() * 100.0),
            }
            for thr in sens_thresholds:
                row[f"failure_rate_{int(round(thr * 100)):03d}_pct"] = float((fo["dice"] < thr).mean() * 100.0)

            row["coverage_mean"] = float(fo["coverage"].mean())
            row["coverage_median"] = float(fo["coverage"].median())
            row["median_area_error_pct"] = float(fo["area_error_pct"].median())
            row["median_abs_area_error_pct"] = float(fo["area_error_pct"].abs().median())

            if scenario == "S0":
                row["failure_diff_vs_s0_pct"] = 0.0
                row["failure_diff_ci_low_pct"] = 0.0
                row["failure_diff_ci_high_pct"] = 0.0
                row["coverage_diff_vs_s0"] = 0.0
                row["coverage_diff_ci_low"] = 0.0
                row["coverage_diff_ci_high"] = 0.0
            else:
                f_cmp = paired_comparison(
                    avg_df,
                    baseline="S0",
                    scenario=scenario,
                    source=ho,
                    metric="failed",
                    n_boot=n_boot,
                    seed=boot_seed,
                    alpha=alpha,
                    run_wilcoxon=False,
                )
                c_cmp = paired_comparison(
                    avg_df,
                    baseline="S0",
                    scenario=scenario,
                    source=ho,
                    metric="coverage",
                    n_boot=n_boot,
                    seed=boot_seed,
                    alpha=alpha,
                    run_wilcoxon=False,
                )
                row["failure_diff_vs_s0_pct"] = f_cmp["mean_diff"] * 100.0
                row["failure_diff_ci_low_pct"] = f_cmp["ci_low"] * 100.0
                row["failure_diff_ci_high_pct"] = f_cmp["ci_high"] * 100.0
                row["coverage_diff_vs_s0"] = c_cmp["mean_diff"]
                row["coverage_diff_ci_low"] = c_cmp["ci_low"]
                row["coverage_diff_ci_high"] = c_cmp["ci_high"]

            rows.append(row)
    return pd.DataFrame(rows)


def build_t4_tuning(
    manifest: pd.DataFrame,
    processed_dir: Path,
    tables_dir: Path,
    cfg: dict,
) -> pd.DataFrame:
    """T4_tuning.csv: 12 CLAHE combinations per fold + inter-source intensity distance."""
    rows: list[dict] = []
    for fold in cfg["folds"]:
        tpath = tables_dir / f"tuning_{fold}.csv"
        if not tpath.exists():
            tpath = ROOT / "results" / "tables" / f"tuning_{fold}.csv"
        if not tpath.exists():
            continue

        tdf = pd.read_csv(tpath)
        winner_idx = int(tdf["val_dice"].idxmax())

        split = fold_split(manifest, fold)
        if split["train"].empty or split["foreign"].empty:
            continue
        train_imgs, _ = load_arrays(split["train"], processed_dir)
        foreign_imgs, _ = load_arrays(split["foreign"], processed_dir)

        for idx, r in tdf.iterrows():
            clip = float(r["clip_limit"])
            tile = _parse_tile_grid(r["tile_grid"])
            clahe = ClaheParams(clip, tile)

            train_enh = np.stack([enhance(im, "S3", clahe) for im in train_imgs])
            foreign_enh = np.stack([enhance(im, "S3", clahe) for im in foreign_imgs])
            ref_h = pooled_hist(train_enh)
            foreign_h = pooled_hist(foreign_enh)

            mean_a1 = float(np.mean([intensity_distance(im, ref_h) for im in foreign_enh]))
            pooled_w1 = float(np.abs(np.cumsum(foreign_h) - np.cumsum(ref_h)).sum() / 255.0)

            rows.append(
                {
                    "fold": fold,
                    "clip_limit": clip,
                    "tile_grid": str(r["tile_grid"]),
                    "val_dice": float(r["val_dice"]),
                    "is_winner": bool(idx == winner_idx),
                    "mean_foreign_a1": mean_a1,
                    "pooled_hist_dist": pooled_w1,
                }
            )
    return pd.DataFrame(rows)


def compute_shift_records(
    pred_dir: Path,
    manifest: pd.DataFrame,
    processed_dir: Path,
    s3_params_path: Path,
    avg_df: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-image A1 (after scenario) and seed-averaged A2 scores, plus per-(fold,scenario,alarm) tau.

    Strict anti-leakage guarantee:
      - Reference histogram for A1 is pooled strictly from `train` images of the 2 training sources.
      - Threshold tau is the 95th percentile of `val` scores from the 2 training sources only.
    """
    perc = float(cfg["readiness"]["alarm_percentile"])
    score_rows: list[dict] = []
    tau_rows: list[dict] = []

    for fold in cfg["folds"]:
        ho = held_out_source(fold)
        split = fold_split(manifest, fold)

        train_imgs, _ = load_arrays(split["train"], processed_dir)
        val_imgs, _ = load_arrays(split["val"], processed_dir)
        ins_imgs, _ = load_arrays(split["in_source_test"], processed_dir)
        fo_imgs, _ = load_arrays(split["foreign"], processed_dir)

        val_ids = split["val"]["id"].astype(str).tolist()
        ins_ids = split["in_source_test"]["id"].astype(str).tolist()
        fo_ids = split["foreign"]["id"].astype(str).tolist()

        for scenario in cfg["scenarios"]:
            scen_dir = pred_dir / fold / scenario
            if not scen_dir.exists():
                continue
            seed_dirs = sorted(
                (p for p in scen_dir.iterdir() if p.is_dir() and p.name.startswith("seed")),
                key=lambda p: int(p.name.removeprefix("seed")),
            )
            if not seed_dirs:
                continue

            clahe = _get_clahe_for_scenario(fold, scenario, s3_params_path)

            # A1: enhanced train pooled reference histogram -> A1 on val, in_source, foreign
            train_enh = np.stack([enhance(im, scenario, clahe) for im in train_imgs])
            ref_hist = pooled_hist(train_enh)

            a1_by_group_id: dict[tuple[str, str], float] = {}
            for group_name, ids_list, raw_imgs in (
                ("val", val_ids, val_imgs),
                ("in_source", ins_ids, ins_imgs),
                ("foreign", fo_ids, fo_imgs),
            ):
                for img_id, im in zip(ids_list, raw_imgs):
                    enh = enhance(im, scenario, clahe)
                    a1_by_group_id[(group_name, img_id)] = intensity_distance(enh, ref_hist)

            # A2: average per-seed mean_entropy per (group, id)
            a2_acc: dict[tuple[str, str], list[float]] = {}
            for sdir in seed_dirs:
                npz_path = sdir / "pred.npz"
                if not npz_path.exists():
                    continue
                with np.load(npz_path, allow_pickle=True) as data:
                    for img_id_raw, grp_raw, prob in zip(data["ids"], data["group"], data["prob_u8"]):
                        key = (str(grp_raw), str(img_id_raw))
                        a2_acc.setdefault(key, []).append(mean_entropy(prob))

            if not a2_acc:
                continue

            val_a1 = [a1_by_group_id[("val", i)] for i in val_ids]
            val_a2 = [float(np.mean(a2_acc[("val", i)])) for i in val_ids if ("val", i) in a2_acc]
            if not val_a2:
                continue

            tau_a1 = alarm_threshold(val_a1, perc)
            tau_a2 = alarm_threshold(val_a2, perc)

            tau_rows.append({"fold": fold, "foreign_source": ho, "scenario": scenario, "alarm": "A1", "tau": tau_a1})
            tau_rows.append({"fold": fold, "foreign_source": ho, "scenario": scenario, "alarm": "A2", "tau": tau_a2})

            eval_sub = avg_df[(avg_df["fold"] == fold) & (avg_df["scenario"] == scenario)]
            eval_map = eval_sub.set_index(["group", "id"])

            for group_name, ids_list in (("val", val_ids), ("in_source", ins_ids), ("foreign", fo_ids)):
                for img_id in ids_list:
                    key = (group_name, img_id)
                    if key not in a2_acc:
                        continue
                    a1_val = a1_by_group_id[key]
                    a2_val = float(np.mean(a2_acc[key]))
                    if key in eval_map.index:
                        dice_val = float(eval_map.loc[key, "dice"])
                        fail_val = bool(eval_map.loc[key, "failed"])
                    else:
                        dice_val = float("nan")
                        fail_val = False

                    score_rows.append(
                        {
                            "fold": fold,
                            "foreign_source": ho,
                            "scenario": scenario,
                            "group": group_name,
                            "id": img_id,
                            "A1": a1_val,
                            "A2": a2_val,
                            "dice": dice_val,
                            "failed": fail_val,
                        }
                    )

    return pd.DataFrame(score_rows), pd.DataFrame(tau_rows)


def build_t5_alarm(scores_df: pd.DataFrame, tau_df: pd.DataFrame) -> pd.DataFrame:
    """T5_alarm.csv: Spearman, AUROC (or 'tidak dapat dihitung'), and alarm rate per scenario & foreign source."""
    rows: list[dict] = []
    for t in tau_df.itertuples():
        alarm = t.alarm
        sub = scores_df[(scores_df["fold"] == t.fold) & (scores_df["scenario"] == t.scenario)]
        fo = sub[sub["group"] == "foreign"]
        ins = sub[sub["group"] == "in_source"]
        if fo.empty:
            continue
        ev = compute_alarm_evaluation(fo[alarm].values, fo["dice"].values, fo["failed"].values)
        fo_rate = alarm_rate(fo[alarm].values, t.tau) * 100.0
        ins_rate = alarm_rate(ins[alarm].values, t.tau) * 100.0 if not ins.empty else float("nan")
        rows.append(
            {
                "fold": t.fold,
                "foreign_source": t.foreign_source,
                "scenario": t.scenario,
                "alarm": alarm,
                "tau": float(t.tau),
                "n_foreign": ev["n_images"],
                "n_failed": ev["n_failed"],
                "alarm_rate_pct": fo_rate,
                "in_source_alarm_rate_pct": ins_rate,
                "spearman_rho": ev["spearman_rho"],
                "spearman_p": ev["spearman_p"],
                "auroc": ev["auroc"],
            }
        )
    return pd.DataFrame(rows)


def build_t6_readiness(
    avg_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """T6_readiness.csv: Label-based 'siap' vs alarm decision across 12 foreign + 12 in-source cases."""
    rows: list[dict] = []
    for t in tau_df.itertuples():
        alarm = t.alarm
        eval_sub = avg_df[(avg_df["fold"] == t.fold) & (avg_df["scenario"] == t.scenario)]
        ins_eval = eval_sub[eval_sub["group"] == "in_source"]
        if ins_eval.empty:
            continue
        ins_med = float(ins_eval["dice"].median())

        for grp in ("foreign", "in_source"):
            grp_eval = eval_sub[eval_sub["group"] == grp]
            grp_sc = scores_df[
                (scores_df["fold"] == t.fold)
                & (scores_df["scenario"] == t.scenario)
                & (scores_df["group"] == grp)
            ]
            if grp_eval.empty or grp_sc.empty:
                continue
            fail_rate = float(grp_eval["failed"].mean())
            med_dice = float(grp_eval["dice"].median())
            truth = readiness_label(fail_rate, med_dice, ins_med, cfg)
            rate = alarm_rate(grp_sc[alarm].values, t.tau)
            decision = decision_from_alarm(rate, cfg)
            rows.append(
                {
                    "alarm": alarm,
                    "fold": t.fold,
                    "foreign_source": t.foreign_source,
                    "scenario": t.scenario,
                    "group": grp,
                    "n_images": int(len(grp_eval)),
                    "failure_rate_pct": fail_rate * 100.0,
                    "median_dice": med_dice,
                    "in_source_median_dice": ins_med,
                    "dice_drop": ins_med - med_dice,
                    "truth_label": truth,
                    "alarm_rate_pct": rate * 100.0,
                    "alarm_decision": decision,
                    "match": bool(truth == decision),
                }
            )
    return pd.DataFrame(rows)


def build_t7_sample50(scores_df: pd.DataFrame, tau_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T7_sample50.csv: Distribution of alarm rates from 1,000 random samples of 50 images."""
    n_sample_cfg = int(cfg["stats"]["sample_size_check"])
    n_repeats = int(cfg["stats"]["sample_repeats"])
    seed = int(cfg["data"]["sample_seed"])
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for t in tau_df.itertuples():
        alarm = t.alarm
        fo = scores_df[
            (scores_df["fold"] == t.fold)
            & (scores_df["scenario"] == t.scenario)
            & (scores_df["group"] == "foreign")
        ]
        if fo.empty:
            continue
        vals = fo[alarm].to_numpy(dtype=np.float64)
        k = min(n_sample_cfg, len(vals))
        replace = len(vals) < n_sample_cfg
        full_rate = alarm_rate(vals, t.tau)
        full_dec = decision_from_alarm(full_rate, cfg)

        rates = np.empty(n_repeats, dtype=np.float64)
        agree = 0
        for r_i in range(n_repeats):
            draw = rng.choice(vals, size=k, replace=replace)
            r_rate = float((draw > t.tau).mean())
            rates[r_i] = r_rate * 100.0
            if decision_from_alarm(r_rate, cfg) == full_dec:
                agree += 1

        rows.append(
            {
                "fold": t.fold,
                "foreign_source": t.foreign_source,
                "scenario": t.scenario,
                "alarm": alarm,
                "sample_size": int(k),
                "n_repeats": n_repeats,
                "full_alarm_rate_pct": full_rate * 100.0,
                "sample_mean_pct": float(rates.mean()),
                "sample_std_pct": float(rates.std(ddof=1)) if n_repeats > 1 else 0.0,
                "sample_median_pct": float(np.median(rates)),
                "sample_p05_pct": float(np.percentile(rates, 5)),
                "sample_p95_pct": float(np.percentile(rates, 95)),
                "decision_agreement_pct": float(100.0 * agree / n_repeats),
            }
        )
    return pd.DataFrame(rows)


def plot_f1_dice_by_scenario(avg_df: pd.DataFrame, cfg: dict, fig_dir: Path) -> None:
    """F1_dice_by_scenario.png: Boxplot of per-image Dice by contrast scenario for each foreign source."""
    folds = cfg["folds"]
    scenarios = cfg["scenarios"]
    fail_thr = float(cfg["metrics"]["failure_dice"])
    fig, axes = plt.subplots(1, len(folds), figsize=(4.3 * len(folds), 4.0), sharey=True)
    axes_arr = np.atleast_1d(axes)

    for ax, fold in zip(axes_arr, folds):
        ho = held_out_source(fold)
        sub = avg_df[(avg_df["fold"] == fold) & (avg_df["group"] == "foreign")]
        present_scen = [s for s in scenarios if not sub[sub["scenario"] == s].empty]
        data = [sub[sub["scenario"] == s]["dice"].values for s in present_scen]
        if data:
            bp = ax.boxplot(data, patch_artist=True, showfliers=True, flierprops={"markersize": 3, "alpha": 0.4})
            ax.set_xticks(range(1, len(present_scen) + 1), present_scen)
            for patch, s in zip(bp["boxes"], present_scen):
                patch.set_facecolor(SCEN_COLORS.get(s, "#4C72B0"))
                patch.set_alpha(0.75)
        ax.axhline(fail_thr, color="crimson", linestyle="--", linewidth=1.0, label=f"Ambang gagal ({fail_thr:.2f})")
        ax.set_title(f"Sumber asing: {SOURCE_TITLE[ho]}")
        ax.set_xlabel("Skenario kontras (kode)")
        ax.set_ylim(0.0, 1.03)
        ax.grid(axis="y", alpha=0.25)

    axes_arr[0].set_ylabel("Koefisien Dice (tanpa satuan)")
    axes_arr[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Sebaran Koefisien Dice per Skenario pada Sumber Asing", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "F1_dice_by_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f2_gap(t1_df: pd.DataFrame, cfg: dict, fig_dir: Path) -> None:
    """F2_gap.png: Generalization gap (in-source vs foreign median Dice per fold and scenario)."""
    folds = cfg["folds"]
    scenarios = cfg["scenarios"]
    fig, axes = plt.subplots(1, len(folds), figsize=(4.5 * len(folds), 4.2), sharey=True)
    axes_arr = np.atleast_1d(axes)

    for ax, fold in zip(axes_arr, folds):
        ho = held_out_source(fold)
        sub = t1_df[t1_df["fold"] == fold]
        present = [s for s in scenarios if not sub[sub["scenario"] == s].empty]
        x = np.arange(len(present))
        w = 0.35
        ins_vals = [float(sub[sub["scenario"] == s]["in_source_median_dice"].iloc[0]) for s in present]
        fo_vals = [float(sub[sub["scenario"] == s]["foreign_median_dice"].iloc[0]) for s in present]

        ax.bar(x - w / 2, ins_vals, w, label="Dalam sumber", color="#4C72B0")
        ax.bar(x + w / 2, fo_vals, w, label="Sumber asing", color="#DD8452")
        for xi, iv, fv in zip(x, ins_vals, fo_vals):
            ax.annotate(f"Δ={iv - fv:.3f}", xy=(xi, max(iv, fv) + 0.012), ha="center", fontsize=8)

        ax.set_xticks(x, present)
        ax.set_title(f"Lipatan {SOURCE_TITLE[ho]}")
        ax.set_xlabel("Skenario kontras (kode)")
        ax.set_ylim(0.0, 1.12)
        ax.grid(axis="y", alpha=0.25)

    axes_arr[0].set_ylabel("Median koefisien Dice (tanpa satuan)")
    axes_arr[0].legend(loc="lower right", frameon=False)
    fig.suptitle("Celah Generalisasi: Median Dice Dalam Sumber vs Sumber Asing", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "F2_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f3_alarm_scatter(
    scores_df: pd.DataFrame,
    tau_df: pd.DataFrame,
    cfg: dict,
    fig_dir: Path,
) -> None:
    """F3_alarm_scatter.png: Alarm score vs Dice on foreign sources (S0 baseline), failed points highlighted."""
    folds = cfg["folds"]
    fail_thr = float(cfg["metrics"]["failure_dice"])
    fig, axes = plt.subplots(2, len(folds), figsize=(4.3 * len(folds), 7.0), sharey=True)
    if len(folds) == 1:
        axes = axes[:, None]

    sub_s0 = scores_df[(scores_df["scenario"] == "S0") & (scores_df["group"] == "foreign")]
    if sub_s0.empty:
        sub_s0 = scores_df[scores_df["group"] == "foreign"]

    for row_idx, (alarm, xlabel) in enumerate(
        (("A1", "Jarak intensitas A1 (selisih CDF)"), ("A2", "Entropi biner A2 (nat)"))
    ):
        for col_idx, fold in enumerate(folds):
            ax = axes[row_idx, col_idx]
            ho = held_out_source(fold)
            df_f = sub_s0[sub_s0["fold"] == fold]
            if not df_f.empty:
                ok = df_f[~df_f["failed"]]
                bad = df_f[df_f["failed"]]
                ax.scatter(ok[alarm], ok["dice"], s=14, alpha=0.55, color="#4C72B0", label="Berhasil (Dice ≥ 0,90)")
                ax.scatter(bad[alarm], bad["dice"], s=18, alpha=0.75, color="crimson", label="Gagal (Dice < 0,90)")

                t_rows = tau_df[(tau_df["fold"] == fold) & (tau_df["scenario"] == df_f["scenario"].iloc[0]) & (tau_df["alarm"] == alarm)]
                if not t_rows.empty:
                    tau_val = float(t_rows["tau"].iloc[0])
                    ax.axvline(tau_val, color="darkorange", linestyle="--", linewidth=1.1, label=f"Ambang tau ({tau_val:.3f})")

            ax.axhline(fail_thr, color="gray", linestyle=":", linewidth=1.0)
            if row_idx == 0:
                ax.set_title(f"Sumber asing: {SOURCE_TITLE[ho]}")
            ax.set_xlabel(xlabel)
            if col_idx == 0:
                ax.set_ylabel("Koefisien Dice (tanpa satuan)")
            ax.set_ylim(0.0, 1.03)
            ax.grid(alpha=0.2)
            if row_idx == 0 and col_idx == len(folds) - 1:
                ax.legend(loc="lower right", frameon=True, fontsize=7.5)

    fig.suptitle("Skor Alarm Tanpa Label vs Koefisien Dice pada Sumber Asing (S0)", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "F3_alarm_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f4_clahe_grid(t4_df: pd.DataFrame, cfg: dict, fig_dir: Path) -> None:
    """F4_clahe_grid.png: Heatmap of 12 CLAHE parameter combinations (validation Dice) per fold."""
    folds = cfg["folds"]
    fig, axes = plt.subplots(1, len(folds), figsize=(4.8 * len(folds), 4.0), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)

    if t4_df.empty:
        for ax, fold in zip(axes_arr, folds):
            ax.text(0.5, 0.5, "Data tuning tidak tersedia", ha="center", va="center")
            ax.set_title(fold)
        plt.savefig(fig_dir / "F4_clahe_grid.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    vmin = float(t4_df["val_dice"].min())
    vmax = float(t4_df["val_dice"].max())
    tile_order = [f"{t[0]}x{t[1]}" for t in cfg["clahe_grid"]["tile_grid"]]
    clip_order = [float(c) for c in cfg["clahe_grid"]["clip_limit"]]

    for ax, fold in zip(axes_arr, folds):
        sub = t4_df[t4_df["fold"] == fold]
        piv = sub.pivot(index="clip_limit", columns="tile_grid", values="val_dice")
        piv = piv.reindex(index=clip_order, columns=tile_order)
        im = ax.imshow(piv.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

        win = sub[sub["is_winner"]]
        win_c = float(win["clip_limit"].iloc[0]) if not win.empty else None
        win_t = str(win["tile_grid"].iloc[0]) if not win.empty else None

        for i, c in enumerate(clip_order):
            for j, tg in enumerate(tile_order):
                val = piv.iloc[i, j]
                if np.isnan(val):
                    continue
                star = "*" if (c == win_c and tg == win_t) else ""
                txt_color = "black" if val > (vmin + vmax) / 2.0 else "white"
                ax.text(j, i, f"{val:.4f}{star}", ha="center", va="center", color=txt_color, fontsize=8)

        ax.set_xticks(range(len(tile_order)), tile_order)
        ax.set_yticks(range(len(clip_order)), [f"{c:g}" for c in clip_order])
        ax.set_xlabel("Ukuran grid ubin (piksel)")
        ax.set_title(f"Lipatan {SOURCE_TITLE[held_out_source(fold)]}")

    axes_arr[0].set_ylabel("Batas kliping CLAHE (clip limit)")
    fig.colorbar(im, ax=axes_arr.tolist(), shrink=0.85, label="Dice validasi (tanpa satuan)")
    fig.suptitle("Pencarian Grid 12 Kombinasi CLAHE (* = pemenang per lipatan)", fontsize=11)
    plt.savefig(fig_dir / "F4_clahe_grid.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f5_worst_cases(
    pred_dir: Path,
    avg_df: pd.DataFrame,
    manifest: pd.DataFrame,
    processed_dir: Path,
    cfg: dict,
    fig_dir: Path,
) -> None:
    """F5_worst_cases.png: 3 lowest-S0-Dice foreign images per source (GT, S0 pred, CLAHE S2/S3 pred)."""
    folds = cfg["folds"]
    man_idx = manifest.set_index("id")
    seed0 = int(cfg["train"]["seeds"][0])
    pp = cfg["postprocess"]

    # Pick up to 3 worst S0 foreign images per fold
    chosen: list[tuple[str, str, float]] = []
    for fold in folds:
        sub = avg_df[(avg_df["fold"] == fold) & (avg_df["scenario"] == "S0") & (avg_df["group"] == "foreign")]
        if sub.empty:
            continue
        worst = sub.sort_values("dice", ascending=True).head(3)
        for r in worst.itertuples():
            chosen.append((fold, str(r.id), float(r.dice)))

    if not chosen:
        return

    # Determine available contrast scenarios for columns
    col_specs = [("GT", "GT (Mask Asli)")]
    for scen, title in (("S0", "Prediksi S0"), ("S2", "Prediksi CLAHE (S2)"), ("S3", "Prediksi CLAHE Tuning (S3)")):
        if not avg_df[avg_df["scenario"] == scen].empty:
            col_specs.append((scen, title))

    # Load probability maps for seed0 once per (fold, scenario)
    prob_lookup: dict[tuple[str, str, str], np.ndarray] = {}
    for fold in folds:
        for scen, _ in col_specs[1:]:
            sdir = pred_dir / fold / scen
            if not sdir.exists():
                continue
            npz = sdir / f"seed{seed0}" / "pred.npz"
            if not npz.exists():
                cands = sorted(sdir.glob("seed*/pred.npz"))
                if not cands:
                    continue
                npz = cands[0]
            with np.load(npz, allow_pickle=True) as data:
                for img_id, prob in zip(data["ids"], data["prob_u8"]):
                    prob_lookup[(fold, scen, str(img_id))] = prob

    nrows = len(chosen)
    ncols = len(col_specs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.9 * nrows))
    if nrows == 1:
        axes = axes[None, :]
    if ncols == 1:
        axes = axes[:, None]

    dice_lookup = avg_df.set_index(["fold", "scenario", "group", "id"])["dice"]

    for r_i, (fold, img_id, _) in enumerate(chosen):
        mrow = man_idx.loc[img_id]
        img = cv2.imread(str(processed_dir / str(mrow["image_path"])), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(processed_dir / str(mrow["mask_path"])), cv2.IMREAD_GRAYSCALE) > 0

        for c_i, (key, title) in enumerate(col_specs):
            ax = axes[r_i, c_i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.1)
            if key == "GT":
                ax.set_title(f"{img_id}\n{title}", fontsize=8)
            else:
                prob = prob_lookup.get((fold, key, img_id))
                if prob is not None:
                    pmask = postprocess_mask(
                        prob,
                        threshold=float(pp["threshold"]),
                        keep_components=int(pp["keep_components"]),
                        closing_kernel=int(pp["closing_kernel"]),
                    )
                    ax.contour(pmask, levels=[0.5], colors="red", linewidths=1.1)
                d_val = dice_lookup.get((fold, key, "foreign", img_id), float("nan"))
                ax.set_title(f"{title}\nDice={d_val:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_i == 0:
                ax.set_ylabel(f"{SOURCE_TITLE[held_out_source(fold)]}\n(piksel)", fontsize=8)

    fig.suptitle("Kasus Terburuk per Sumber Asing (hijau=GT, merah=prediksi)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(fig_dir / "F5_worst_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f6_histograms(
    manifest: pd.DataFrame,
    processed_dir: Path,
    s3_params_path: Path,
    cfg: dict,
    fig_dir: Path,
) -> None:
    """F6_histograms.png: Pixel intensity distributions per source before (S0) and after S1, S2, S3."""
    scenarios = cfg["scenarios"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    bins = np.arange(256)

    # Load all images per source once
    raw_by_source: dict[str, np.ndarray] = {}
    for src in cfg["sources"]:
        rows = manifest[manifest["source"] == src]
        if not rows.empty:
            imgs, _ = load_arrays(rows, processed_dir)
            raw_by_source[src] = imgs

    for ax, scenario in zip(axes_flat, scenarios):
        for src, imgs in raw_by_source.items():
            clahe = _get_clahe_for_scenario(f"ho_{src}", scenario, s3_params_path)
            enh = np.stack([enhance(im, scenario, clahe) for im in imgs])
            h = pooled_hist(enh)
            ax.plot(bins, h, linewidth=1.4, label=SOURCE_TITLE[src])
        ax.set_title(f"Skenario {scenario}")
        ax.set_xlabel("Intensitas piksel (0-255)")
        ax.set_ylabel("Proporsi piksel (fraksi)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Sebaran Intensitas Piksel per Sumber Sebelum dan Sesudah Skenario Kontras", fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "F6_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    pred_dir: Path,
    out_dir: Path,
    processed_dir: Path,
    tables_ref_dir: Path,
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    """Run evaluation, alarm, statistical tests, tables T1..T7, and figures F1..F6."""
    _configure_matplotlib()
    metrics_dir = out_dir / "metrics"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(processed_dir)
    raw_df = evaluate_all(pred_dir, metrics_dir, cfg, processed_dir)
    if raw_df.empty:
        raise FileNotFoundError(f"No pred.npz files found under {pred_dir}")

    avg_df = average_over_seeds(raw_df, failure_dice=float(cfg["metrics"]["failure_dice"]))
    s3_path = _resolve_s3_params_path(tables_ref_dir)

    t1 = build_t1_gap(avg_df, cfg)
    t2 = build_t2_scenarios(avg_df, cfg)
    t3 = build_t3_clinical(avg_df, cfg)
    t4 = build_t4_tuning(manifest, processed_dir, tables_ref_dir, cfg)

    scores_df, tau_df = compute_shift_records(pred_dir, manifest, processed_dir, s3_path, avg_df, cfg)
    t5 = build_t5_alarm(scores_df, tau_df)
    t6 = build_t6_readiness(avg_df, scores_df, tau_df, cfg)
    t7 = build_t7_sample50(scores_df, tau_df, cfg)

    tables = {
        "T1_gap": t1,
        "T2_scenarios": t2,
        "T3_clinical": t3,
        "T4_tuning": t4,
        "T5_alarm": t5,
        "T6_readiness": t6,
        "T7_sample50": t7,
    }
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)

    plot_f1_dice_by_scenario(avg_df, cfg, figures_dir)
    plot_f2_gap(t1, cfg, figures_dir)
    plot_f3_alarm_scatter(scores_df, tau_df, cfg, figures_dir)
    plot_f4_clahe_grid(t4, cfg, figures_dir)
    plot_f5_worst_cases(pred_dir, avg_df, manifest, processed_dir, cfg, figures_dir)
    plot_f6_histograms(manifest, processed_dir, s3_path, cfg, figures_dir)

    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate official CXR-Seg evaluation metrics, tables, and figures.")
    ap.add_argument("--pred-dir", default="results/predictions", help="Predictions root ({fold}/{scenario}/seed{seed}/pred.npz)")
    ap.add_argument("--out", default="results", help="Output root directory (writes metrics/, tables/, figures/)")
    ap.add_argument("--processed-dir", default=None, help="Override processed data directory")
    ap.add_argument("--tables-dir", default="results/tables", help="Reference tables directory for tuning_*.csv and s3_params.csv")
    ap.add_argument("--config", default=None, help="Override config YAML path")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_absolute():
        pred_dir = ROOT / pred_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    proc_dir = Path(args.processed_dir) if args.processed_dir else ROOT / cfg["data"]["processed_dir"]
    if not proc_dir.is_absolute():
        proc_dir = ROOT / proc_dir
    tables_ref = Path(args.tables_dir)
    if not tables_ref.is_absolute():
        tables_ref = ROOT / tables_ref

    tables = run_analysis(pred_dir, out_dir, proc_dir, tables_ref, cfg)
    print(f"Wrote 7 tables and 6 figures -> {out_dir}")
    for name, df in tables.items():
        print(f"  {name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
