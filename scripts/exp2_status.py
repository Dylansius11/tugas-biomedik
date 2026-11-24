"""Progress of Percobaan 2: python scripts/exp2_status.py"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cxrseg.config import ROOT, load_config  # noqa: E402

cfg = load_config()
R = ROOT / "results" / "exp2"
e, folds, seeds = cfg["exp2"], cfg["folds"], cfg["train"]["seeds"]

done = {a: [] for a in e["arms"]}
for a in e["arms"]:
    for f in folds:
        for s in seeds:
            v = R / a / f / f"seed{s}" / "val_dice.txt"
            if v.exists():
                done[a].append((f, s, float(v.read_text())))
total = len(folds) * len(seeds)
print("== Tahap 1: latih E1 / E2 ==")
for a in e["arms"]:
    print(f"{a}: {len(done[a])}/{total} run selesai" + (f", rata-rata val Dice {sum(d for *_, d in done[a]) / len(done[a]):.4f}" if done[a] else ""))

# the run in progress
cur = [p for p in R.glob("E*/*/seed*/last.pt") if not (p.parent / "val_dice.txt").exists()]
for p in cur:
    import torch
    st = torch.load(p, map_location="cpu", weights_only=False)["state"]
    print(f"sedang jalan: {p.parent.relative_to(R)}  epoch {st['epoch']}, val terbaik {st['best']:.4f}")

fa = R / "final_arm.txt"
print("\n== Tahap 2: lengan akhir ==")
print(fa.read_text().strip() if fa.exists() else "belum dipilih (menunggu semua run E1 dan E2 selesai)")

print("\n== Tahap 3: few-shot ==")
n = len(list((R / "fewshot").rglob("*.npz"))) if (R / "fewshot").exists() else 0
want = len(folds) * len(e["fewshot"]["k"]) * len(seeds)
print(f"{n}/{want} adaptasi selesai")

print("\n== Tahap 4: laporan ==")
s = R / "summary.txt"
print(f"ada, dibuat {datetime.fromtimestamp(s.stat().st_mtime):%H:%M}: {s}" if s.exists() else "belum dibuat. Jalankan: python scripts/exp2_report.py")
print(f"gambar: {R / 'figures'}")
print("\nlog langsung: tail -f results/exp2/run.log")
