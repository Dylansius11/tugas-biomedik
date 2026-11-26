"""Cek progres pekerjaan Nabil (QC, Percobaan 2 skenario A dan B) tanpa harus membuka chat.

Membaca keadaan nyata: cabang git, berkas yang sudah di-commit di tiap cabang, proses yang sedang jalan,
dan catatan kerja di `progress/log.txt` (tidak masuk git, jadi tetap ada saat pindah cabang).

Usage:
    python scripts/progress.py               # tampilkan sekali
    python scripts/progress.py --watch 20    # segarkan tiap 20 detik (Ctrl+C untuk berhenti)
    python scripts/progress.py log A N9 "selesai statistik"   # tambah satu baris catatan
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "progress" / "log.txt"
RUNNING = ROOT / "progress" / "running.txt"

# (cabang, langkah, keterangan, berkas bukti di cabang itu)
CHECKS: list[tuple[str, str, str, list[str]]] = [
    ("nabil/analysis", "QC", "Keputusan sementara QC + verifikasi duplikat", ["results/tables/qc_flags.csv"]),
    ("nabil/analysis", "DOC", "Dokumen branching A/B", ["docs/plans/BRANCHING-PERCOBAAN-2.md"]),
    ("nabil/exp2-a", "N7", "Validasi 54 berkas", ["results/exp2/nabil/X0_validation.csv"]),
    ("nabil/exp2-a", "N8", "Verifikasi angka Huda", ["results/exp2/nabil/X1_verify_huda.csv"]),
    ("nabil/exp2-a", "N9", "Statistik E2 vs E0", ["results/exp2/nabil/X2_e2_vs_e0.csv", "results/exp2/nabil/X3_clinical.csv"]),
    ("nabil/exp2-a", "N10", "Alarm A1/A2 pada E2", ["results/exp2/nabil/X4_alarm.csv", "results/exp2/nabil/X5_readiness.csv"]),
    ("nabil/exp2-a", "N11", "Few-shot", ["results/exp2/nabil/X6_fewshot.csv"]),
    ("nabil/exp2-a", "N12", "Gambar + dokumen hasil A", ["results/exp2/nabil/figures/XF1_dice_box.png", "docs/HASIL-NABIL-PERCOBAAN-2.md"]),
    ("nabil/exp2-b", "B1", "Plan B (Percobaan 2 = hasil utama)", ["docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2-B.md"]),
    ("nabil/exp2-b", "B2", "Tabel utama B (T1-T7 versi E0/E1/E2)", ["results/exp2/main/tables/T2_arms.csv", "results/exp2/main/tables/T6_readiness.csv"]),
    ("nabil/exp2-b", "B3", "Gambar + dokumen hasil B", ["results/exp2/main/figures/F1_dice_box.png", "docs/HASIL-NABIL-PERCOBAAN-2-B.md"]),
]


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def branch_exists(b: str) -> bool:
    return bool(git("rev-parse", "--verify", "--quiet", b))


def in_branch(b: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{b}:{path}"], cwd=ROOT, capture_output=True).returncode == 0


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_log(track: str, step: str, msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {track:<3} | {step:<4} | {msg}\n")


def show() -> None:
    print(f"=== Progres Nabil   ({datetime.now():%Y-%m-%d %H:%M:%S}) ===")
    cur = git("branch", "--show-current")
    print(f"Cabang aktif: {cur}   berkas belum di-commit: {len(git('status', '--porcelain').splitlines())}")

    print("\n-- Cabang --")
    for b in ("nabil/analysis", "nabil/exp2-a", "nabil/exp2-b"):
        if not branch_exists(b):
            print(f"  {b:<16} belum dibuat")
            continue
        last = git("log", "-1", "--format=%h %cd %s", "--date=format:%d-%m %H:%M", b)
        remote = git("rev-parse", "--verify", "--quiet", f"origin/{b}")
        pushed = "sudah push" if remote and remote == git("rev-parse", b) else ("belum push" if not remote else "ada commit belum push")
        print(f"  {b:<16} [{pushed}] {last}")

    print("\n-- Langkah (selesai = bukti sudah di-commit di cabangnya) --")
    for b, step, desc, files in CHECKS:
        if branch_exists(b) and all(in_branch(b, f) for f in files):
            mark = "[x]"
        elif any((ROOT / f).exists() for f in files) and cur == b:
            mark = "[~]"  # ada di folder kerja tapi belum di-commit
        else:
            mark = "[ ]"
        print(f"  {mark} {b:<16} {step:<4} {desc}")
    print("  ([x] selesai, [~] sedang/belum di-commit, [ ] belum)")

    print("\n-- Proses yang sedang jalan --")
    if RUNNING.exists():
        txt = RUNNING.read_text(encoding="utf-8").strip()
        try:
            pid = int(txt.split("|")[0].strip())
            state = "JALAN" if pid_alive(pid) else "berhenti (tidak selesai normal?)"
        except ValueError:
            state = "?"
        print(f"  {state}: {txt}")
    else:
        print("  tidak ada")

    print("\n-- Catatan kerja terakhir (progress/log.txt) --")
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines()[-15:]:
            print("  " + line)
    else:
        print("  belum ada catatan")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "log":
        if len(sys.argv) < 5:
            sys.exit('pakai: python scripts/progress.py log <A|B|QC> <langkah> "pesan"')
        write_log(sys.argv[2], sys.argv[3], " ".join(sys.argv[4:]))
        return
    ap = argparse.ArgumentParser(description="Cek progres pekerjaan Nabil.")
    ap.add_argument("--watch", type=int, default=0, help="Segarkan tiap N detik")
    a = ap.parse_args()
    if not a.watch:
        show()
        return
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            show()
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
