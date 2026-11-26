# BRANCHING: analisis Percobaan 2, skenario A dan B

Dibuat 26 November 2026. Pemilik: Nabil.

Posisi Percobaan 2 di laporan belum diputuskan. Dylan meminta analisis memakai data Percobaan 2, sedangkan `docs/STATE-PERCOBAAN-2.md` menyebutnya analisis tambahan, bukan pengganti hasil utama. Kedua kemungkinan dikerjakan di cabang terpisah. Cabang yang dipilih tim yang nanti di-merge.

## Dua skenario

| | Skenario A: Percobaan 2 sebagai tambahan | Skenario B: Percobaan 2 sebagai hasil utama |
|---|---|---|
| Hasil utama laporan | 36 run S0–S3 (T1–T7, F1–F6) | E0 vs E1 vs E2 (18 run baru + E0) |
| Peran Percobaan 2 | Menjelaskan penyebab kegagalan, berlabel post-hoc | Menjawab pertanyaan riset utama |
| Peran Percobaan 1 | Tetap utama, termasuk jawaban RM2 (CLAHE) | Pendukung: hasil CLAHE pindah ke lampiran |
| Uji Holm | 3 uji (E2 vs E0, 3 sumber asing). E1 hanya deskriptif | 6 uji (E1 dan E2 vs E0, 3 sumber asing), sama seperti T2 yang memakai 9 uji |
| Alarm A1/A2 | Pada E2, dibandingkan dengan A3 Huda | Pada E0, E1, E2 (seperti T5–T7 per skenario) |
| Luaran | `results/exp2/nabil/` (X0–X6, XF1–XF4) | `results/exp2/main/` (T1–T7, F1–F5 versi lengan) |
| Dokumen | `docs/HASIL-NABIL-PERCOBAAN-2.md` | `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2-B.md`, `docs/HASIL-NABIL-PERCOBAAN-2-B.md` |
| Risiko | Rendah. Tidak ada klaim yang berubah | RM2 tidak terjawab oleh Percobaan 2. Lengan E1/E2 dirancang setelah melihat hasil Percobaan 1, jadi tetap harus disebut post-hoc |

## Struktur cabang

```
main
 └── nabil/analysis            Percobaan 1 (T1–T7, F1–F6), QC, keputusan sementara QC
      ├── nabil/exp2-a         skenario A (dikerjakan lebih dulu)
      └── nabil/exp2-b         skenario B (skrip analisis di-cherry-pick dari A, lalu ditambah mode utama)
```

| Cabang | Dibuat dari | Isi | Kapan di-merge |
|---|---|---|---|
| `nabil/analysis` | `main` | QC, evaluasi, alarm, statistik, analisis Percobaan 1 | Saat tim siap. Tidak bergantung pada keputusan A/B |
| `nabil/exp2-a` | `nabil/analysis` | `scripts/analyze_exp2.py`, tes, `results/exp2/nabil/`, dokumen hasil A | Bila Dylan memilih A |
| `nabil/exp2-b` | `nabil/analysis` | Skrip yang sama + mode `--mode main`, `results/exp2/main/`, plan dan dokumen B | Bila Dylan memilih B |

Mengapa B tidak dibuat dari A: kedua cabang saling menggantikan. Bila B dipilih, tabel tambahan versi A tidak ikut masuk ke `main`. Commit skrip yang sama di kedua cabang adalah duplikat yang disengaja.

## Aturan

1. **Hanya satu dari A atau B yang di-merge.** Cabang yang tidak dipilih dihapus setelah merge.
2. Urutan merge: `nabil/analysis` dulu, lalu cabang terpilih di-rebase ke `main`, lalu PR terpisah.
3. Kedua cabang tidak mengubah `results/tables`, `results/figures`, `results/metrics`, atau `scripts/analyze.py`.
4. Ambang alarm (tau) hanya dari citra validasi dua sumber latih. Head untuk Darwin tetap `visible` (dari metadata, bukan hasil uji).
5. Data Percobaan 2 dibaca dari luar repo (`--exp2-dir`). `pred.npz` tidak masuk git.
6. Commit memakai identitas git laptop ini.

## Data

| Isi | Lokasi | Jumlah |
|---|---|---|
| E0 = S0 Percobaan 1 | `E:\NabilRZ\1. Tugas\UNS\sem 7\Biomedik\code\predictions\{fold}\S0\seed{s}\pred.npz` | 9 |
| E1, E2 | `E:\NabilRZ\1. Tugas\UNS\sem 7\Biomedik\code\percobaan2\{E1,E2}\{fold}\seed{s}\` (`pred.npz`, `val_dice.txt`) | 18 + 18 |
| Few-shot E2 | `...\percobaan2\fewshot\E2\{fold}\k{5,10,20,50}_seed{s}.npz` | 36 |

## Perintah

```
# skenario A (di cabang nabil/exp2-a)
python scripts/analyze_exp2.py --mode a --pred-dir <folder predictions> --exp2-dir <folder percobaan2> --out results/exp2/nabil

# skenario B (di cabang nabil/exp2-b)
python scripts/analyze_exp2.py --mode main --pred-dir <folder predictions> --exp2-dir <folder percobaan2> --out results/exp2/main

# cek progres kapan saja, dari cabang mana saja
python scripts/progress.py
python scripts/progress.py --watch 20
```

## Yang menunggu keputusan

| Dari | Pertanyaan |
|---|---|
| Dylan | A atau B? Bila B, RM2 di PROTOCOL perlu diubah atau dijawab dari Percobaan 1 sebagai lampiran |
| Dylan | Izin merge `nabil/analysis` |
