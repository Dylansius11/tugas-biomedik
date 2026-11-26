# STATE: Nabil (QC, evaluasi, alarm, statistik, tabel/gambar)

Keadaan per 26 November 2026, cabang `nabil/analysis`.
Rencana: `docs/plans/BUILD-PLAN-NABIL.md` (hasil utama), `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2.md` (Percobaan 2), `docs/plans/BRANCHING-PERCOBAAN-2.md` (cabang skenario A dan B).
Definisi: `docs/PROTOCOL.md` bagian 2, 7, 8, 9.

**Cabang ini belum diminta merge ke `main`.** Analisis Percobaan 2 dikerjakan di `nabil/exp2-a` (tambahan) dan `nabil/exp2-b` (hasil utama). Cek progres: `python scripts/progress.py`.

## Status

| Langkah | Status | Bukti |
|---|---|---|
| N0 Lingkungan | selesai | Python 3.13, torch CPU, `python -m pytest -q` 51 lolos |
| N1 QC | selesai | `results/figures/qc_*.png`, `results/tables/qc_*.csv`, `docs/QC-REPORT.md`. Keputusan sementara 26-11: citra 00006967 ditandai, verifikasi duplikat kedua (0 duplikat) |
| N2 Evaluasi | selesai | 36 CSV di `results/metrics`, baris = jumlah citra, tanpa NaN |
| N3 Alarm | selesai | T5–T7 |
| N4 Statistik | selesai | T2 (Wilcoxon, bootstrap, Holm 9 uji) |
| N5 Analisis | selesai | 7 tabel + 6 gambar dari prediksi asli. Dua kali jalan pada prediksi palsu: tabel, metrik, dan PNG identik byte per byte |
| N6 Referensi | sebagian | DOI dicek. Cek angka laporan ke CSV menunggu laporan Dylan |
| B1–B3 Percobaan 2 skenario B | selesai (cabang `nabil/exp2-b`) | `results/exp2/main/` T0–T9, F1–F5, `docs/HASIL-NABIL-PERCOBAAN-2-B.md`, plan `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2-B.md`. Verifikasi 105/105 cocok |

## Data yang dipakai

- `data/processed` dibangun ulang dari Mendeley (`download_data.sh`, 4 SHA-256 cocok) dengan `prepare_data.py`: Darwin 600, Montgomery 138, Shenzhen 566.
- Prediksi: 36 `pred.npz` percobaan utama (U-Net satu head) dari Huda, dibaca dengan `--pred-dir` dari luar repo. Urutan `ids` dan `group` identik dengan prediksi palsu, jadi pembagian data sama.
- Belum dipakai: prediksi Percobaan 2 (E1, E2 dua head, few-shot).

## Pencocokan dengan angka eksploratif Huda (`STATE-PERCOBAAN-1.md`)

- RM1, RM2, A1: cocok persis.
- A2 berbeda: Huda memakai seed 11 saja, `analyze.py` merata-ratakan 3 seed per citra. Dengan seed 11 saja, kode ini menghasilkan angka Huda persis (Darwin 18,2%, Montgomery 2,2%, Shenzhen 0,5%). Alarm A2 Darwin S0 per seed: 18,2%, 60,8%, 50,0%; rata-rata 3 seed 51,7%.
- `ho_shenzhen` S3 = S2: tuning memilih clip 2,0 tile 8x8 (sama dengan bawaan), `pred.npz` identik byte per byte. Dua dari sembilan uji Holm tidak independen.
- Baris E0 di `results/exp2/summary.txt` sama dengan S0 di T1.

## Untuk Dylan

1. Keputusan sementara QC (Dylan belum menjawab): `darwin/00006967_...annot.original` dipertahankan dan ditandai (tidak pernah gagal di 36 run), verifikasi duplikat kedua ditambahkan (korelasi piksel ≥ 0,99, hasil 0 dari 860). Setuju atau dibalik?
2. Posisi Percobaan 2 di laporan: A (tambahan) atau B (hasil utama). Keduanya dikerjakan di cabang terpisah.

## Catatan kode

- `analyze.py`: judul F5 tidak lagi menumpuk, label "(unitless)" diganti "(tanpa satuan)".
- `make_fake_predictions.py` bawaannya hanya seed 11. Untuk 36 kombinasi pakai `--seeds 11 22 33`.
- `results/fake_*` di-`.gitignore`. Angka dari prediksi palsu tidak boleh masuk laporan.
- `qc_data.py`: kolom `pixel_corr`, `is_duplicate` di `qc_duplicates.csv`, berkas baru `qc_flags.csv`, ambang di `configs/default.yaml` bagian `qc`.
- `scripts/progress.py`: cek progres. Catatan kerja di `progress/log.txt` (tidak masuk git).
- `evaluate.py` membaca mask GT dengan `> 0`. Aman karena mask processed hanya 0/255.

## Cara mengulang

```
python scripts/qc_data.py
python scripts/analyze.py --pred-dir <folder predictions> --out results     # sekitar 6 menit di CPU
python scripts/analyze_exp2.py --mode main --pred-dir <folder predictions> --exp2-dir <folder percobaan2> --out results/exp2/main   # cabang nabil/exp2-b, sekitar 5 menit
```
