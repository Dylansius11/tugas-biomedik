# STATE: Nabil (QC, evaluasi, alarm, statistik, tabel/gambar)

Keadaan per 25 November 2026, cabang `nabil/analysis`.
Rencana: `docs/plans/BUILD-PLAN-NABIL.md` (hasil utama) dan `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2.md` (usulan, post-hoc).
Definisi: `docs/PROTOCOL.md` bagian 2, 7, 8, 9.

**Cabang ini belum diminta merge ke `main`.** Menunggu keputusan tim apakah laporan memakai hasil utama saja atau ditambah Percobaan 2.

## Status

| Langkah | Status | Bukti |
|---|---|---|
| N0 Lingkungan | selesai | Python 3.13, torch CPU, `python -m pytest -q` 51 lolos |
| N1 QC | selesai | `results/figures/qc_*.png`, `results/tables/qc_*.csv`, `docs/QC-REPORT.md` |
| N2 Evaluasi | selesai | 36 CSV di `results/metrics`, baris = jumlah citra, tanpa NaN |
| N3 Alarm | selesai | T5–T7 |
| N4 Statistik | selesai | T2 (Wilcoxon, bootstrap, Holm 9 uji) |
| N5 Analisis | selesai | 7 tabel + 6 gambar dari prediksi asli. Dua kali jalan pada prediksi palsu: tabel, metrik, dan PNG identik byte per byte |
| N6 Referensi | sebagian | DOI dicek. Cek angka laporan ke CSV menunggu laporan Dylan |

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

1. `darwin/00006967_fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original` (split test): kotak anotasi tercetak di citra, mask ikut berbentuk kotak. Dibuang atau ditandai?
2. dHash 8x8 menandai 860 pasangan lintas sumber, semuanya bukan duplikat (korelasi piksel maksimum 0,91). Usul verifikasi kedua di `qc_data.py`.
3. Posisi Percobaan 2 di laporan dan persetujuan plan analisisnya.

## Catatan kode

- `analyze.py`: judul F5 tidak lagi menumpuk, label "(unitless)" diganti "(tanpa satuan)".
- `make_fake_predictions.py` bawaannya hanya seed 11. Untuk 36 kombinasi pakai `--seeds 11 22 33`.
- `results/fake_*` di-`.gitignore`. Angka dari prediksi palsu tidak boleh masuk laporan.
- `evaluate.py` membaca mask GT dengan `> 0`. Aman karena mask processed hanya 0/255.

## Cara mengulang

```
python scripts/qc_data.py
python scripts/analyze.py --pred-dir <folder predictions> --out results     # sekitar 6 menit di CPU
```
