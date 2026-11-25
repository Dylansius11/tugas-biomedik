# STATE: Nabil (QC, evaluasi, alarm, statistik, tabel/gambar)

Catatan keadaan setelah implementasi modul N1–N6. Diperiksa ulang pada 25 November 2026
di working tree `main` (belum ada commit untuk berkas Nabil).
Rencana: `docs/plans/BUILD-PLAN-NABIL.md`. Definisi: `docs/PROTOCOL.md` bagian 2, 7, 8, 9.

## Yang sudah ada di working tree

| Modul | Berkas | Tes |
|---|---|---|
| N1 QC | `scripts/qc_data.py` | `tests/test_qc.py` |
| N2 Evaluasi | `src/cxrseg/evaluate.py` | `tests/test_evaluate.py` |
| N3 Alarm | `src/cxrseg/shift.py` | `tests/test_shift.py` |
| N4 Statistik | `src/cxrseg/stats.py` | `tests/test_stats.py` |
| N5 Analisis | `scripts/analyze.py` | `tests/test_analyze.py` |
| N6 Referensi | `paper/proposal/REFERENSI.md` (diubah) | — |

## Verifikasi yang dijalankan

| Perintah | Hasil |
|---|---|
| `pytest` 9 berkas non-torch | **33 passed** (145 detik) |
| `pytest` seluruh suite | **gagal koleksi 5 berkas** karena `torch` belum terpasang (`test_dataset`, `test_model`, `test_train`, `test_exp2`, `test_tune`) |
| `python -m cxrseg.evaluate --help` | exit 0 |
| `python scripts/qc_data.py --help` | exit 0 |
| `python scripts/analyze.py --help` | exit 0 |
| `test_analyze.py` | 7 CSV identik byte-per-byte pada dua kali jalan, 6 PNG terbentuk (data sintetis) |

Lingkungan `.venv`: Python 3.13.9, numpy 2.4.4, pandas 3.0.2, scipy/opencv/scikit-learn/matplotlib tersedia.
`torch` **belum** terpasang.

## Status terhadap kriteria selesai

| Langkah | Status | Catatan |
|---|---|---|
| N0 Lingkungan | **sebagian** | venv + dependensi inti ada; torch belum; `python -m pytest -q` belum hijau; data belum diekstrak; belum di cabang `nabil/analysis` |
| N1 QC | **kode selesai, luaran belum** | `data/processed` kosong (hanya `.gitkeep`), jadi 3 gambar + 2 tabel belum dibuat dan `docs/QC-REPORT.md` belum ada |
| N2 Evaluasi | **kode + tes selesai** | kriteria "36 kombinasi prediksi palsu terevaluasi" belum terbukti di workspace: `results/fake_predictions` tidak ada dan tidak bisa dibuat tanpa manifest |
| N3 Alarm | **selesai** | A1, A2, tau, alarm rate, aturan siap, Spearman/AUROC sesuai PROTOCOL bagian 8 |
| N4 Statistik | **selesai** | Holm, Wilcoxon berpasangan, bootstrap CI, aturan efek minimum |
| N5 Analisis | **kode + tes selesai** | 7 tabel + 6 gambar terbentuk pada data sintetis; jalan pada prediksi asli menunggu Huda (`results/predictions` kosong) |
| N6 Referensi | **sebagian** | DOI dan angka referensi sudah dicek; bagian "setiap angka laporan Dylan bisa ditunjuk ke satu baris CSV" belum dikerjakan |

## Berkas yang belum ada

- `docs/QC-REPORT.md` (wajib N1)
- `results/figures/qc_{source}.png`, `results/tables/qc_summary.csv`, `results/tables/qc_duplicates.csv`
- `results/fake_predictions/` dan `results/fake_metrics/`
- `results/predictions/{fold}/{scenario}/seed{seed}/pred.npz` (milik Huda)

Catatan: `results/fake_predictions` dan seluruh `*.npz` memang di-`.gitignore`, jadi
ketiadaannya tidak otomatis berarti belum pernah dijalankan. Tetapi luaran yang bisa
diperiksa ulang (CSV/gambar) juga belum ada di working tree.

## Catatan kode (bukan penghalang)

- `evaluate.py` membaca mask GT dengan `gt_u8 > 0`. Aman karena `prepare_data.py` sudah
  menulis mask 0/255, tetapi `binarize_mask` (ambang 127) lebih tepat bila mask mentah
  suatu saat dipakai.
- `compute_alarm_evaluation` mensyaratkan minimal 10 citra gagal **dan** 10 citra lulus
  untuk AUROC. Ini lebih ketat dari "minimal 10 gagal" di rencana; hasilnya tetap
  "tidak dapat dihitung" bila salah satu kelas terlalu kecil.
- `intensity_distance` membagi jumlah selisih CDF dengan 255, persis seperti bunyi rencana.
- `plot_f5_worst_cases` menyimpan peta probabilitas semua lipatan/skenario di memori.
  Pada data asli (hingga ~600 citra per sumber) perlu dipantau, bisa dibatasi ke id terpilih.
- `build_t4_tuning` mengulang `enhance` untuk seluruh citra latih + asing pada 12 kombinasi.
  Benar, tetapi boros waktu; optimasi bisa dilakukan setelah data asli tersedia.
- Uji determinisme membandingkan CSV; PNG tidak dibandingkan byte-per-byte.

## Langkah berikutnya

1. Pasang torch (CPU cukup untuk tes) lalu jalankan `python -m pytest -q` sampai hijau.
2. Ekstrak `data/processed.zip` dari Drive Dylan ke `data/processed`.
3. `python scripts/make_fake_predictions.py` -> `python -m cxrseg.evaluate ...` ->
   `python scripts/analyze.py ...` untuk membuktikan kriteria N2/N5 pada 36 kombinasi.
4. `python scripts/qc_data.py` lalu tulis `docs/QC-REPORT.md` dan kirim daftar citra
   bermasalah ke Dylan (Dylan yang memutuskan pembuangan).
5. Commit di cabang `nabil/analysis`, satu commit satu perilaku.
6. Setelah laporan Dylan ada, kerjakan bagian N6 "cek angka ke satu baris CSV".
