# BUILD-PLAN: Nabil (kualitas data, evaluasi, alarm, statistik, gambar)

Tugasmu satu kalimat: **ubah prediksi model menjadi angka, tabel, dan gambar yang menjawab RM1, RM2, dan RM3**, dan pastikan datanya bersih sejak awal.

Kamu tidak perlu menunggu Huda. Jalankan `python scripts/make_fake_predictions.py` untuk membuat prediksi palsu yang mengikuti kontrak asli, lalu kerjakan semuanya di atasnya. Saat prediksi asli jadi, ganti folder masukannya saja. **Angka dari prediksi palsu tidak boleh masuk laporan.**

Yang sudah disediakan Dylan: `cxrseg.metrics` (Dice, IoU, HD95, cakupan, galat luas), `cxrseg.postprocess`, `cxrseg.data` (lipatan), `configs/default.yaml`.
Yang kamu tulis: `scripts/qc_data.py`, `src/cxrseg/evaluate.py`, `src/cxrseg/shift.py`, `src/cxrseg/stats.py`, `scripts/analyze.py`, plus tes untuk tiap modul.

Baca dulu `docs/PROTOCOL.md` bagian 2, 7, 8, 9. Itu definisi yang harus kamu ikuti persis.

## Langkah

### N0. Lingkungan (0,5 hari)
Clone, cabang `nabil/analysis`, `pip install -r requirements.txt`, `pip install -e .`, `python -m pytest -q` hijau. Ambil `processed.zip` dari Drive Dylan dan ekstrak ke `data/processed`.

### N1. QC data (1 hari): `scripts/qc_data.py`
Keluaran dan isinya:
1. `results/figures/qc_{source}.png`: 20 citra acak per sumber (seed 42), citra dengan garis tepi mask berwarna di atasnya. Kamu melihat sendiri apakah mask masuk akal.
2. `results/tables/qc_summary.csv`: per sumber jumlah citra, rentang ukuran asli, rata-rata dan rentang luas mask (`mask_area_frac`), rata-rata dan simpangan baku intensitas.
3. `results/tables/qc_duplicates.csv`: pasangan citra yang sangat mirip **antar sumber**. Hitung dHash 8x8 (ubah ke 9x8, bandingkan piksel bertetangga) dengan numpy atau OpenCV, tandai jarak Hamming <= 4. Tulis kolom `id_a, id_b, hamming`.
4. Catatan visual: adakah citra Darwin yang terbalik (tulang putih jadi hitam), berupa foto, atau dengan mask tampak salah? Tulis di `docs/QC-REPORT.md` (satu halaman) dengan daftar id.
- Mask dalam koleksi punya noise anti-aliasing (nilai 1 sampai 254). Itu sudah ditangani `binarize_mask`, tidak perlu kamu perbaiki, cukup pastikan hasil akhirnya hanya 0 dan 255.
- **Selesai bila:** 3 gambar, 2 tabel, dan `QC-REPORT.md` ada. Kirim daftar citra bermasalah ke Dylan. **Dylan yang memutuskan** apakah citra dibuang. Jangan dibuang sendiri, karena itu mengubah jumlah sampel di proposal.

### N2. Evaluasi (1 hari): `src/cxrseg/evaluate.py`
- Fungsi `evaluate_run(pred_npz_path, manifest, processed_dir, cfg) -> pandas.DataFrame`. Untuk tiap citra: muat probabilitas, `postprocess_mask`, bandingkan dengan mask asli lewat `metrics.evaluate_pair`.
- Kolom keluaran: `fold, scenario, seed, id, source, group, dice, iou, hd95, coverage, area_error_pct, failed`.
- CLI: `python -m cxrseg.evaluate --pred-dir results/fake_predictions --out results/fake_metrics` menulis `{out}/{fold}/{scenario}/seed{seed}.csv`.
- **Tes yang wajib ada:** prediksi sempurna (probabilitas = mask) menghasilkan Dice 1 dan gagal = False. Prediksi kosong menghasilkan `hd95 = 362` dan gagal = True. Jumlah baris = jumlah citra.
- **Selesai bila:** semua 36 kombinasi prediksi palsu terevaluasi tanpa error dan angkanya masuk akal (sumber asing lebih buruk dari dalam sumber).

### N3. Alarm pergeseran (1,5 hari): `src/cxrseg/shift.py`
Definisi persis ada di PROTOCOL bagian 8. Fungsi yang diperlukan:

| Fungsi | Arti |
|---|---|
| `intensity_distance(img_u8, reference_hist)` | A1: jarak Wasserstein-1 antara histogram 256 bin citra dan histogram acuan, dihitung sebagai jumlah selisih mutlak CDF dibagi 255 |
| `pooled_hist(images_u8)` | Histogram gabungan citra latih (setelah skenario yang sama) |
| `mean_entropy(prob_u8)` | A2: rata-rata entropi biner `-(p ln p + (1-p) ln(1-p))` seluruh piksel, p dalam [0,1] |
| `alarm_threshold(val_scores, percentile=95)` | Ambang tau dari skor citra validasi dua sumber latih |
| `alarm_rate(scores, tau)` | Persentase citra dengan skor > tau |
| `readiness_label(failure_rate, median_dice, in_source_median_dice, cfg)` | "siap" bila gagal <= 5% **dan** turun median Dice <= 0,02 |
| `decision_from_alarm(rate, cfg)` | "siap" bila tingkat alarm <= 10% |

- **A1 dihitung dari citra setelah skenario**, jadi untuk S1, S2, S3 histogramnya ikut berubah. Pakai `cxrseg.preprocess.enhance`. Untuk S3 ambil parameter dari `results/tables/s3_params.csv` (pada prediksi palsu, buat berkas contoh sendiri).
- Histogram acuan hanya dari citra **latih** dua sumber latih. Ambang tau hanya dari citra **validasi** dua sumber latih. Sumber ditahan tidak dipakai di keduanya.
- **Tes yang wajib ada:** A1 = 0 untuk citra yang histogramnya sama dengan acuan, dan naik bila seluruh citra digeser terang. A2 = 0 untuk probabilitas 0 atau 1 dan maksimum (ln 2) di 0,5. Persentase alarm pada skor validasi itu sendiri sekitar 5% (karena tau persentil ke-95).
- **Selesai bila:** untuk tiap lipatan dan skenario kamu bisa mencetak skor A1 dan A2 per citra, tau, dan tingkat alarm tiap sumber.

### N4. Statistik (1 hari): `src/cxrseg/stats.py`
- `average_over_seeds(df)`: rata-rata metrik per citra lintas seed 11, 22, 33.
- `paired_comparison(df, baseline="S0", scenario, source)`: uji Wilcoxon berpasangan (`scipy.stats.wilcoxon`) pada Dice per citra, selisih rata-rata, dan CI 95% bootstrap (10.000 ulang, persentil, seed tetap).
- `holm(pvalues)`: koreksi Holm. Kamu menulis sendiri (15 baris), jangan tambah dependensi baru.
- `is_meaningful(p_holm, mean_diff)`: benar bila p terkoreksi < 0,05 **dan** |selisih| >= 0,01.
- Hal yang sama untuk tingkat kegagalan dan cakupan (selisih rata-rata dengan CI bootstrap, tanpa Wilcoxon).
- Total uji yang dikoreksi: 9 (3 skenario S1, S2, S3 dibanding S0, pada 3 sumber asing).
- **Tes yang wajib ada:** nilai Holm cocok dengan hitungan tangan untuk `[0.01, 0.04, 0.03]` (hasil 0,03, 0,06, 0,06). Bootstrap CI pada data sintetis dengan selisih 0,05 memuat 0,05. Selisih sepele (0,002) dengan p kecil tetap `is_meaningful == False`.

### N5. Analisis, tabel, gambar (2 hari): `scripts/analyze.py`
Satu perintah menghasilkan semuanya dari folder prediksi (palsu atau asli):

| Berkas | Isi | Menjawab |
|---|---|---|
| `T1_gap.csv` | Per lipatan x skenario: Dice median dalam sumber, Dice median asing, celah | RM1 |
| `T2_scenarios.csv` | Per sumber asing: selisih Dice S1/S2/S3 vs S0, CI, p terkoreksi, bermakna ya/tidak | RM2 |
| `T3_clinical.csv` | Tingkat gagal (ambang 0,90 dan sensitivitas 0,85 dan 0,93), cakupan, galat luas absolut median per skenario dan sumber asing | RM2 |
| `T4_tuning.csv` | Skor 12 kombinasi CLAHE per lipatan, dan jarak intensitas antar sumber untuk tiap kombinasi bila sempat | RM2 |
| `T5_alarm.csv` | Per sumber asing dan skenario: Spearman skor vs Dice, AUROC skor vs gagal (atau "tidak dapat dihitung"), tingkat alarm | RM3 |
| `T6_readiness.csv` | Tabel 2x2: label "siap" (dengan label) vs keputusan alarm (tanpa label), 12 kasus asing + 12 kontrol dalam sumber | RM3 |
| `T7_sample50.csv` | Sebaran tingkat alarm dari 1.000 sampel acak 50 citra per sumber | RM3 |

Gambar (PNG 300 dpi, font >= 8 pt, label sumbu berupa kata, satuan di kurung):
`F1_dice_by_scenario.png` (boxplot Dice per skenario, panel per sumber asing), `F2_gap.png` (celah generalisasi), `F3_alarm_scatter.png` (skor alarm vs Dice, titik gagal berwarna), `F4_clahe_grid.png` (heatmap 12 kombinasi), `F5_worst_cases.png` (tiap sumber asing: 3 citra dengan Dice terendah pada S0, GT, prediksi S0, prediksi CLAHE), `F6_histograms.png` (sebaran intensitas tiap sumber sebelum dan sesudah skenario).
- **Selesai bila:** `python scripts/analyze.py --pred-dir results/fake_predictions --out results/fake_analysis` jalan penuh. Lalu dengan `results/predictions` setelah Huda selesai.

### N6. Referensi dan cek angka (1 hari, minggu 4)
- Pastikan setiap referensi di proposal punya DOI yang terbuka, tahun 2021 sampai 2026, dan angka yang dikutip benar. Tabel referensi ada di `paper/proposal/REFERENSI.md`.
- Cek bahwa setiap angka di laporan Dylan bisa ditunjuk ke satu baris CSV. Tulis temuan sebagai daftar.

## Kriteria sukses Nabil

| Pemeriksaan | Ambang |
|---|---|
| QC | 3 gambar, 2 tabel, `QC-REPORT.md`, nol duplikat lintas sumber yang tidak dilaporkan |
| Evaluasi | 36 CSV metrik, jumlah baris sama dengan jumlah citra |
| Tes | Tiap modul punya tes. `python -m pytest -q` hijau |
| Analisis | 7 tabel dan 6 gambar terbuat dari satu perintah, dua kali dijalankan hasilnya identik (seed tetap) |
| Kejujuran | Tidak ada angka dari prediksi palsu di laporan. Tidak ada ambang yang disetel setelah melihat hasil asli |

## Jebakan yang sering terjadi

- Menghitung ambang alarm dari sumber yang ditahan. Itu kebocoran dan merusak seluruh klaim RM3.
- Memakai probabilitas tanpa pasca-pemrosesan untuk metrik. Metrik harus dari `postprocess_mask`.
- Menguji Wilcoxon per seed (menggandakan n secara palsu). Rata-ratakan seed dulu.
- AUROC dengan kurang dari 10 citra gagal itu tidak stabil. Tulis "tidak dapat dihitung" saja.
- Mengira p kecil berarti penting. Aturan efek minimum 0,01 Dice yang menentukan.
- Gambar tanpa label sumbu berupa kata atau tanpa satuan.
