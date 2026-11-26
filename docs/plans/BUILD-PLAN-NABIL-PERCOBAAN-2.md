# BUILD-PLAN: Nabil, analisis Percobaan 2 skenario A (post-hoc, analisis tambahan)

Tugasmu satu kalimat: **periksa ulang angka Percobaan 2 Huda dengan alat yang sama dengan hasil utama, lalu jawab pertanyaan yang belum dijawab: apakah alarm A1/A2 lebih berguna pada model dua head (E2).**

Status: **dikerjakan 26 November 2026 di cabang `nabil/exp2-a`** atas instruksi Dylan memakai data Percobaan 2. Ini skenario A (Percobaan 2 sebagai tambahan). Skenario B (hasil utama) ada di `nabil/exp2-b`, lihat `docs/plans/BRANCHING-PERCOBAAN-2.md`. Percobaan 2 bersifat eksploratif dan post-hoc (`docs/STATE-PERCOBAAN-2.md`). Hasil utama (36 run, `results/tables` T1–T7) tidak diubah.

## Syarat mulai

- Dylan meminta analisis memakai data Percobaan 2 (terpenuhi). Posisinya di laporan (A atau B) belum diputuskan.
- Huda mengirim prediksi Percobaan 2 (terpenuhi, `E:\...\code\percobaan2`, 87 berkas):
  - `results/exp2/E1/{fold}/seed{11,22,33}/pred.npz` dan `val_dice.txt` (9 run)
  - `results/exp2/E2/{fold}/seed{11,22,33}/pred.npz` dan `val_dice.txt` (9 run)
  - `results/exp2/fewshot/E2/{fold}/k{5,10,20,50}_seed{s}.npz` (36 adaptasi)

## Aturan

1. Semua luaran ke `results/exp2/nabil/`. Tidak menimpa `results/tables`, `results/figures`, `results/metrics`.
2. Semua angka diberi label **post-hoc**.
3. Ambang tau hanya dari citra validasi dua sumber latih. Sumber ditahan tidak dipakai untuk ambang atau pemilihan apa pun.
4. `scripts/analyze.py` tidak diubah. Analisis memakai skrip baru `scripts/analyze_exp2.py` yang memanggil `cxrseg.evaluate`, `cxrseg.shift`, `cxrseg.stats`.
5. Pasca-pemrosesan dan definisi metrik sama persis dengan hasil utama (`postprocess_mask`, `evaluate_pair`, gagal = Dice < 0,90).

## Langkah

### N7. Terima dan validasi data (0,5 hari)
- 18 `pred.npz` (E1, E2 x 3 lipatan x 3 seed) dan 36 berkas few-shot ada.
- Kunci `ids`, `group`, `prob_u8`, `head`. Urutan `ids` dan `group` sama dengan prediksi utama.
- Kolom `head`: pada `ho_darwin` head yang dipakai harus `visible` (head `anatomical` tidak pernah dilatih).
- **Selesai bila:** 54 berkas valid, hasil pengecekan dicatat di `X0_validation.csv`.

### N8. Verifikasi angka Huda (0,5 hari)
- Evaluasi E1 dan E2 dengan `cxrseg.evaluate`.
- Cocokkan dengan `results/exp2/tables/main.csv`, `vs_E0.csv`, `fewshot.csv` (Dice median, tingkat gagal, cakupan, selisih vs E0), dan baris E0 alarm dengan T5 S0 hasil utama. Luaran `X1_verify.csv`.
- **Selesai bila:** semua angka cocok sampai 3 desimal, atau setiap selisih dijelaskan.

### N9. Statistik E2 vs E0 sesuai protokol (0,5 hari)
- Wilcoxon berpasangan pada Dice per citra (rata-rata 3 seed dulu), CI bootstrap 10.000 ulang, koreksi Holm atas 3 uji (3 sumber asing), bermakna bila p Holm < 0,05 dan |selisih| >= 0,01.
- Selisih tingkat gagal dan cakupan dengan CI bootstrap.
- Ringkasan klinis seperti T3: gagal pada 0,85 / 0,90 / 0,93, galat luas absolut median.
- **Selesai bila:** `X2_e2_vs_e0.csv` dan `X3_clinical.csv` terbentuk.

### N10. Alarm A1/A2 pada E2 (1 hari)
- A1 hanya bergantung pada citra (S0), jadi skornya sama dengan E0 S0. Yang berubah adalah label siap/belum siap karena E2 lebih baik. Pertanyaan: apakah keputusan A1 lebih sering benar?
- A2 dihitung ulang dari probabilitas E2, rata-rata 3 seed per citra (aturan sama dengan hasil utama), tau persentil 95 dari validasi.
- Spearman skor vs Dice, AUROC skor vs gagal (atau "tidak dapat dihitung" bila < 10 gagal atau < 10 lulus), tabel 2x2 siap vs keputusan alarm, sebaran tingkat alarm 1.000 sampel acak 50 citra.
- Bandingkan dengan A3 Huda (`results/exp2/tables/xai_alarm.csv`) dalam satu tabel.
- **Selesai bila:** `X4_alarm.csv`, `X5_readiness.csv`, `X7_sample50.csv` terbentuk dan ada satu kalimat jawaban.

### N11. Few-shot (0,5 hari)
- Kurva k = 0, 5, 10, 20, 50 vs tingkat gagal dan Dice median per sumber, dengan rentang antar seed.
- Catat: evaluasi hanya pada val+test sumber baru (bukan seluruh sumber), tiga seed few-shot tidak saling bebas.
- **Selesai bila:** `X6_fewshot.csv` dan gambar kurva terbentuk.

### N12. Tabel, gambar, dokumen (1 hari)
- Gambar (PNG 300 dpi, label sumbu berupa kata, satuan di kurung): `XF1` boxplot Dice E0/E1/E2 per sumber asing, `XF2` kurva few-shot, `XF3` scatter A1/A2 vs Dice pada E2, `XF4` tiga kasus terburuk E2 per sumber.
- Satu perintah `python scripts/analyze_exp2.py --mode a --pred-dir ... --exp2-dir ... --out results/exp2/nabil` menghasilkan semuanya. Dua kali jalan, CSV identik.
- Tes `tests/test_analyze_exp2.py` dengan data sintetis.
- `docs/HASIL-NABIL-PERCOBAAN-2.md` (1–2 halaman, berlabel post-hoc). Perbarui `docs/STATE-NABIL.md`.
- **Selesai bila:** semua luaran ada, `python -m pytest -q` hijau, commit di `nabil/exp2-a`.

## Ringkasan

| Langkah | Isi | Waktu | Menunggu |
|---|---|---|---|
| N7 | Validasi data | 0,5 hari | Data Huda |
| N8 | Verifikasi angka Huda | 0,5 hari | N7 |
| N9 | Statistik E2 vs E0 | 0,5 hari | N8 |
| N10 | Alarm A1/A2 pada E2 | 1 hari | N8 |
| N11 | Few-shot | 0,5 hari | N7 |
| N12 | Tabel, gambar, dokumen | 1 hari | N9–N11 |

## Kriteria sukses

| Pemeriksaan | Ambang |
|---|---|
| Verifikasi | Angka Huda cocok atau selisihnya dijelaskan |
| Alarm | Pertanyaan "apakah alarm membaik pada E2" terjawab dengan angka |
| Pemisahan | Tidak ada berkas hasil utama yang berubah |
| Kejujuran | Semua angka berlabel post-hoc, tidak ada ambang yang disetel setelah melihat hasil |

## Jebakan

- Menyebut E2 sebagai hasil utama.
- Memilih head untuk Darwin berdasarkan hasil uji.
- Menyetel ulang tau setelah melihat hasil.
- Membaca kurva few-shot sebagai bukti kuat (seed tidak saling bebas, satu sumber per lipatan).
