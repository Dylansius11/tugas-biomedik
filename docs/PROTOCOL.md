# PROTOCOL: spesifikasi eksperimen yang dikunci

Dokumen ini satu-satunya sumber kebenaran untuk eksperimen. Kalau kode, proposal, dan dokumen ini berbeda, dokumen ini yang benar. Ubah isinya hanya lewat commit tersendiri dan kabari semua anggota.

Judul kerja: *Peran CLAHE dalam Kesiapan Segmentasi Paru Rontgen Thorax Lintas Rumah Sakit untuk Skrining Tuberkulosis*

## 1. Pertanyaan

| Kode | Pertanyaan | Dijawab oleh |
|---|---|---|
| RM1 | Seberapa besar performa U-Net turun saat diuji di sumber data yang tidak dilihat saat latihan? | Tabel celah generalisasi (bagian 7) |
| RM2 | Apakah CLAHE, bawaan maupun hasil tuning, mengubah performa dan tingkat kegagalan di sumber asing dibanding tanpa peningkatan kontras (S0) dan histogram equalization global (S1)? | Tabel perbandingan skenario + uji statistik (bagian 9) |
| RM3 | Apakah skor pergeseran per citra yang dihitung tanpa label bisa memperingatkan citra yang gagal, dan apakah hasilnya cukup untuk memutuskan "siap" atau "belum siap" di sumber baru? | Alarm + aturan kesiapan (bagian 8) |

Hasil "CLAHE tidak berpengaruh" tetap hasil yang sah. RM3 yang membuat hasil itu tetap berguna.

## 2. Data

Sumber: Danilov dkk. (2022), *Chest X-ray dataset for lung segmentation*, Mendeley Data, DOI 10.17632/8gf9vpkhgy.2, lisensi CC BY 4.0. Unduh dengan `scripts/download_data.sh`.

| Sumber | Di koleksi | Dipakai | Asal | Peran |
|---|---|---|---|---|
| Shenzhen | 566 | 566 | Shenzhen No.3 People's Hospital, Cina. Fokus TB | Inti skrining TB |
| Montgomery | 139 (deskripsi), 138 (berkas) | 138 | Program TB Montgomery County, AS. Fokus TB | Inti skrining TB |
| Darwin | 6.106 | 600 (acak, seed 42, mask tidak kosong) | Koleksi COVID-19 dan pneumonia viral (V7 Labs), alat dan kondisi pengambilan beragam, termasuk rontgen portabel | Uji tekanan: sumber yang sangat berbeda |

Catatan kejujuran: Darwin bukan koleksi TB. Klaim kita adalah kesiapan lintas sumber untuk segmentasi paru yang dipakai di skrining TB, bukan akurasi diagnosis TB.

Pemeriksaan wajib sebelum eksperimen (dikerjakan `scripts/prepare_data.py` dan `scripts/qc_data.py`):
1. Jumlah citra per sumber sesuai tabel di atas, setiap citra punya tepat satu mask.
2. Mask hanya 0 dan 255 setelah `binarize_mask` (mask asli membawa noise anti-aliasing sekitar 0,1% piksel bernilai 1 sampai 254, jadi dibinerkan di ambang 127), tidak kosong, luas mask wajar.
3. 20 sampel per sumber dicek visual (tumpang tindih citra dan mask).
4. Deteksi duplikat dekat antar sumber dengan perceptual hash. Duplikat lintas sumber membuat uji "asing" bocor, jadi harus dibuang dari sumber uji.

## 3. Skema pembagian: leave-one-source-out (LOSO)

- Pembagian dalam sumber: tiap sumber dibagi acak 80/10/10 (latih/validasi/uji) dengan seed 42. Disimpan di `data/processed/manifest.csv`, kolom `split`.
- Tiga lipatan, dinamai menurut sumber yang ditahan:

| Lipatan | Latih + validasi dari | Uji dalam sumber (in-source) | Uji asing (foreign) |
|---|---|---|---|
| `ho_darwin` | Montgomery + Shenzhen | bagian uji Montgomery + Shenzhen | seluruh Darwin (600) |
| `ho_montgomery` | Darwin + Shenzhen | bagian uji Darwin + Shenzhen | seluruh Montgomery (139) |
| `ho_shenzhen` | Darwin + Montgomery | bagian uji Darwin + Montgomery | seluruh Shenzhen (566) |

- Sumber yang ditahan tidak boleh dipakai untuk apa pun selain uji asing: tidak untuk validasi, tuning CLAHE, early stopping, ataupun menentukan ambang alarm.
- Metadata pasien tidak tersedia. Satu pasien bisa saja muncul di beberapa bagian pada pembagian dalam sumber. Risiko ini tidak menyentuh uji asing, yang menjadi dasar kesimpulan utama. Tulis sebagai batasan.

## 4. Skenario kontras

Urutan pra-pemrosesan, sama untuk semua skenario:

`baca citra -> grayscale uint8 -> resize 256x256 (cv2.INTER_AREA, mask pakai INTER_NEAREST) -> skenario -> bagi 255 (rentang 0 sampai 1)`

| Kode | Perlakuan pada citra uint8 256x256 |
|---|---|
| S0 | Tanpa peningkatan kontras |
| S1 | Histogram equalization global (`cv2.equalizeHist`) |
| S2 | CLAHE bawaan: `clipLimit=2.0`, `tileGridSize=(8, 8)` |
| S3 | CLAHE hasil tuning, parameter dipilih per lipatan (bagian 5) |

Semua citra di koleksi sudah berukuran 512x512 (terverifikasi saat `prepare_data.py`), jadi penurunan ke 256x256 tepat separuh. INTER_AREA dipilih karena paling bersih untuk penurunan ukuran; mask memakai nearest neighbor.

Augmentasi hanya geometris, diterapkan ke citra dan mask bersamaan, tidak mengubah kontras: rotasi sampai 10 derajat, skala 0,9 sampai 1,1, translasi 5%. Tanpa flip horizontal, karena posisi jantung membawa informasi.

## 5. Model, pelatihan, tuning

U-Net 4 tingkat (32, 64, 128, 256 filter), bottleneck 512, konvolusi 3x3, batch norm, ReLU, keluaran 1x1 dengan sigmoid. Masukan 1x256x256.

| Komponen | Nilai |
|---|---|
| Loss | BCE + Dice loss, bobot sama |
| Optimizer | Adam, lr 0,001, dibagi dua jika Dice validasi tidak membaik 5 epoch |
| Batch / epoch | 16 / maksimal 60, early stopping setelah 10 epoch tanpa perbaikan |
| Seed | 11, 22, 33 |
| Pemilihan checkpoint | Dice validasi tertinggi (ambang 0,5, tanpa pasca-pemrosesan) |
| Presisi | Mixed precision (AMP) boleh dipakai |

Tuning S3 (per lipatan, hanya data validasi dari dua sumber latih):
- Grid `clipLimit` {1, 2, 3, 4} x `tileGridSize` {4x4, 8x8, 16x16} = 12 kombinasi.
- Tiap kombinasi dilatih 20 epoch dengan satu seed (11). Pemenang = Dice validasi tertinggi.
- Pemenang dipakai untuk pelatihan penuh S3 (3 seed). Simpan seluruh 12 skor di `results/tables/tuning_{fold}.csv`. Tabel ini juga bahan analisis sensitivitas parameter (RM2).

Total pelatihan: 36 penuh (4 skenario x 3 lipatan x 3 seed) + 36 tuning singkat. Opsional (P2): Attention U-Net pada S0 dan skenario CLAHE terbaik, 1 seed, 3 lipatan.

## 6. Pasca-pemrosesan dan kontrak keluaran

Dikerjakan di `cxrseg.postprocess.postprocess_mask`: ambang 0,5, pertahankan dua komponen terbesar, closing 5x5, isi lubang. Model menyimpan probabilitas mentah, pasca-pemrosesan dilakukan saat evaluasi.

Kontrak berkas keluaran pelatihan (jangan diubah tanpa kabar):

```
results/checkpoints/{fold}/{scenario}/seed{seed}/best.pt
results/predictions/{fold}/{scenario}/seed{seed}/pred.npz
    ids      : array str, (N,)         id citra, format "{source}/{name}"
    group    : array str, (N,)         "val", "in_source" atau "foreign"
    prob_u8  : array uint8, (N,256,256) probabilitas sigmoid x 255, dibulatkan
results/metrics/{fold}/{scenario}/seed{seed}.csv      (dibuat evaluate.py)
results/tables/tuning_{fold}.csv                      (dibuat tuning)
results/tables/s3_params.csv   kolom: fold, clip_limit, tile_grid
```

`fold` salah satu dari `ho_darwin`, `ho_montgomery`, `ho_shenzhen`. `scenario` salah satu dari `S0`, `S1`, `S2`, `S3`.

## 7. Metrik (semua per citra pada 256x256, setelah pasca-pemrosesan)

| Metrik | Definisi | Alasan |
|---|---|---|
| Dice (utama) | 2\|P∩G\| / (\|P\|+\|G\|) | Standar segmentasi |
| IoU | \|P∩G\| / \|P∪G\| | Pelengkap Dice |
| HD95 (piksel) | Persentil ke-95 jarak batas dua arah. Jika prediksi kosong: 362 (diagonal citra) | Kesalahan batas yang tidak terlihat di Dice (Metrics Reloaded) |
| Cakupan paru | \|P∩G\| / \|G\| | Bagian paru yang hilang adalah bagian yang tidak diperiksa lesi TB-nya |
| Galat luas (%) | 100 x (\|P\|-\|G\|) / \|G\|, bertanda | Dampak pada pengukuran turunan dan potongan ROI |
| Gagal | Dice < 0,90 | Jaring pengaman klinis. Analisis kepekaan di 0,85 dan 0,93 |

Skor per citra dirata-ratakan lintas seed sebelum uji statistik. Celah generalisasi = Dice median uji dalam sumber minus Dice median uji asing, per lipatan dan skenario.

Dice pada segmentasi paru sudah mendekati batas atas, jadi selisih antar skenario bisa kecil. Karena itu cakupan, galat luas, dan tingkat kegagalan dilaporkan setara, tidak sebagai pelengkap.

## 8. Alarm pergeseran dan aturan kesiapan (RM3)

Jawaban untuk pertanyaan "kapan model dinyatakan siap di rumah sakit baru?". Ada dua tahap: definisi "siap" yang butuh label (hanya untuk eksperimen kita), dan alarm tanpa label (yang nanti dipakai rumah sakit baru).

**Definisi "siap" (butuh label, dipakai untuk validasi):** pada satu sumber uji dan satu skenario, model **siap** bila tingkat kegagalan <= 5% **dan** Dice median turun paling banyak 0,02 dari uji dalam sumber. Selain itu **belum siap**.

**Skor alarm per citra (tanpa label):**
- `A1` jarak intensitas: jarak Wasserstein-1 antara histogram 256 bin citra itu (setelah skenario, sebelum dibagi 255) dan histogram gabungan citra latih pada skenario yang sama. Dihitung dengan selisih CDF.
- `A2` ketidakpastian model: rata-rata entropi biner piksel peta probabilitas. Pembanding berbasis model.

**Ambang:** tau = persentil ke-95 skor pada citra validasi dua sumber latih. Citra dengan skor > tau diberi alarm. **Tingkat alarm** = persentase citra yang beralarm pada satu sumber.

**Keputusan tanpa label:** tingkat alarm <= 10% berarti siap, di atas itu belum siap dan perlu validasi lokal dengan sampel kecil berlabel. Angka 5% dan 10% adalah konvensi awal yang dikunci sebelum melihat hasil, bukan klaim klinis. Analisis kepekaan boleh dilaporkan terpisah.

**Evaluasi alarm:**
1. Per citra, pada tiap sumber asing: korelasi Spearman antara skor dan Dice. AUROC skor untuk memprediksi citra gagal (hanya jika ada minimal 10 citra gagal, kalau kurang dilaporkan "tidak dapat dihitung").
2. Per sumber dan skenario: tabel 2x2 antara "siap/belum siap" (dengan label) dan keputusan alarm (tanpa label). Ada 12 kasus asing, ditambah 12 kasus uji dalam sumber sebagai kontrol. Dilaporkan deskriptif, karena n kecil.
3. Berapa citra tanpa label yang cukup: 1.000 kali ambil sampel acak 50 citra dari satu sumber, laporkan sebaran tingkat alarmnya.
4. Efek CLAHE: bandingkan A1 antar sumber sebelum dan sesudah tiap skenario. Apakah CLAHE benar-benar mendekatkan sebaran antar sumber?

## 9. Statistik

- Satuan analisis: citra (skor sudah dirata-ratakan lintas seed).
- Perbandingan S1, S2, S3 terhadap S0 pada tiap sumber asing: uji Wilcoxon berpasangan, koreksi Holm atas 9 uji (3 skenario x 3 sumber).
- Besaran efek: selisih rata-rata Dice dan selang kepercayaan 95% bootstrap (10.000 ulang). Hal yang sama untuk tingkat kegagalan dan cakupan.
- Bermakna bila p terkoreksi < 0,05 **dan** |selisih rata-rata Dice| >= 0,01. Dengan ratusan citra, uji saja akan menandai selisih yang terlalu kecil untuk berarti, jadi ambang efek wajib dipakai.
- Hubungan jarak intensitas dengan celah generalisasi hanya tiga pasang sumber: dilaporkan deskriptif, tanpa uji korelasi.

## 10. Reproduksibilitas

- Seed dikunci (`torch`, `numpy`, `random`, DataLoader worker). Catat versi pustaka di `results/env.txt`.
- Semua parameter di `configs/default.yaml`. Skrip tidak boleh punya angka bawaan yang berbeda.
- Pelatihan harus bisa dilanjutkan setelah sesi Colab putus (simpan checkpoint terakhir tiap epoch, lewati run yang `pred.npz`-nya sudah ada).
- Repositori + `scripts/download_data.sh` + `scripts/prepare_data.py` cukup untuk mengulang dari nol.
