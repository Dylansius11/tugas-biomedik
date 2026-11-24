# STATE-PERCOBAAN-1: catatan keadaan percobaan utama (LOSO, 36 pelatihan)

Dicatat oleh Huda. Tujuan berkas ini: satu tempat yang merekam apa yang sudah dijalankan, apa hasil awalnya, apa yang belum diketahui, dan apa yang diusulkan berikutnya, supaya Dylan dan Nabil bisa melanjutkan tanpa menebak.

**Status angka:** semua angka hasil di bawah adalah analisis **eksploratif** Huda dari 36 `pred.npz` asli, dihitung dengan skrip sementara. Ini **bukan** tabel resmi. Tabel resmi untuk laporan harus berasal dari `scripts/analyze.py` (Nabil) dan dicocokkan dengan angka di sini. Jangan mengutip angka berkas ini langsung di laporan.

## 1. Apa yang sudah dijalankan

| Hal | Isi |
|---|---|
| Spesifikasi | `docs/PROTOCOL.md` tanpa perubahan |
| Cabang | `huda/train`, 10 commit di atas `main` (e255874), kepala e6c84bb, belum di-push |
| Lingkungan | Python 3.12.10, torch 2.6.0+cu124, RTX 4060 8 GiB, lihat `results/env.txt` |
| Data | `scripts/download_data.sh` lalu `scripts/prepare_data.py`, jumlah sesuai README: Darwin 600 (13 mask kosong dibuang sebelum sampling), Montgomery 138, Shenzhen 566 |
| Model | U-Net 32/64/128/256, bottleneck 512, 7.762.465 parameter (`src/cxrseg/model.py`) |
| Latihan | BCE + Dice loss, Adam lr 0,001 (dibagi dua tiap 5 epoch tanpa perbaikan), batch 16, maksimal 60 epoch, early stopping 10, AMP fp16, seed 11/22/33 |
| Augmentasi | rotasi 10 derajat, skala 0,9 sampai 1,1, translasi 5%, tanpa flip, tanpa ubah kontras (`cv2.warpAffine`) |
| Tuning CLAHE | 12 kombinasi x 3 lipatan, 20 epoch, seed 11. Hasil di `results/tables/tuning_*.csv` dan `s3_params.csv` |
| Pelatihan penuh | `scripts/run_all.py`: 36 dari 36 `pred.npz`. Run penuh berlangsung beberapa jam di GPU lokal (sekitar 5 detik per epoch) |
| Tes | `python -m pytest -q`: 32 lolos |

Ukuran data per lipatan (setelah pembagian 80/10/10 dalam sumber):

| Lipatan | Foto latih | Validasi | Uji dalam sumber | Uji asing |
|---|---|---|---|---|
| `ho_darwin` | 562 | 71 | 71 | 600 |
| `ho_montgomery` | 932 | 117 | 117 | 138 |
| `ho_shenzhen` | 590 | 74 | 74 | 566 |

Parameter S3 hasil tuning: `ho_darwin` clip 4 tile 4x4, `ho_montgomery` clip 3 tile 8x8, `ho_shenzhen` clip 2 tile 8x8. Selisih antar 12 kombinasi hanya 0,001 sampai 0,004 Dice validasi (satu seed), jadi pemenang kemungkinan dalam derau.

Reproduksi: dua run seed sama (`ho_shenzhen`, S0, seed 11, 20 epoch) memberi Dice in-source identik (selisih 0,0).

## 2. Hasil awal (eksploratif)

Metrik: Dice per citra setelah pasca-pemrosesan, dirata-ratakan lintas 3 seed. "Gagal" = Dice < 0,90.

### RM1: celah generalisasi (S0)

| Sumber asing | Dice median dalam sumber | Dice median asing | Celah | Gagal | Cakupan paru | Galat luas median |
|---|---|---|---|---|---|---|
| Darwin | 0,975 | 0,773 | 0,202 | 97,3% | 0,62 | -34% |
| Montgomery | 0,966 | 0,945 | 0,021 | 6,5% | 0,96 | +4% |
| Shenzhen | 0,966 | 0,900 | 0,066 | 50,5% | 0,98 | +18% |

### RM2: selisih Dice terhadap S0 pada sumber asing (Wilcoxon, Holm atas 9 uji, efek minimum 0,01)

| Sumber | S1 (HE) | S2 (CLAHE bawaan) | S3 (CLAHE tuning) |
|---|---|---|---|
| Darwin | +0,004 (tidak bermakna) | +0,034 bermakna | +0,014 bermakna |
| Montgomery | -0,025 bermakna (merugikan) | +0,005 tidak bermakna | +0,005 tidak bermakna (p Holm 0,052) |
| Shenzhen | +0,002 tidak bermakna | -0,002 tidak bermakna | -0,002 tidak bermakna |

Tingkat gagal Darwin hampir tidak berubah antar skenario (97,3% S0 ke 97,0% S2). Tingkat gagal Montgomery: S0 6,5%, S1 21,7%, S2 4,3%, S3 3,6%.

**Catatan penting:** pada `ho_shenzhen`, S3 memakai parameter yang sama dengan S2 (clip 2, tile 8x8), dan prediksinya **identik** (dicek dengan `np.array_equal`). Dua dari sembilan uji di sana bukan dua hasil independen. Tulis di Metode dan Batasan.

### RM3: alarm (ambang persentil 95 skor validasi, dikunci sebelum hasil)

- Tingkat alarm pada data dalam sumber sekitar 5% (A1 4,85%, A2 4,61%), sesuai rancangan.
- Pada sumber asing alarm tidak menangkap kegagalan:

| Sumber (S0) | Alarm A1 | Alarm A2 | Spearman A1 / A2 dengan Dice | AUROC A1 / A2 |
|---|---|---|---|---|
| Darwin | 7,2% | 18,2% | -0,18 / -0,10 | 0,51 / 0,53 |
| Montgomery | 21,0% | 2,2% | +0,19 / -0,47 | tidak dapat dihitung (9 gagal) |
| Shenzhen | 43,5% | 0,5% | 0,00 / -0,29 | 0,52 / 0,64 |

- Pada 12 kasus asing, dari 10 kasus yang sebenarnya "belum siap", A1 menyatakan "siap" di 7 kasus dan A2 di 6 kasus. Kasus "siap" yang sebenarnya hanya dua (Montgomery S2 dan S3).
- Skor entropi A2 dihitung dari seed 11 saja dalam analisis ini. `shift.py` harus menentukan cara menggabungkan seed.

## 3. Temuan tentang Darwin

Fakta yang diukur:

- Model yang ikut belajar dari Darwin bagus pada foto Darwin lain: Dice median 0,96 dan hanya 2 sampai 3% gagal (dari `in_source` di `ho_montgomery` dan `ho_shenzhen`). Jadi foto Darwin, termasuk yang COVID, bukan sesuatu yang sulit digambar bagi model yang pernah melihat gayanya.
- Statistik mask (median): luas paru Darwin 32% dari frame (Montgomery 26%, Shenzhen 25%). Tepi bawah paru di 0,91 tinggi frame (Montgomery 0,72, Shenzhen 0,81).
- Pada Darwin model memprediksi luas median 21%, lebih kecil dari kebiasaan data latih (sekitar 25%). Bagian yang terlewat berada di tengah sampai bawah paru (posisi baris rata-rata 0,56).
- Pemeriksaan visual 8 foto per sumber (`seed 1`): garis tandaan Darwin tampak lebih luas daripada prediksi.

Dugaan, **belum terbukti**:

- Perbedaan konvensi anotasi (mask Darwin mungkin mencakup area di belakang jantung dan lebih ke bawah). Sebuah studi lain tentang koleksi V7 Labs COVID-19 melaporkan bahwa maskernya buatan anotator manusia dan mencakup sebagian besar jantung. Itu dari sumber sekunder dan harus diverifikasi ke makalah Danilov dkk. (2022).
- Perbedaan cara pengambilan foto dan pembingkaian (paru lebih besar dan lebih rendah di frame, termasuk rontgen portabel).
- Jangkauan augmentasi geometris yang sempit (skala 0,9 sampai 1,1, geser 5%) membuat paru Darwin berada di luar apa yang pernah dipelajari model.
- Ketiganya tidak bisa dipisahkan dari data yang ada sekarang.

Dampak untuk penulisan: hasil Darwin sebaiknya dibahas sebagai "model dari sumber TB tidak siap pada data yang sangat berbeda (termasuk kemungkinan perbedaan konvensi anotasi)", bukan "model gagal pada penyakit COVID". Tugas segmentasi tidak bergantung pada jenis penyakit.

## 4. Pertanyaan terbuka

1. Apakah celah Darwin terutama karena **konvensi anotasi**, **pembingkaian foto**, atau **intensitas dan alat**? Belum bisa dipisahkan.
2. Apakah jumlah dan keragaman data latih memengaruhi hasil? Tiga lipatan membingungkan faktor ini: `ho_montgomery` (latih 932, termasuk Darwin yang beragam) hasilnya terbaik, `ho_darwin` (latih 562, dua sumber TB yang mirip) terburuk, `ho_shenzhen` (latih 590) di tengah. Dengan tiga titik, ukuran data dan kemiripan sumber tidak bisa dipisahkan. Perlu kurva belajar (latih dengan 25%, 50%, 100% data latih) untuk memisahkannya.
3. Alarm A1 dan A2 gagal pada data ini. Apakah ukuran yang lebih dekat ke tugas (bentuk dan letak mask prediksi) bisa jadi alarm yang lebih baik?
4. Apakah beberapa kegagalan Shenzhen karena prediksi terlalu besar (mencakup lambung) atau karena tandaan yang berbeda?

## 5. Rencana berikutnya (diusulkan, disetujui Dylan secara prinsip, urutan usulan)

Semua ini **eksperimen tambahan yang diberi label terpisah**. Hasil utama 36 run tidak diubah, dan ambang A1/A2 asli tetap dilaporkan apa adanya. Aturan emas tetap berlaku: sumber asing tidak dipakai untuk memilih apa pun di eksperimen utama.

| Kode | Isi | Mengapa | Catatan aturan |
|---|---|---|---|
| X1 | Statistik dan visualisasi mask lebih luas antar sumber (luas, letak, bentuk, bagian yang terlewat) | Memisahkan dugaan anotasi vs pembingkaian | Analisis saja |
| X2 | Kurva belajar: latih dengan 25%, 50%, 100% data latih per lipatan | Memisahkan efek ukuran data vs kemiripan sumber | Hanya data latih sumber yang terlihat |
| X3 | Augmentasi geometris lebih lebar sebagai lengan tambahan | Menguji apakah celah Darwin terkait skala dan letak paru | Mengubah nilai di config hanya untuk lengan ini, jangan menimpa `default.yaml` utama |
| X4 | Sampel kecil berlabel dari sumber baru (10, 25, 50 foto) untuk penyesuaian, uji pada sisanya | Menjawab "berapa foto berlabel yang cukup agar siap" | Pembagian sumber asing menjadi adaptasi dan uji harus ditetapkan sebelum melihat hasil |
| X5 | Alarm tambahan berbasis bentuk dan letak mask prediksi (luas, jumlah komponen, posisi tepi bawah dibanding data latih) dan penjelasan faktor penyebab alarm | Alarm yang memberi alasan, bukan hanya siap atau belum | Ambang dari validasi sumber latih saja. Dilabeli post-hoc |
| X6 | Attention U-Net (P2 di rencana awal) | Pemeriksaan arsitektur | Satu seed, tiga lipatan |

Dataset lain yang mungkin dipakai (**belum diverifikasi lisensi dan kecocokannya**, perlu dicek sebelum memutuskan):

- JSRT dengan SCR (247 foto PA, mask paru buatan manual), kecil dan lama.
- COVID-QU-Ex (sekitar 33 ribu foto dengan mask paru), asal gabungan banyak koleksi sehingga gaya mask campur. Cara pembuatan mask dan lisensinya belum ditemukan.
- CheXmask (ratusan ribu mask, tapi dibuat otomatis oleh model dan tidak termasuk citra). Tidak cocok sebagai "kebenaran" tanpa kehati-hatian.
- Menambah sumber mengubah desain LOSO dan jumlah pelatihan, jadi perlu keputusan Dylan.

## 6. Cara mengulang

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
bash scripts/download_data.sh && python scripts/prepare_data.py --zip
python -m cxrseg.tune_clahe --fold ho_darwin   # ulangi untuk ho_montgomery dan ho_shenzhen
python scripts/run_all.py
python scripts/check_runs.py                    # harus 36 dari 36
```

Berkas keluaran tidak masuk git: `data/`, `results/checkpoints`, `results/predictions`. Yang masuk git: `results/tables/tuning_*.csv`, `s3_params.csv`.
