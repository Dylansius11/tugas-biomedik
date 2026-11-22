# BUILD-PLAN: Huda (machine learning)

Tugasmu satu kalimat: **ubah `data/processed` menjadi 36 model terlatih dan berkas prediksi `pred.npz`**, tanpa pernah menyentuh sumber yang sedang ditahan.

Yang sudah disediakan Dylan (jangan ditulis ulang, tinggal dipakai):
`cxrseg.preprocess` (skenario S0 sampai S3), `cxrseg.data` (pembagian lipatan), `cxrseg.postprocess`, `cxrseg.metrics`, `configs/default.yaml`.
Yang kamu tulis: `model.py`, `dataset.py`, `train.py`, `tune_clahe.py`, `scripts/run_all.py`, notebook Colab.

Baca dulu `docs/PROTOCOL.md` bagian 3, 4, 5, 6. Itu aturan mainnya.

## Hasil akhir yang harus ada

```
results/checkpoints/{fold}/{scenario}/seed{seed}/best.pt     36 berkas
results/predictions/{fold}/{scenario}/seed{seed}/pred.npz    36 berkas
results/tables/tuning_{fold}.csv                              3 berkas (12 baris tiap berkas)
results/tables/s3_params.csv                                  kolom: fold, clip_limit, tile_grid
results/env.txt                                               versi python, torch, GPU
```

## Langkah

### H0. Lingkungan (0,5 hari)
1. `git clone`, buat cabang `huda/train`.
2. Colab: Runtime GPU T4. Lokal: `pip install -r requirements.txt` dengan build torch yang cocok dengan CUDA-mu.
3. `pip install -e .`, lalu `python -m pytest -q` harus hijau (20 tes).
4. Unduh `processed.zip` dari Drive yang dibagikan Dylan, ekstrak ke `data/processed`. Cek `manifest.csv` ada dan jumlahnya sesuai (lihat README).

### H1. Model (0,5 hari): `src/cxrseg/model.py`
- U-Net sesuai PROTOCOL bagian 5: encoder 32, 64, 128, 256, bottleneck 512, decoder simetris dengan skip connection berupa konkatenasi, konvolusi 3x3 + BatchNorm + ReLU (dua per blok), keluaran konvolusi 1x1.
- Keluarkan **logit** dari model dan terapkan sigmoid di luar (lebih stabil untuk BCEWithLogits).
- Cetak jumlah parameter. Perkiraan wajar 7 sampai 9 juta. Laporkan angkanya ke Dylan.
- **Selesai bila:** masukan `(2,1,256,256)` menghasilkan keluaran `(2,1,256,256)`, dan model bisa overfit 16 citra sampai Dice latih > 0,98 dalam kurang dari 150 iterasi. Kalau tidak bisa, ada bug. Jangan lanjut.

### H2. Dataset (0,5 hari): `src/cxrseg/dataset.py`
- Pakai `cxrseg.data.fold_split` dan `load_arrays`. Jangan membaca berkas sendiri dengan cara lain.
- Terapkan `cxrseg.preprocess.preprocess(img, scenario, clahe)` **sekali** per citra dan simpan hasilnya di memori (1.300 citra x 256 x 256 hanya puluhan MB). Setelah itu baru augmentasi geometris pada citra dan mask bersamaan: rotasi sampai 10 derajat, skala 0,9 sampai 1,1, translasi 5%, isi tepi 0. **Tanpa flip.** Tanpa perubahan kecerahan atau kontras.
- Validasi dan uji tidak diaugmentasi.
- Parameter S3 dibaca dari `results/tables/s3_params.csv`, dan S3 tanpa berkas itu harus gagal dengan pesan jelas.
- **Selesai bila:** kamu menampilkan 8 pasangan citra dan mask hasil augmentasi, tepinya masih sejajar, dan tampilan S0, S1, S2 tampak berbeda kontrasnya.

### H3. Pelatihan (1 hari): `src/cxrseg/train.py`
Perintah: `python -m cxrseg.train --fold ho_shenzhen --scenario S0 --seed 11`
- Semua angka dari `configs/default.yaml`: loss = BCE + Dice loss bobot sama, Adam lr 0,001, dibagi dua bila Dice validasi tidak membaik 5 epoch, batch 16, maksimal 60 epoch, early stopping 10 epoch.
- Dice validasi per epoch dihitung dengan ambang 0,5 **tanpa** pasca-pemrosesan, pada validasi dua sumber latih.
- Simpan `best.pt` (Dice validasi terbaik) dan `last.pt` tiap epoch supaya bisa dilanjutkan.
- Kunci seed (`random`, `numpy`, `torch`, worker DataLoader), dan atur `cudnn.deterministic=True`.
- Di akhir, muat `best.pt`, hitung probabilitas untuk grup `val`, `in_source`, `foreign` (urutan dan id sesuai `fold_split`), simpan `pred.npz` sesuai kontrak PROTOCOL bagian 6.
- Bila `pred.npz` untuk kombinasi itu sudah ada, lewati. Bila `last.pt` ada tapi `pred.npz` belum, lanjutkan dari sana.
- **Aturan emas:** sumber yang ditahan hanya muncul saat membuat prediksi `foreign`. Tidak untuk validasi, tidak untuk statistik normalisasi, tidak untuk apa pun.

**Smoke test wajib** (jalankan sebelum yang lain): `ho_shenzhen`, S0, seed 11, maksimal 20 epoch. Laporkan ke grup:
- detik per epoch dan memori GPU (supaya kita bisa memperkirakan total waktu 36 pelatihan),
- Dice validasi terbaik,
- bentuk dan isi `pred.npz`.

### H4. Tuning CLAHE (1 hari): `src/cxrseg/tune_clahe.py`
Perintah: `python -m cxrseg.tune_clahe --fold ho_darwin`
- 12 kombinasi: `clip_limit` {1, 2, 3, 4} x `tile_grid` {4x4, 8x8, 16x16}.
- Tiap kombinasi dilatih 20 epoch, seed 11, dengan aturan latihan yang sama seperti H3. Skor = Dice validasi terbaik pada dua sumber latih.
- Tulis `results/tables/tuning_{fold}.csv` (kolom: `clip_limit, tile_grid, val_dice`) dan perbarui baris lipatan itu di `results/tables/s3_params.csv`.
- Yang diuji hanya validasi dari sumber latih. Sumber yang ditahan tidak dipakai sama sekali.
- **Selesai bila:** tiga CSV tuning ada, tiap CSV 12 baris, dan `s3_params.csv` punya 3 baris. Kirim ke Dylan dan Nabil.

### H5. 36 pelatihan penuh (2 sampai 3 hari komputasi): `scripts/run_all.py`
- Urutan: S0, S1, S2 untuk ketiga lipatan dan seed 11, 22, 33 (27 run) boleh jalan sambil menunggu tuning. S3 (9 run) menyusul setelah `s3_params.csv` lengkap.
- Skrip melewati run yang sudah selesai, jadi aman dijalankan ulang setelah Colab putus.
- Taruh `results/` di Google Drive (symlink atau mount) supaya checkpoint tidak hilang saat sesi berakhir.
- Cek kemajuan: `python scripts/check_runs.py` menampilkan tabel run yang sudah ada dan yang belum.
- **Selesai bila:** `check_runs.py` menunjukkan 36 dari 36 `pred.npz`, dan `results/env.txt` tersimpan.

### H6. Opsional (P2): Attention U-Net
Hanya setelah H5 selesai. Satu seed, tiga lipatan, S0 dan skenario CLAHE terbaik (6 run). Simpan di `results/predictions_attn/` dengan struktur yang sama. Dylan memutuskan apakah dipakai.

## Kriteria sukses Huda

| Pemeriksaan | Ambang |
|---|---|
| Overfit 16 citra | Dice latih > 0,98 |
| Smoke test, Dice validasi | >= 0,90 (di bawah itu: cari bug, jangan lanjut) |
| Run diulang dengan seed sama | Dice uji dalam sumber beda <= 0,005 (GPU tidak selalu persis sama, tetapi harus dekat) |
| Kelengkapan | 36 `pred.npz`, 3 `tuning_*.csv`, `s3_params.csv`, `env.txt` |
| Kebocoran | Tidak ada baris kode yang membaca sumber ditahan sebelum tahap prediksi `foreign` |
| Tes | `python -m pytest -q` tetap hijau setelah semua commit |

## Jebakan yang sering terjadi

- Mask terbalik atau tidak 0/1: dataset sudah memberi 0 dan 255, bagi 255 dan ubah ke float.
- Lupa `model.eval()` dan `torch.no_grad()` saat membuat prediksi. BatchNorm akan memberi hasil acak.
- `best.pt` ditimpa pada tiap epoch, bukan hanya saat membaik.
- Normalisasi memakai rata-rata dan simpangan baku seluruh data. Tidak boleh. Pakai hanya pembagian 255.
- Dua orang menjalankan run yang sama di Drive bersamaan. Beri kabar di grup sebelum menjalankan lipatan tertentu.
- Mengubah angka di `configs/default.yaml` tanpa kabar. Itu mengubah eksperimen.

## Laporan ke tim

Tiap selesai satu tahap, kirim pesan pendek: tahap, commit, jumlah berkas, satu angka kunci (misalnya Dice validasi rata-rata), dan masalah yang masih terbuka.
