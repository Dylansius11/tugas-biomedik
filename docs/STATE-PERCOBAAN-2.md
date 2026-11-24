# STATE PERCOBAAN 2

Catatan keadaan setelah Percobaan 2. Lanjutan dari `STATE-PERCOBAAN-1.md`. Eksploratif dan post-hoc, disetujui Dylan.
Rencana: `PLAN-PERCOBAAN-2.md`. Hasil dan angka: `HASIL-PERCOBAAN-2.md`. Ke-36 run utama dan `PROTOCOL.md` tidak diubah.

## Apa yang dijalankan
- Diagnosis: Darwin menandai paru anatomis, Montgomery dan Shenzhen menandai paru yang terlihat (gambar `results/exp2/figures/convention.png`).
- E1: U-Net dengan dua head (`visible`, `anatomical`), tiap foto hanya melatih head sumbernya. E2: E1 + augmentasi kuat.
- 18 run (2 lengan x 3 lipatan x 3 seed), jadwal latih sama dengan run utama, kontras S0.
- Lengan akhir E2, dipilih dari Dice validasi dengan aturan yang dikunci sebelum hasil (selisih < 0,002 memilih E2).
- Few-shot untuk E2: k = 5, 10, 20, 50 foto berlabel dari sumber baru, 300 iterasi, replay data lama, 36 adaptasi.
- Alarm A3 (faktor dari data validasi + peta ketidakpastian dari TTA).

## Hasil singkat (foto gagal di sumber asing)
Shenzhen 50,5% -> 6,5%. Montgomery 6,5% -> 2,9%. Darwin 97,3% -> 97,3% (definisi label belum pernah dilihat model).
Few-shot Darwin: k=10 gagal 22%, k=50 gagal 10%.
A3 hasilnya campuran: menjelaskan kegagalan Darwin, tetapi alarm palsu di Montgomery dan kalibrasi belum sekitar 5% di Shenzhen.

## Kode dan berkas
- `src/cxrseg/exp2.py`, `src/cxrseg/explain.py`, `tests/test_exp2.py` (38 tes lolos)
- `scripts/run_exp2.py` (latih, pilih lengan, few-shot), `scripts/exp2_report.py`, `scripts/exp2_status.py`
- `configs/default.yaml` bagian `exp2` (bagian lain tidak diubah)
- `results/exp2/tables`, `figures`, `summary.txt`, `final_arm.txt`. Checkpoint dan prediksi tidak masuk git.

## Cara mengulang
```
python scripts/run_exp2.py       # sekitar 2 jam di RTX 4060, bisa dilanjutkan
python scripts/exp2_report.py    # sekitar 15 menit
python scripts/exp2_status.py    # cek progres
```

## Yang masih terbuka
1. Kalibrasi A3: faktor kontras dan jarak histogram menimbulkan alarm palsu di Montgomery. Perbaikan harus dilabeli post-hoc.
2. Judul gambar `xai_cards_*.png` bertumpuk.
3. Head `visible` dipilih dari metadata definisi label. Untuk sumber tanpa dokumentasi perlu cara lain.
4. Few-shot memakai tiga seed yang tidak saling bebas. Belum diuji di sumber keempat.
5. Dylan memutuskan bagaimana hasil ini masuk ke laporan (sebagai analisis tambahan, bukan pengganti hasil utama).
