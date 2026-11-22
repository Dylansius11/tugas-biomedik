# BUILD-PLAN: Dylan (fondasi, proposal, laporan, integrasi)

Perananmu: pemilik **kontrak** (apa yang dikerjakan, dalam format apa), **penulis** (proposal dan laporan akhir), dan **penjaga mutu** (review, gerbang minggu).

## Status awal (sudah selesai)

- [x] Repo, struktur folder, `.gitignore`, `requirements.txt`, `pyproject.toml`
- [x] `docs/PROTOCOL.md`: semua keputusan eksperimen terkunci
- [x] `configs/default.yaml`
- [x] `cxrseg.preprocess`, `postprocess`, `metrics`, `data`, dan 20 tes
- [x] `scripts/download_data.sh` (cek SHA-256), `scripts/prepare_data.py`, `scripts/make_fake_predictions.py`, `scripts/check_runs.py`
- [x] Proposal final (`paper/proposal/`)

## Langkah

### D1. Bagikan data (hari pertama)
1. `bash scripts/download_data.sh`, lalu `python scripts/prepare_data.py --zip`. Hasilnya `data/processed.zip`.
2. Unggah `processed.zip` ke Google Drive bersama, beri akses Huda dan Nabil. Taruh juga tautan di grup. Ukurannya kecil (citra 256x256).
3. Cek angka akhir di terminal sama dengan README (jumlah per sumber, split).
- **Selesai bila:** Huda dan Nabil bisa menjalankan `pytest` hijau dan `load_manifest()` di komputer masing-masing.

### D2. Kirim rencana dan jelaskan satu kali (hari pertama)
Kirim tiga file `BUILD-PLAN-*.md` dan jelaskan cerita proyek lima kalimat di `BUILD-PLAN.md`. Pastikan Huda paham aturan emas (sumber ditahan tidak disentuh) dan Nabil paham aturan ambang alarm (hanya dari validasi sumber latih).

### D3. Review (sepanjang proyek)
Daftar periksa tiap merge request:
- [ ] Tes lolos, ada tes baru untuk logika baru.
- [ ] Tidak ada angka bawaan di kode yang berbeda dengan `configs/default.yaml`.
- [ ] Tidak ada pembacaan sumber ditahan di luar tahap prediksi `foreign` (Huda) atau tahap evaluasi (Nabil).
- [ ] Format keluaran sama dengan PROTOCOL bagian 6.
- [ ] Commit kecil, satu perilaku, subjek sesuai konvensi.
- [ ] Tidak ada data, checkpoint, atau kunci ikut ter-commit.

### D4. Keputusan yang hanya kamu yang boleh ambil
| Keputusan | Kapan | Aturan |
|---|---|---|
| Membuang citra bermasalah dari QC | Setelah laporan N1 | Buang hanya duplikat lintas sumber dan mask jelas rusak. Catat id dan alasan di `docs/QC-REPORT.md`. Perbarui jumlah di proposal |
| Mengubah angka di PROTOCOL | Hanya sebelum pelatihan penuh dimulai | Commit tersendiri, kabari semua. Sesudah pelatihan dimulai: tidak boleh |
| Memakai Attention U-Net | Setelah 36 run selesai | Hanya bila waktu cukup |
| Cara melaporkan hasil negatif | Saat analisis selesai | Laporkan apa adanya. Jangan ganti ambang setelah melihat hasil |

### D5. Tulis laporan akhir (minggu 3 sampai 4)
Kerangka (isi tiap bagian dari tabel dan gambar Nabil, bukan dari ingatan):
1. **Pendahuluan:** ambil dari proposal final, perbarui dengan jumlah data akhir.
2. **Metode:** ambil dari PROTOCOL, tulis dalam kalimat biasa.
3. **Hasil:** satu subbagian per RM. RM1 dari T1 dan F2. RM2 dari T2, T3, F1, F4, F5. RM3 dari T5, T6, T7, F3, F6.
4. **Diskusi:** apakah CLAHE menolong di sumber asing, pada sumber mana, dan seberapa besar. Apakah alarm berguna. Apa arti hasil untuk rumah sakit baru.
5. **Batasan:** hanya tiga sumber, Darwin bukan koleksi TB, tanpa metadata pasien, satu arsitektur utama, ambang kesiapan adalah konvensi awal.
6. **Kesimpulan:** satu paragraf, menjawab RM1 sampai RM3 secara berurutan.
7. **Referensi:** dari `paper/proposal/REFERENSI.md`, tambah yang dipakai di bagian Diskusi.
Aturan: tiap angka di teks harus bisa ditunjuk ke satu baris CSV di `results/tables`.

### D6. Uji ulang dari awal (minggu 4)
Di Colab kosong: clone, `download_data.sh`, `prepare_data.py`, jalankan satu run, `analyze.py`. Pastikan tanpa langkah manual tersembunyi. Ini bukti klaim "mudah direproduksi" di proposal.

## Kriteria sukses Dylan

| Pemeriksaan | Ambang |
|---|---|
| Fondasi | `pytest` hijau, `prepare_data.py` menghasilkan 600 + 138 + 566 citra bila tidak ada yang dibuang |
| Dukungan tim | Huda dan Nabil tidak menunggu lebih dari satu hari untuk keputusan atau review |
| Gerbang mingguan | Tiap gerbang di `BUILD-PLAN.md` bagian 5 diperiksa dan dicatat lolos atau tidak |
| Laporan | Semua RM terjawab, semua angka terlacak, batasan tertulis jujur |
| Reproduksi | D6 berhasil di lingkungan kosong |

## Pertanyaan terbuka untuk dosen

1. Format dan tenggat luaran akhir (laporan, presentasi, atau demo)?
2. Apakah kata "rumah sakit" di judul boleh mewakili sumber data publik dari institusi berbeda? (Kami menjelaskannya di Metode dan Batasan.)
3. Apakah hasil negatif (CLAHE tidak berpengaruh) diterima selama dianalisis jujur?
