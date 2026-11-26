# QC-REPORT: kualitas data (N1)

Dibuat 25 November 2026 dari `data/processed` (dibangun ulang dengan `scripts/download_data.sh` + `scripts/prepare_data.py`, keempat SHA-256 cocok).
Luaran: `results/figures/qc_{darwin,montgomery,shenzhen}.png`, `results/tables/qc_summary.csv`, `results/tables/qc_duplicates.csv`, `results/tables/qc_flags.csv`.
**Tidak ada citra yang dibuang.** Keputusan sementara per 26 November 2026 (Dylan belum menjawab, bisa dibalik): lihat bagian "Keputusan sementara".

## Ringkasan per sumber

| Sumber | Citra | Ukuran | Luas mask rata-rata (min–maks) | Intensitas rata-rata ± SB | Mask non-biner |
|---|---|---|---|---|---|
| Darwin | 600 | 512×512 | 0,325 (0,175–0,534) | 122,5 ± 61,0 | 0 |
| Montgomery | 138 | 512×512 | 0,255 (0,099–0,398) | 112,9 ± 85,5 | 0 |
| Shenzhen | 566 | 512×512 | 0,253 (0,084–0,433) | 157,0 ± 65,1 | 0 |

Semua 1.304 mask hanya berisi 0 dan 255. Luas mask Darwin lebih besar karena konvensi anotasi berbeda (paru anatomis, termasuk area di balik jantung dan diafragma), sesuai temuan Percobaan 2.

## Duplikat lintas sumber

`qc_duplicates.csv` memuat **860 pasangan** dengan dHash 8×8 Hamming ≤ 4 (sesuai rencana): Darwin–Shenzhen 626, Montgomery–Shenzhen 214, Darwin–Montgomery 20.
Setelah diverifikasi, **semuanya bukan duplikat**:

- Korelasi piksel (128×128) tertinggi dari 860 pasangan 0,91 (median 0,66). Duplikat sejati > 0,99.
- dHash 16×16 (256 bit): jarak terkecil 28, tidak ada pasangan ≤ 16.

Kesimpulan: dHash 64 bit terlalu kasar untuk rontgen thorax yang tata letaknya seragam. **Nol duplikat lintas sumber.**

Verifikasi kedua sekarang ada di `qc_data.py`: setiap kandidat dHash diberi kolom `pixel_corr` (korelasi piksel 128×128) dan `is_duplicate` (korelasi ≥ 0,99). Ambangnya di `configs/default.yaml` bagian `qc`. Hasil: 860 kandidat, 0 duplikat terverifikasi, korelasi tertinggi 0,913. Kewajiban PROTOCOL bagian 2 butir 4 (duplikat lintas sumber dibuang dari sumber uji) terpenuhi tanpa membuang citra.

## Pemeriksaan visual dan otomatis (seluruh 1.304 citra)

Pemeriksaan otomatis: paru lebih terang dari cincin di sekitarnya (kandidat terbalik), mask dengan jumlah komponen besar ≠ 2, luas mask di luar 3×IQR per sumber, dan SB intensitas < 25. Semua kandidat dilihat langsung.

| Id | Temuan | Usul |
|---|---|---|
| `darwin/00006967_fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original` | Kotak anotasi putih tercetak di citra, mask paru kanan ikut berbentuk kotak (3 komponen) | **Dipertahankan, ditandai** (keputusan sementara, `qc_flags.csv`) |
| `shenzhen/CHNCXR_0361_1` | Mask paru kiri terbelah dua (3 komponen). Tampak sesuai anatomi (kolaps/kavitas TB) | Dipertahankan |
| `darwin/00006649_2168a917`, `darwin/00006769_a2eba651`, `darwin/00006774_a361d7b7`, `darwin/00006825_c601f50d` | Kontras sangat rendah (tampak pudar), mask baik | Dipertahankan, dicatat sebagai variasi akuisisi |
| `montgomery/MCUCXR_0251_1` | Proyeksi apikal lordotik ("LT APICAL"), bukan PA biasa | Dipertahankan, dicatat |

Tidak ada citra Darwin yang benar-benar terbalik atau berupa foto. 38 citra Darwin yang ditandai "paru lebih terang" ternyata rontgen anak dengan pneumonia berat (konsolidasi), mask wajar.

Catatan lain: Darwin banyak berisi rontgen anak dan beberapa dengan alat (mis. `00006747_19073f37`, pacu jantung di dalam mask). Montgomery punya pita hitam di bawah citra. Shenzhen memuat penanda "L" atau "R" di beberapa posisi. Semua ini variasi sumber, bukan galat.

## Keputusan sementara (26 November 2026)

Dylan belum menjawab dua pertanyaan QC. Dua keputusan di bawah dipakai sampai ada jawaban. Keduanya tidak mengubah angka hasil dan mudah dibalik.

**1. `darwin/00006967_...annot.original` dipertahankan dan ditandai, tidak dibuang.**

| Lipatan | Peran citra | Dice (36 run) | Gagal |
|---|---|---|---|
| ho_darwin | asing | 0,909 – 0,922 | tidak pernah |
| ho_montgomery | in-source test | 0,920 – 0,963 | tidak pernah |
| ho_shenzhen | in-source test | 0,951 – 0,969 | tidak pernah |

Alasan:
- Citra ini tidak pernah gagal, jadi tingkat gagal tidak berubah. Median Dice paling jauh bergeser di desimal keempat (1 dari 600 citra Darwin).
- Membuangnya setelah hasil keluar berarti pengecualian post-hoc.
- PROTOCOL hanya mewajibkan membuang duplikat lintas sumber.
- Citra ada di split test, jadi tidak pernah ikut melatih model.

Dicatat di `configs/default.yaml` (`qc.flagged`) dan `results/tables/qc_flags.csv`. Laporan perlu satu kalimat di bagian keterbatasan.

**2. Verifikasi duplikat kedua ditambahkan ke `qc_data.py`** (korelasi piksel ≥ 0,99). Hasil: 0 duplikat sejati dari 860 kandidat.

## Untuk Dylan

1. Keputusan sementara di atas: setuju atau dibalik? Membuang citra 00006967 berarti `analyze.py` dijalankan ulang tanpa citra itu (tidak perlu melatih ulang).
2. Ambang verifikasi duplikat 0,99: setuju?
