# Daftar referensi proposal dan status pemeriksaan

Aturan dosen: artikel jurnal atau konferensi, 5 tahun terakhir. Semua entri di bawah bertahun 2021 sampai 2026, kecuali WHO (2024), yang dokumen resmi dan dipakai hanya untuk angka beban TB.

Kolom "Diperiksa" jujur: **sudah** berarti sumbernya dibuka dan angkanya dicocokkan pada sesi penyusunan ini. **belum** berarti dibawa dari proposal pertama dan belum dicek ulang. Tugas Nabil (N6) adalah mengubah semua "belum" menjadi "sudah".

| Rujukan di teks | Klaim yang dikutip | DOI atau tautan | Diperiksa |
|---|---|---|---|
| WHO (2024) | Indonesia 10% kasus TB dunia 2023, kedua setelah India (26%). Survei inventori: 14% kasus tidak terdiagnosis, insidensi 394 per 100.000 | who.int, Global TB Report 2024, bagian 1.1 dan studi inventori Indonesia | sudah |
| Qin dkk. (2024) | 12 produk CAD TB dibandingkan pada survei prevalensi Afrika Selatan. Pemahaman pemilihan ambang antar populasi masih kurang | 10.1016/S2589-7500(24)00118-3 | sudah (abstrak) |
| Xue dkk. (2023) | Analisis pergeseran domain deteksi paru pada lima dataset, dibandingkan lewat metadata, tampilan citra, fitur jaringan terlatih | 10.3390/diagnostics13061068 | sudah (abstrak terpotong, baca isi penuh untuk temuan) |
| González dkk. (2022) | Jarak Mahalanobis pada ruang fitur untuk mendeteksi kegagalan diam-diam segmentasi lesi COVID-19 di CT | 10.1016/j.media.2022.102596 | sudah (abstrak) |
| Wang dkk. (2024) | CLAHE vs ps-KDE pada 247 citra JSRT, Dice kiri 0,780 vs 0,717, belum ada validasi eksternal | 10.1371/journal.pone.0299623 | sudah (247 citra JSRT, Dice kiri 0,780 ps-KDE vs 0,717 CLAHE, dan catatan belum ada validasi eksternal cocok) |
| Bassi & Attux (2022) | Segmentasi sebelum klasifikasi COVID-19 menaikkan akurasi 74% ke 78,7% pada 150 citra eksternal | 10.1007/s42600-022-00242-y | sebagian (DOI, judul, dan kenaikan 74% ke 78,7% terkonfirmasi; jumlah 150 citra eksternal perlu cek naskah penuh) |
| Mabina dkk. (2025) | U-Net + CLAHE, Dice 95,87%, IoU 92,07% pada Shenzhen dan Montgomery | 10.30871/jaic.v9i4.9869 | sudah (Shenzhen & Montgomery, Dice 95,87%, IoU 92,07% cocok) |
| Ahmad dkk. (2025) | CLAHE vs HE vs anisotropic diffusion pada X-ray tulang belakang, signifikan di U-Net, tidak di Mask R-CNN | 10.3390/a18120796 | sudah (signifikan di U-Net, tidak signifikan di Mask R-CNN cocok) |
| Danilov dkk. (2022) | Koleksi 6.810 citra dari Darwin, Montgomery, Shenzhen | 10.1038/s41598-022-15013-z, dataset 10.17632/8gf9vpkhgy.2 | sudah (6.810 entri & 3 sumber terkonfirmasi; berkas Montgomery di Mendeley 138, bukan 139) |
| Din dkk. (2025) | CXR-Seg, tiap dataset dievaluasi dengan pembagian sendiri | 10.3390/bioengineering12020167 | sudah (memakai Darwin, Montgomery, Shenzhen dengan pembagian masing-masing dataset) |
| Azhar dkk. (2025) | CLAHE: akurasi latih 93,26%, validasi 91,31% vs tanpa pra-proses 91,50% | 10.14569/IJACSA.2025.0160161 | sudah (93,26%, 91,31%, dan 91,50% tanpa pra-proses cocok) |
| Singthongchai & Wangkhamhan (2025) | Normalisasi adaptif membantu data kontras stabil, tidak pada MIMIC-CXR heterogen | 10.3390/jimaging12010014 | sudah (abstrak, arah temuan cocok) |
| Zhou dkk. (2023) | Survei domain generalization, pergeseran sebaran menurunkan performa | 10.1109/TPAMI.2022.3195549 | sudah (TPAMI 2023, domain shift menurunkan performa di target OOD) |
| Mohd-Isa dkk. (2021) | Cara kerja CLAHE, tile, clip limit | 10.12688/f1000research.73236.1 | sudah (cara kerja CLAHE, pembagian tile, dan clip limit cocok) |
| Isensee dkk. (2021) | nnU-Net, U-Net yang dikonfigurasi baik adalah pembanding kuat | 10.1038/s41592-020-01008-z | sudah (Nature Methods 2021, U-Net yang dikonfigurasi baik sebagai baseline kuat) |
| Maier-Hein dkk. (2024) | Metrics Reloaded, pasangan metrik tumpang tindih dan jarak batas | 10.1038/s41592-023-02151-z | sudah (Nature Methods 2024, pasangan metrik overlap dan boundary distance) |
| Mustapha dkk. (2025) | Attention U-Net untuk paru | 10.1016/j.jrras.2025.101650 | sudah (Optimized attention U-Net untuk segmentasi paru di CXR dan CT) |

## Catatan

- Agrawal & Choudhary (2023), yang ada di proposal pertama, dihapus karena tidak lagi dikutip di teks.
- Kalimat "kami belum menemukan studi yang menguji CLAHE pada segmentasi paru dengan skema leave-one-source-out" adalah klaim penelusuran kami, bukan klaim mutlak. Penelusuran ulang (2021–2026: `"CLAHE" "lung segmentation" ("cross-dataset" OR "external validation" OR "leave-one-source-out")`) mengonfirmasi bahwa studi eksternal/cross-dataset masih jarang dan belum ditemukan uji CLAHE khusus segmentasi paru dengan protokol *leave-one-source-out* penuh. Tetap cek ulang sebelum laporan akhir bila terbit studi baru.
