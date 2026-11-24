# Hasil Percobaan 2 (eksploratif, post-hoc)

Rencana dan aturan: `docs/PLAN-PERCOBAAN-2.md` (dikunci sebelum hasil keluar). Angka mentah: `results/exp2/summary.txt` dan `results/exp2/tables/`.
Ini bukan tabel resmi. Tabel resmi tetap dari `analyze.py` milik Nabil. Seed 11, 22, 33 dirata-ratakan per foto.

## 1. Temuan utama

Perbaikan terbesar datang dari **memisahkan definisi label** (head `visible` untuk Montgomery dan Shenzhen, head `anatomical` untuk Darwin), bukan dari kontras.
Lengan akhir **E2** dipilih dari Dice validasi (E1 0,95758 vs E2 0,95750, selisih di bawah 0,002, jadi aturan memilih E2).

### Sumber asing: foto gagal (Dice < 0,90) dan median Dice

| Sumber asing | E0 (lama) | E1 | E2 |
|---|---|---|---|
| Shenzhen | 50,5% / 0,900 | 10,6% / 0,944 | **6,5% / 0,960** |
| Montgomery | 6,5% / 0,945 | 2,9% / 0,974 | **2,9% / 0,974** |
| Darwin | 97,3% / 0,773 | 97,8% / 0,776 | 97,3% / 0,779 |

Selisih Dice rata-rata E2 terhadap E0 (CI bootstrap 95%): Shenzhen +0,055 (0,052 sampai 0,058), Montgomery +0,026 (0,022 sampai 0,030), Darwin +0,012 (0,007 sampai 0,016).
Di Darwin selisihnya melewati batas efek minimum 0,01 tetapi tidak berarti secara klinis: 97% foto tetap gagal.
Dice di sumber sendiri tidak turun (median 0,966 sampai 0,975 untuk ketiganya).

### Darwin: few-shot (E2 + k foto berlabel dari Darwin, dievaluasi di split val+test Darwin)

| k | 0 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| foto gagal | 96% | 54% | 22% | 16% | 10% |
| median Dice | 0,79 | 0,90 | 0,92 | 0,93 | 0,94 |

Darwin tidak bisa ditebak tanpa label karena definisinya belum pernah dilihat model, tapi sekitar 10 sampai 20 foto berlabel sudah cukup membawa sebagian besar perbaikan.
Dice belum menyamai sumber sendiri (0,975).
Di Shenzhen dan Montgomery few-shot tidak menambah banyak karena sudah bagus.

## 2. Alarm yang menjelaskan (A3): hasil campuran

A3 = skor-z terbesar dari 9 faktor, termasuk peta ketidakpastian (prediksi diulang dengan foto digeser dan di-zoom). Ambang dari validasi.

| Model | Sumber asing | foto beralarm | foto gagal | gagal yang tertangkap | AUROC A1 / A2 / A3 |
|---|---|---|---|---|---|
| E0 | Darwin | 55% | 97% | 56% | 0,51 / 0,65 / 0,71 |
| E0 | Shenzhen | 18% | 51% | 23% | 0,52 / 0,54 / 0,60 |
| E2 | Darwin | 56% | 97% | 57% | 0,45 / 0,75 / 0,75 |
| E2 | Shenzhen | 17% | 6,5% | 50% | 0,54 / 0,84 / 0,76 |
| E2 | Montgomery | 32% | 2,9% | 83% | tidak dapat dihitung (gagal < 10) |

Yang berhasil:
- Alarm A3 menyatakan "belum siap" untuk Darwin dan Shenzhen dengan alasan yang masuk akal dan bisa dicek: paru kiri-kanan tidak seimbang dan prediksi goyah saat foto digeser. Faktor `lr_balance` punya AUROC 0,83 sampai 0,86 terhadap kegagalan di Darwin.
- AUROC A3 pada Darwin (0,71 sampai 0,75) lebih baik daripada A1 (0,45 sampai 0,51).
- Alarm hanya menyala di sekitar separuh foto Darwin, sedangkan 97% gagal. Jadi alarm menangkap sinyal, tapi belum menangkap semuanya.

Yang **tidak** berhasil:
- Di Montgomery, E2 sebenarnya siap (gagal 2,9%) tetapi alarm menyala 32% sampai 39% dan menyatakan "belum siap". Ini alarm palsu. Penyebab utamanya faktor kontras dan jarak histogram, yang tidak berhubungan dengan kegagalan di sana.
- Tingkat alarm di sumber sendiri tidak selalu sekitar 5%: 5 sampai 7% di Darwin dan Montgomery tetapi 11 sampai 14% di Shenzhen. Ambang persentil-95 dari validasi tidak sepenuhnya terkalibrasi karena skor-z memakai nilai maksimum dari 9 faktor.
- Di Shenzhen AUROC A2 (0,84) lebih baik daripada A3 (0,76). Menggabungkan faktor tidak selalu lebih baik daripada satu faktor yang tepat.
- Keputusan siap/belum siap berdasarkan batas alarm 10% benar untuk Darwin dan Shenzhen tetapi salah untuk Montgomery (E2).

Kesimpulan jujur: penjelasan alarm membantu memahami mengapa model gagal di Darwin (bentuk paru dan ketidakstabilan), tetapi A3 belum layak dipakai sebagai alarm yang berdiri sendiri. Perlu kalibrasi dan pemilihan faktor yang divalidasi di lebih banyak sumber.

## 3. Keterbatasan

- Hanya tiga sumber dan tiga lipatan. Pemilihan head untuk sumber asing memakai metadata definisi label, yang tidak tersedia kalau sumbernya tidak punya dokumentasi.
- Few-shot memakai k foto dari split train sumber baru dan jadwal tetap 300 iterasi. Hasil memakai tiga seed yang tidak saling bebas.
- A3 ditambahkan setelah melihat hasil Percobaan 1 (post-hoc), sehingga angkanya tidak boleh dibaca sebagai uji konfirmasi.
- Ketiga angka di bagian 2 belum diuji ulang di sumber di luar ketiga ini.
- Gambar `xai_cards_*.png`: judul kolom bertumpuk, perlu dirapikan.

## 4. Gambar

`results/exp2/figures/`: `convention.png` (definisi tanda per sumber), `foreign_fail.png`, `fewshot.png`, `xai_cards_E0.png`, `xai_cards_E2.png`.
