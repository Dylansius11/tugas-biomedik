# Rencana Percobaan 2

Status: eksperimen tambahan **post-hoc** (dirancang setelah melihat hasil Percobaan 1), disetujui Dylan.
Ke-36 run utama dan `docs/PROTOCOL.md` tidak diubah. Semua angka ada di `configs/default.yaml` bagian `exp2`.
Dokumen ini ditulis **sebelum** hasil Percobaan 2 dilihat. Keputusan di bawah tidak diubah setelah hasil keluar.

## 1. Diagnosis dari Percobaan 1

Tiga sumber ternyata **tidak memakai definisi "paru" yang sama**. Ini dilihat dari mask aslinya, bukan dari model.

| Sumber | Yang ditandai | Bukti |
|---|---|---|
| Darwin | paru **anatomis**: ikut bagian di belakang jantung dan di belakang diafragma | gambar contoh mask; luas mask 32% vs 25-26% |
| Montgomery, Shenzhen | paru **yang terlihat**: berhenti di tepi jantung dan diafragma | gambar contoh mask |

Akibatnya pada model baseline (S0, seed 11):

| Fold (sumber asing) | presisi (bagian prediksi yang ada di dalam tanda) | cakupan (bagian tanda yang tertutup prediksi) | arti |
|---|---|---|---|
| Darwin | 0.976 | 0.648 | model menggambar paru yang terlihat dengan benar, tapi tanda Darwin lebih luas |
| Shenzhen | 0.830 | 0.983 | model ikut belajar gaya Darwin, jadi menggambar terlalu luas di Shenzhen |
| Montgomery | 0.916 | 0.969 | campuran, efeknya kecil |

Kesimpulan: kegagalan terbesar **bukan** soal kontras atau jumlah data, tapi **perbedaan definisi label**. CLAHE tidak bisa
memperbaiki ini, dan model apa pun yang hanya melihat satu definisi tidak bisa menebak definisi yang lain.

## 2. Lengan percobaan

| Lengan | Isi | Untuk menjawab |
|---|---|---|
| E0 | baseline S0 dari Percobaan 1 (tidak dilatih ulang) | pembanding |
| E1 | U-Net dengan **dua head**: `visible` dan `anatomical`. Tiap foto latih hanya melatih head sesuai sumbernya | apakah memisahkan definisi label memperbaiki Shenzhen dan Montgomery |
| E2 | E1 + **augmentasi kuat**: geometri lebih lebar + gamma, kecerahan, kontras, blur, noise | apakah ketahanan terhadap gaya foto menambah perbaikan |
| F  | lengan terpilih + **few-shot**: fine-tune dengan k = 5, 10, 20, 50 foto berlabel dari sumber baru | berapa banyak label yang dibutuhkan untuk sumber dengan definisi baru (Darwin) |

Aturan yang dikunci sebelumnya:
- Head untuk sumber asing dipilih dari **metadata definisi label**, bukan dari gambarnya. Kalau definisi itu belum pernah
  dilatih (fold `ho_darwin`), dipakai head `visible`.
- Lengan akhir dipilih dari **rata-rata Dice validasi** (sumber latih saja). Kalau selisih E1 dan E2 < 0.002, pilih E2.
- Few-shot: foto adaptasi diambil dari split *train* sumber baru; evaluasi di split *val + test* sumber baru, yang tidak
  pernah dipakai adaptasi. Jadwal tetap 300 iterasi, tanpa early stopping, jadi tidak butuh label tambahan.
  Data latih lama ikut diputar ulang (replay) supaya model tidak lupa.
- Jadwal latih, seed, batch, lr, early stopping sama dengan run utama. Kontras S0.

## 3. Alarm yang bisa menjelaskan (A3)

Bukan hanya "siap / belum siap", tapi juga **alasannya**. Dua jenis bukti, keduanya dibandingkan dengan foto validasi
sumber latih (tidak pernah dengan sumber asing):

1. **Faktor dari data latih**: kecerahan, kontras, jarak histogram (A1), luas paru prediksi, posisi batas bawah,
   keseimbangan kiri-kanan, jumlah potongan mask, keraguan rata-rata (A2), dan bagian paru yang goyah.
   Tiap faktor jadi skor-z robust (median dan MAD validasi).
2. **Peta ketidakpastian model**: prediksi diulang dengan foto digeser 8 piksel dan di-zoom 0.95 / 1.05, lalu dihitung
   sebarannya per piksel. Daerah yang goyah ditampilkan sebagai peta panas.

Skor A3 = |z| terbesar. Ambang tau = persentil 95 di validasi, aturan yang sama dengan A1/A2. Faktor disebut sebagai
alasan kalau |z|-nya sendiri di atas tau. Penjelasan **divalidasi**: AUROC tiap faktor terhadap kegagalan di sumber asing,
dan berapa persen foto gagal yang tertangkap.

## 4. Yang dilaporkan

- Persentase foto gagal dan median Dice di sumber asing per fold, E0 vs E1 vs E2 (Wilcoxon berpasangan, CI bootstrap).
- Kurva few-shot.
- A3 vs A1/A2: AUROC, alarm di sumber sendiri (harus sekitar 5%), keputusan siap/belum siap per sumber beserta alasannya.
- Semua dilabeli **eksploratif / post-hoc**. Hasil negatif tetap dilaporkan.

## 5. Dataset tambahan

Tidak dipakai. Masalah utamanya definisi label, jadi menambah sumber dengan definisi label yang tidak jelas
(COVID-QU-Ex) atau label buatan model (CheXmask) bisa memperkeruh, bukan membantu. JSRT/SCR bisa jadi sumber keempat
di masa depan, tapi tidak perlu untuk menjawab masalah ini.

## 6. Cara menjalankan

```
python scripts/run_exp2.py       # latih E1, E2, pilih lengan, few-shot (bisa dilanjutkan kalau terputus)
python scripts/exp2_report.py    # tabel, alarm A3, gambar -> results/exp2/
```
