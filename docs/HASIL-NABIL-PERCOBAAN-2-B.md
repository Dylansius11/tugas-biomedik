# HASIL NABIL: Percobaan 2, skenario B (Percobaan 2 sebagai hasil utama)

26 November 2026, cabang `nabil/exp2-b`. Plan: `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2-B.md`. Pasangan: `docs/HASIL-NABIL-PERCOBAAN-2.md` di cabang `nabil/exp2-a`.
**Lengan E1/E2 dirancang setelah melihat Percobaan 1.** Walaupun di skenario ini menjadi hasil utama, temuannya tetap post-hoc dan perlu dikonfirmasi pada sumber keempat.

Luaran: `results/exp2/main/tables` (T0–T3, T5–T9), `results/exp2/main/figures` (F1–F5), `results/exp2/main/metrics`.
```
python scripts/analyze_exp2.py --mode main --pred-dir <predictions> --exp2-dir <percobaan2> --out results/exp2/main
```
Sekitar 5 menit di CPU. Aturan sama dengan hasil utama: Dice per citra dirata-ratakan atas 3 seed, gagal = Dice < 0,90, tau dari validasi dua sumber latih.

Pemeriksaan: 54 + 9 berkas valid (T0). 105 dari 105 angka cocok dengan tabel Huda dan T5 S0 hasil utama (T9). T3 identik dengan X3 skenario A.

## RM1. Celah generalisasi (T1, F2)

| Sumber asing | Lengan | Median Dice dalam sumber | Median Dice asing | Celah | Gagal | HD95 median (piksel) |
|---|---|---|---|---|---|---|
| Darwin | E0 | 0,975 | 0,773 | 0,202 | 97,3% | 33,9 |
| | E1 | 0,974 | 0,776 | 0,198 | 97,8% | 33,7 |
| | E2 | 0,975 | 0,779 | 0,196 | 97,3% | 33,8 |
| Montgomery | E0 | 0,966 | 0,946 | 0,021 | 6,5% | 11,1 |
| | E1 | 0,969 | 0,974 | −0,005 | 2,9% | 3,3 |
| | E2 | 0,968 | 0,974 | −0,006 | 2,9% | 2,8 |
| Shenzhen | E0 | 0,966 | 0,900 | 0,066 | 50,5% | 22,8 |
| | E1 | 0,967 | 0,944 | 0,023 | 10,6% | 10,0 |
| | E2 | 0,966 | 0,960 | 0,006 | 6,5% | 4,8 |

Dengan E2, celah di Montgomery dan Shenzhen praktis hilang. Celah di Darwin tetap sekitar 0,20 karena definisi label anatomis belum pernah dilihat model.

## RM2 (usulan B). Apakah dua head + augmentasi kuat mengurangi kegagalan? (T2, T3, F1)

Wilcoxon berpasangan, CI bootstrap 10.000 ulang, Holm atas **6 uji** (E1 dan E2 vs E0 × 3 sumber), bermakna bila p Holm < 0,05 dan |selisih| ≥ 0,01.

| Sumber asing | E1 − E0 [CI 95%] | Bermakna | E2 − E0 [CI 95%] | Bermakna |
|---|---|---|---|---|
| Darwin | +0,001 [−0,001; 0,002] | tidak | +0,012 [0,008; 0,016] | ya (batas) |
| Montgomery | +0,024 [0,020; 0,028] | ya | +0,026 [0,022; 0,030] | ya |
| Shenzhen | +0,041 [0,039; 0,043] | ya | +0,055 [0,052; 0,058] | ya |

Tingkat gagal E0 → E2: Shenzhen 50,5% → 6,5% (selisih −44,0 poin [−48,2; −39,6]), Montgomery 6,5% → 2,9% (−3,6 poin [−8,0; 0,7], belum pasti), Darwin tidak berubah.

**Jawaban:** ya untuk Shenzhen dan Montgomery. Head per konvensi (E1) menyumbang sebagian besar perbaikan, dan augmentasi kuat (E2) menambah sedikit. Untuk Darwin tidak ada perbaikan yang berarti secara klinis.

## RM3. Apakah alarm tanpa label bisa memberi tahu kesiapan? (T5, T6, T7, F3)

Keputusan alarm vs label kesiapan pada sumber asing (benar dari 3 sumber):

| Alarm | E0 | E1 | E2 |
|---|---|---|---|
| A1 (jarak histogram) | 2 / 3 | 1 / 3 | 1 / 3 |
| A2 (entropi) | 1 / 3 | **3 / 3** | 2 / 3 |
| A3 (Huda, seed 11) | 3 / 3 | tidak dihitung | 2 / 3 |

Pada data dalam sumber, A1 dan A2 benar 9 dari 9 (tidak ada alarm palsu yang membalik keputusan).

| AUROC A2 (gagal vs lulus) | E0 | E1 | E2 |
|---|---|---|---|
| Darwin | 0,66 | 0,70 | 0,76 |
| Shenzhen | 0,63 | 0,83 | 0,87 |

AUROC A1 tetap sekitar 0,5 di semua lengan. Montgomery: AUROC tidak dapat dihitung (< 10 gagal).

**Jawaban:** A2 makin berguna saat model lebih baik. AUROC naik, dan keputusan kesiapannya benar di semua sumber pada E1. A1 tidak berguna karena hanya mengukur gaya citra. Kesalahan A2 pada E2 Shenzhen adalah kasus di dekat batas (gagal 6,5% vs batas 5%, alarm 4,1% vs batas 10%). Pada E1 Shenzhen, alarm A2 64% dengan tingkat gagal 10,6%, jadi keputusannya benar. Keputusan dari 50 citra acak sama dengan keputusan dari seluruh citra di ≥ 99% sampel untuk A2 (T7).

## Pendukung: few-shot (T8, F4)

Sama dengan skenario A. Darwin turun dari sekitar 96% gagal ke 29% (k = 10) dan 14% (k = 50), rata-rata per seed dengan rentang lebar. Montgomery 0% sejak k = 5. Shenzhen tidak berubah (sekitar 8%).

## Kasus terburuk (F5)

- Darwin: rontgen anak dengan pneumonia, model hanya menandai sebagian paru.
- Shenzhen: `CHNCXR_0074_0`–`0076_0` (rontgen anak AP berurutan). Prediksi lebih luas dari mask asli yang tampak sempit. Kandidat masalah anotasi, perlu dicek Dylan.

## Lampiran: RM2 lama (CLAHE) dari Percobaan 1

Dari `results/tables/T2_scenarios.csv` (Holm 9 uji): CLAHE bawaan (S2) bermakna hanya di Darwin (+0,034), CLAHE tuning (S3) di Darwin (+0,014), HE (S1) merugikan Montgomery (−0,025). Shenzhen tidak berubah. Efek CLAHE jauh lebih kecil daripada efek head per konvensi.

## Yang harus diputuskan bila B dipilih

1. Rumusan RM2 di PROTOCOL diganti, dan RM2 lama dipindah ke lampiran.
2. Laporan menyebut E1/E2 post-hoc dan head Darwin dipilih dari metadata.
3. Kasus Shenzhen 0074–0076 dibahas atau ditandai seperti citra Darwin 00006967.
