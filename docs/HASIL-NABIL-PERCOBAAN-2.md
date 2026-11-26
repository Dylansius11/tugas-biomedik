# HASIL NABIL: Percobaan 2, skenario A (analisis tambahan, post-hoc)

26 November 2026, cabang `nabil/exp2-a`. Plan: `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2.md`.
**Semua angka di sini post-hoc.** Lengan E1/E2 dirancang setelah melihat hasil Percobaan 1. Hasil utama laporan tetap T1–T7 (36 run S0–S3).

Luaran: `results/exp2/nabil/` (X0–X7, `metrics/`, `figures/XF1–XF4`). Perintah:
```
python scripts/analyze_exp2.py --mode a --pred-dir <predictions> --exp2-dir <percobaan2> --out results/exp2/nabil
```
Sekitar 5 menit di CPU. Dua kali jalan, semua CSV identik.

## 1. Data dan verifikasi (N7, N8)

- 54 berkas Percobaan 2 dan 9 berkas E0 valid (`X0_validation.csv`). Urutan `ids` dan `group` sama dengan E0. Head untuk Darwin = `visible` di semua 9 run E1/E2. Citra adaptasi few-shot tidak pernah ikut dievaluasi.
- **105 dari 105 angka cocok** (`X1_verify.csv`):
  - Tabel Huda `main.csv` (45 angka), `vs_E0.csv` (12), `fewshot.csv` (36), sampai 3 desimal.
  - Ambang dan tingkat alarm E0 sama persis dengan T5 S0 hasil utama (12 angka).
- Angka Huda bisa dipakai. Satu perbedaan cara: Huda tidak memakai koreksi Holm. Koreksi Holm ada di bawah.

## 2. E2 vs E0 di sumber asing (N9, `X2`, `X3`)

Dice per citra dirata-ratakan atas 3 seed. Wilcoxon berpasangan, CI bootstrap 10.000 ulang, Holm atas 3 uji (E2 vs E0). E1 hanya deskriptif.

| Sumber asing | Selisih Dice E2 − E0 [CI 95%] | p Holm | Bermakna | Gagal E0 → E2 | Selisih gagal [CI 95%] |
|---|---|---|---|---|---|
| Darwin (n = 600) | +0,012 [0,008; 0,016] | 4×10⁻¹⁰ | ya | 97,3% → 97,3% | 0,0 [−0,5; 0,5] poin |
| Montgomery (n = 138) | +0,026 [0,022; 0,030] | 3×10⁻²⁰ | ya | 6,5% → 2,9% | −3,6 [−8,0; 0,7] poin |
| Shenzhen (n = 566) | +0,055 [0,052; 0,058] | 1×10⁻⁸⁴ | ya | 50,5% → 6,5% | −44,0 [−48,2; −39,6] poin |

- **Shenzhen**: perbaikan besar dan jelas. Celah generalisasi turun dari 0,066 ke 0,006.
- **Montgomery**: Dice naik bermakna, tetapi penurunan tingkat gagal belum pasti (CI melewati 0, karena hanya 9 → 4 citra gagal).
- **Darwin**: Dice naik bermakna secara statistik (+0,012, sedikit di atas batas 0,01), tetapi **tidak bermakna secara klinis**. Hampir semua citra tetap gagal karena Darwin memakai definisi paru anatomis yang tidak pernah dilihat model.
- Kepekaan ambang: tingkat gagal pada ambang 0,85/0,90/0,93 ada di `X3_clinical.csv`. Shenzhen E2 pada ambang 0,93 masih 15,7% gagal.

## 3. Alarm A1/A2 pada E2 (N10, `X4`, `X5`, `X7`)

Pertanyaan: apakah alarm tanpa label lebih berguna pada E2? Ambang tau diambil dari validasi dua sumber latih, rata-rata 3 seed per citra (sama dengan hasil utama). A3 diambil dari tabel Huda sebagai pembanding.

| Sumber | Alarm | AUROC E0 → E2 | Tingkat alarm E2 | Label E2 | Keputusan alarm E2 |
|---|---|---|---|---|---|
| Darwin | A1 | 0,51 → 0,46 | 7,2% | belum siap | siap ✗ |
| | A2 | 0,66 → 0,76 | 50,7% | belum siap | belum siap ✓ |
| | A3 (Huda) | 0,71 → 0,75 | 57,8% | belum siap | belum siap ✓ |
| Montgomery | A1 | tidak dapat dihitung | 21,0% | siap | belum siap ✗ |
| | A2 | tidak dapat dihitung | 0,0% | siap | siap ✓ |
| | A3 (Huda) | tidak dapat dihitung | 39,1% | siap | belum siap ✗ |
| Shenzhen | A1 | 0,52 → 0,54 | 43,5% | belum siap | belum siap ✓ |
| | A2 | 0,63 → 0,87 | 4,1% | belum siap | siap ✗ |
| | A3 (Huda) | 0,60 → 0,76 | 16,5% | belum siap | belum siap ✓ |

**Jawaban:** pada E2, **A2 lebih baik membedakan citra gagal dan lulus** (AUROC Shenzhen 0,63 → 0,87, Darwin 0,66 → 0,76), dan keputusan kesiapannya benar di 2 dari 3 sumber (pada E0 hanya 1 dari 3). **A1 tetap tidak berguna**: skornya hanya bergantung pada citra, AUROC sekitar 0,5. Keputusan A1 juga benar hanya karena kebetulan (Shenzhen) atau salah (Darwin, Montgomery).

Catatan:
- Keputusan A2 yang salah di Shenzhen E2 terjadi di dekat batas: tingkat gagal 6,5% (batas siap 5%) dan alarm 4,1% (batas 10%). Ini kasus "hampir siap".
- Sampel acak 50 citra (`X7`) memberi keputusan yang sama dengan keputusan pada seluruh citra di ≥ 99% sampel untuk A2. A1 Darwin hanya 87%.
- A3 Huda dihitung per seed lalu dirata-ratakan. A1/A2 di sini dirata-ratakan per citra dulu. Pembandingnya mendekati, tidak identik.

## 4. Few-shot (N11, `X6`, `XF2`)

Evaluasi pada citra val+test sumber baru (Darwin 120, Montgomery 28, Shenzhen 114) yang tidak pernah dipakai adaptasi. Kolom `fail_pct` adalah cara Huda (Dice dirata-ratakan per citra lalu dihitung gagal). `fail_pct_seed_mean` adalah rata-rata tingkat gagal per seed.

| Sumber | k = 0 (E2) | k = 5 | k = 10 | k = 20 | k = 50 |
|---|---|---|---|---|---|
| Darwin, gagal rata-rata per seed [rentang] | 95,8% | 46,1% [45,0–46,7] | 28,9% [14,2–40,0] | 19,2% [14,2–24,2] | 13,6% [7,5–18,3] |
| Montgomery | 3,6% | 0% | 0% | 0% | 0% |
| Shenzhen | 7,6% | 9,9% | 9,6% | 7,6% | 7,9% |

- Darwin: 10–20 citra berlabel sudah menurunkan kegagalan dari sekitar 96% ke sekitar 20–30%. Tetapi rentang antar seed lebar (k = 10: 14–40%), jadi hasilnya bergantung pada citra mana yang dipilih.
- Shenzhen tidak membaik dengan few-shot. Sisa kegagalannya kemungkinan bukan soal konvensi label (lihat butir 5).
- Cara Huda bisa keluar dari rentang per seed (Darwin k = 5: 54,2% padahal per seed 45–47%), karena satu citra bisa lulus di dua seed dan gagal parah di seed ketiga. Gambar XF2 memakai rata-rata per seed.
- Tiga seed few-shot tidak saling bebas (model dasar sama) dan hanya ada satu sumber per lipatan. Kurva ini bukan bukti kuat.

## 5. Kasus terburuk (`XF4`)

- Darwin: model hanya menandai sebagian paru pada rontgen anak dengan pneumonia. Ini sesuai dengan konvensi paru anatomis yang berbeda.
- Shenzhen: tiga kasus terburuk E2 (`CHNCXR_0074_0`, `0075_0`, `0076_0`) adalah rontgen anak AP berurutan. **Prediksi lebih luas dari mask asli**, sedangkan mask asli tampak sempit. Ini kandidat masalah anotasi di sumber, bukan kegagalan model. Perlu dicek Dylan sebelum dibahas di laporan.
- Montgomery: kasus terburuk masih Dice 0,86–0,89, dengan kesalahan di sudut kostofrenikus.

## Keterbatasan

- Post-hoc: lengan E1/E2 dan ambang pemilihan lengan ditetapkan setelah Percobaan 1.
- Head Darwin dipilih dari metadata definisi label. Untuk sumber tanpa dokumentasi perlu cara lain.
- Montgomery hanya 138 citra asing dengan 4–9 citra gagal, jadi AUROC tidak dapat dihitung.
