# BUILD-PLAN: Nabil, analisis Percobaan 2 skenario B (Percobaan 2 sebagai hasil utama)

Cabang `nabil/exp2-b`. Pasangan dari `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2.md` (skenario A, cabang `nabil/exp2-a`). Hanya satu dari keduanya yang di-merge (`docs/plans/BRANCHING-PERCOBAAN-2.md`).

Tugasmu satu kalimat: **susun tabel dan gambar utama laporan dari lengan E0, E1, E2 dengan aturan yang sama seperti T1–T7, sehingga laporan bisa menjadikan Percobaan 2 hasil utama bila Dylan memilih B.**

## Bedanya dengan skenario A

| | A (tambahan) | B (utama) |
|---|---|---|
| Lengan yang diuji | E2 vs E0 (Holm 3 uji), E1 deskriptif | E1 dan E2 vs E0 (Holm 6 uji), sama seperti T2 yang menguji S1–S3 vs S0 (9 uji) |
| Alarm A1/A2 | E0 dan E2 | E0, E1, E2 |
| Nama luaran | X0–X7, XF1–XF4 di `results/exp2/nabil/` | T0–T9, F1–F5 di `results/exp2/main/` (pola nama hasil utama) |
| Percobaan 1 | Tetap hasil utama | Pendukung. CLAHE (T4, F4, F6, S1–S3) menjadi lampiran |

## Pertanyaan riset bila B dipilih (usulan, perlu persetujuan Dylan)

| | PROTOCOL sekarang | Usulan B |
|---|---|---|
| RM1 | Seberapa besar celah generalisasi ke sumber asing? | Sama, dijawab dari E0 (= S0) |
| RM2 | Apakah CLAHE mengurangi kegagalan di sumber asing? | **Apakah model dua head (konvensi label) + augmentasi kuat mengurangi kegagalan di sumber asing?** RM2 lama dijawab di lampiran dari Percobaan 1 (T2: CLAHE S2 bermakna hanya di Darwin, +0,034; HE merugikan Montgomery, −0,025; Shenzhen tidak berubah) |
| RM3 | Apakah alarm tanpa label bisa memberi tahu kesiapan? | Sama, diuji pada E0, E1, E2 |

Wajib ditulis di laporan: lengan E1/E2 dirancang **setelah** melihat hasil Percobaan 1. Jadi walaupun menjadi hasil utama, temuan ini post-hoc dan perlu dikonfirmasi pada sumber keempat.

## Langkah

| Langkah | Isi | Luaran |
|---|---|---|
| B0 | Cherry-pick skrip `analyze_exp2.py` + tes dari `nabil/exp2-a` | |
| B1 | Plan ini | `docs/plans/BUILD-PLAN-NABIL-PERCOBAAN-2-B.md` |
| B2 | Mode `--mode main`: T0 validasi, T1 celah, T2 lengan vs E0 (Holm 6), T3 klinis, T5 alarm, T6 kesiapan, T7 sampel 50, T8 few-shot, T9 verifikasi | `results/exp2/main/tables/` |
| B3 | Gambar F1 boxplot, F2 celah, F3 alarm E2, F4 few-shot, F5 kasus terburuk E2 + dokumen hasil | `results/exp2/main/figures/`, `docs/HASIL-NABIL-PERCOBAAN-2-B.md` |

Tidak ada T4 (tuning CLAHE) karena Percobaan 2 hanya memakai S0.

## Aturan

Sama dengan skenario A: tau hanya dari validasi dua sumber latih, head Darwin `visible` dari metadata, `scripts/analyze.py` dan `results/tables` tidak diubah, semua angka berlabel post-hoc.
