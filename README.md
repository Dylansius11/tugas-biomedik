# cxr-clahe-loso

*Peran CLAHE dalam Kesiapan Segmentasi Paru Rontgen Thorax Lintas Rumah Sakit untuk Skrining Tuberkulosis*

Proyek akhir Komputasi Biomedik (Informatika UNS). Kami melatih U-Net pada dua sumber rontgen thorax, mengujinya pada sumber ketiga yang tidak pernah dilihat model, dan bertanya dua hal: apakah CLAHE membantu di sumber baru, dan apakah ada alarm sederhana tanpa label yang memberi tahu kapan model belum siap dipakai.

## Mulai cepat

```bash
git clone <url-repo> && cd cxr-clahe-loso
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                   # 20 tes harus lolos

bash scripts/download_data.sh                         # sekitar 2 GB, cek SHA-256 otomatis
python scripts/prepare_data.py --zip                  # data/processed + manifest.csv + processed.zip
```

Jumlah yang harus muncul: Shenzhen 566, Montgomery 138, Darwin 600 (acak seed 42), total 1.304 pasang citra dan mask. Montgomery berjumlah 138, bukan 139 seperti tertulis di deskripsi koleksi.

Huda dan Nabil tidak perlu mengunduh 2 GB: ambil `processed.zip` dari Drive bersama dan ekstrak ke `data/processed`.

## Siapa mengerjakan apa

| Orang | Rencana | Hasil |
|---|---|---|
| Dylan | [BUILD-PLAN-DYLAN.md](docs/plans/BUILD-PLAN-DYLAN.md) | Kontrak kode, data, proposal, laporan akhir, review |
| Huda | [BUILD-PLAN-HUDA.md](docs/plans/BUILD-PLAN-HUDA.md) | `model.py`, `train.py`, `tune_clahe.py`, 36 pelatihan, `pred.npz` |
| Nabil | [BUILD-PLAN-NABIL.md](docs/plans/BUILD-PLAN-NABIL.md) | QC data, `evaluate.py`, `shift.py`, `stats.py`, tabel dan gambar |

Mulai dari [BUILD-PLAN.md](docs/plans/BUILD-PLAN.md) (cerita proyek, garis waktu, kriteria sukses), lalu [PROTOCOL.md](docs/PROTOCOL.md) (aturan eksperimen yang dikunci).

## Struktur folder

```
cxr-clahe-loso/
  README.md
  configs/default.yaml        semua parameter eksperimen (satu-satunya sumber angka)
  docs/
    PROTOCOL.md               spesifikasi eksperimen
    plans/                    BUILD-PLAN utama dan per orang
  src/cxrseg/
    preprocess.py             S0 sampai S3, binarisasi mask, resize     (siap, diuji)
    postprocess.py            ambang, 2 komponen, closing, isi lubang   (siap, diuji)
    metrics.py                Dice, IoU, HD95, cakupan, galat luas      (siap, diuji)
    data.py                   manifest, logika lipatan                  (siap, diuji)
    model.py dataset.py train.py tune_clahe.py                          (Huda)
    evaluate.py shift.py stats.py                                       (Nabil)
  scripts/
    download_data.sh          unduh koleksi Mendeley + cek SHA-256
    prepare_data.py           bangun data/processed dan manifest
    make_fake_predictions.py  prediksi palsu agar Nabil bisa mulai lebih dulu
    check_runs.py             tabel 36 run yang sudah dan belum selesai
    run_all.py qc_data.py analyze.py                                    (Huda, Nabil)
  tests/                      pytest
  notebooks/                  runner Colab (Huda)
  data/                       raw dan processed, tidak masuk git
  results/                    checkpoints dan predictions tidak masuk git; metrics, tables, figures masuk
  paper/proposal/             proposal final (docx, pdf), gambar, skrip pembuat, status referensi
```

## Aturan emas

1. Sumber yang ditahan hanya dipakai untuk uji asing. Bukan untuk validasi, tuning, normalisasi, atau ambang alarm.
2. Tidak ada angka eksperimen di kode di luar `configs/default.yaml`.
3. Angka dari prediksi palsu tidak boleh masuk laporan.
4. Hasil negatif dilaporkan apa adanya.

## Lisensi data

Koleksi dari Danilov dkk. (2022), Mendeley Data, DOI 10.17632/8gf9vpkhgy.2, CC BY 4.0. Data tidak disimpan di repositori ini.
