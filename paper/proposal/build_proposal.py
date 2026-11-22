"""Build the final proposal (docx) from the content below.

Usage: python paper/proposal/build_proposal.py
Writes paper/proposal/Proposal_Proyek_Akhir_Komputasi_Biomedik_v2.docx and prints word counts
for the limits in the assignment (title <= 14 words, background <= 1000, methods <= 1000).
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).parent
FIG = HERE / "figures"
OUT = HERE / "Proposal_Proyek_Akhir_Komputasi_Biomedik_v2.docx"
FONT = "Times New Roman"

TITLE = "Peran CLAHE dalam Kesiapan Segmentasi Paru Rontgen Thorax Lintas Rumah Sakit untuk Skrining Tuberkulosis"

AUTHORS = [
    ("DYLANSIUS PUTRA PRASETIO", "NIM. L0123046"),
    ("NABIL RIANO ZAKY", "NIM. "),
    ("MUH MISBAH ULHUDA", "NIM. L0123085"),
]

# Content model: ("h1"|"h2", text) ("p", text) ("item", text) ("table", spec) ("fig", spec)
LATAR = [
    ("h2", "2.1 Latar Belakang Masalah"),
    ("p", "Tuberkulosis (TB) masih menjadi beban kesehatan besar di Indonesia. Pada 2023 Indonesia menyumbang sekitar 10% kasus TB dunia, kedua setelah India. Survei inventori nasional memperkirakan 14% kasus TB tahun itu belum terdiagnosis (WHO, 2024). Skrining dengan rontgen thorax adalah salah satu cara menemukan kasus lebih awal, dan kini sering dibantu perangkat lunak yang membaca citra secara otomatis."),
    ("p", "Banyak sistem seperti itu berangkat dari segmentasi paru, yaitu memisahkan area paru dari tulang rusuk, jantung, dan latar. Bagian paru yang terlewat oleh segmentasi tidak ikut diperiksa, dan kesalahan di langkah ini terbawa ke langkah sesudahnya. Bassi dan Attux (2022) menunjukkan bahwa menambahkan segmentasi paru sebelum klasifikasi COVID-19 menaikkan akurasi pada 150 citra uji eksternal dari 74% menjadi 78,7%."),
    ("p", "Masalahnya, model biasanya dilatih di satu tempat lalu dipakai di tempat lain. Rumah sakit berbeda dalam alat rontgen, paparan, resolusi, dan cara anotasi, dan pergeseran sebaran data seperti ini diketahui menurunkan performa model (Zhou dkk., 2023). Qin dkk. (2024) membandingkan 12 produk deteksi TB berbantuan komputer pada data survei di Afrika Selatan dan mencatat bahwa pemilihan ambang antar produk di populasi yang berbeda masih kurang dipahami. Pada klasifikasi COVID-19, akurasi pada data eksternal jauh di bawah angka validasi internal yang sering melewati 90% (Bassi & Attux, 2022). Karena itu, sebelum sebuah model dipakai di rumah sakit baru, pertanyaan yang paling berguna adalah apakah model itu sudah siap dipakai di sana."),
    ("p", "Citra rontgen sering berkontras rendah, sehingga batas paru dengan jaringan di sekitarnya tidak selalu tegas. Karena itu peningkatan kontras umum dipakai pada pra-pemrosesan, dan yang paling sering adalah *Contrast Limited Adaptive Histogram Equalization* (CLAHE). CLAHE membagi citra menjadi ubin kecil, menyetarakan histogram tiap ubin, lalu membatasi puncak histogram dengan *clip limit* supaya noise tidak ikut menguat (Mohd-Isa dkk., 2021). Hasilnya bergantung pada dua pengaturan, yaitu *clip limit* dan ukuran grid ubin. Yang belum jelas, apakah CLAHE yang menaikkan skor pada data sendiri juga menolong model di sumber data baru, atau hanya membuatnya makin cocok dengan data latihnya."),
    ("h2", "2.2 Penelitian yang Sudah Dilakukan"),
    ("p", "Kombinasi CLAHE dan U-Net sudah banyak dicoba. Mabina dkk. (2025) melaporkan Dice 95,87% dan IoU 92,07% untuk U-Net dengan CLAHE pada dataset Shenzhen dan Montgomery. Ahmad dkk. (2025) membandingkan CLAHE, *histogram equalization*, dan *anisotropic diffusion* pada segmentasi X-ray tulang belakang. Peningkatannya signifikan pada U-Net, tetapi tidak terlihat pada Mask R-CNN, jadi manfaat peningkatan kontras tampaknya bergantung pada modelnya. Wang dkk. (2024) membandingkan CLAHE dengan ps-KDE pada 247 citra JSRT. ps-KDE unggul di paru kiri (Dice 0,780 dibanding 0,717), sedangkan CLAHE lebih baik di paru kanan dan klavikula. Para penulis sendiri mencatat belum adanya validasi eksternal."),
    ("p", "Untuk data, Danilov dkk. (2022) menyusun koleksi 6.810 citra rontgen dengan *mask* paru dari tiga sumber, yaitu Darwin, Montgomery, dan Shenzhen. Din dkk. (2025) memakai ketiganya untuk jaringan CXR-Seg, tetapi tiap dataset dievaluasi dengan pembagian datanya sendiri. Xue dkk. (2023) menganalisis pergeseran domain pada deteksi area paru di lima dataset rontgen dari sumber berbeda, dengan membandingkan metadata, tampilan citra, dan fitur yang diambil dari jaringan terlatih."),
    ("p", "Bukti tentang pengaruh pra-pemrosesan terhadap generalisasi lebih banyak datang dari klasifikasi, dan hasilnya campuran. Azhar dkk. (2025) melaporkan model COVID-19 dengan CLAHE punya akurasi latih tertinggi (93,26%), tetapi akurasi validasinya (91,31%) kalah dari model tanpa pra-pemrosesan (91,50%). Singthongchai dan Wangkhamhan (2025) menemukan bahwa normalisasi adaptif membantu pada dataset dengan kontras stabil, tetapi hampir tidak menolong pada MIMIC-CXR yang heterogen. Untuk mengenali kegagalan model tanpa label, González dkk. (2022) memakai jarak Mahalanobis pada ruang fitur guna mendeteksi kegagalan diam-diam pada segmentasi lesi COVID-19 di CT."),
    ("h2", "2.3 Celah Penelitian dan Inovasi"),
    ("p", "Dari penelitian di atas ada empat hal yang belum terjawab. Pertama, hasil CLAHE pada segmentasi paru umumnya dilaporkan pada data yang satu sumber dengan data latih (Mabina dkk., 2025; Wang dkk., 2024). Kedua, bukti tentang generalisasi berasal dari klasifikasi dan arahnya tidak seragam. Ketiga, *clip limit* dan ukuran grid belum diperlakukan sebagai variabel yang diuji pada sumber data lain. Keempat, belum ada aturan sederhana yang memberi tahu rumah sakit baru apakah model sudah siap dipakai pada citra mereka tanpa harus menyiapkan *mask*. Dalam penelusuran kami atas artikel 2021 sampai 2026, kami belum menemukan studi yang menguji CLAHE pada segmentasi paru dengan skema *leave-one-source-out*, apalagi yang menghubungkannya dengan peringatan tanpa label. Inovasi penelitian ini ada pada empat hal:"),
    ("item", "(1) Protokol uji lintas sumber. Tiga sumber data bergantian menjadi sumber uji yang sama sekali tidak dilihat model. *Clip limit* dan ukuran grid dituning hanya dengan data validasi dari sumber latih."),
    ("item", "(2) Ukuran yang dekat dengan kebutuhan skrining. Selain Dice, IoU, dan HD95, kami mengukur cakupan paru, galat luas paru, dan tingkat kegagalan. Dice pada segmentasi paru sudah mendekati batas atas, sehingga sering tidak lagi membedakan perlakuan."),
    ("item", "(3) Alarm tanpa label dan aturan kesiapan. Tiap citra diberi skor pergeseran (jarak sebaran intensitas dan ketidakpastian model) untuk memperingatkan citra yang kemungkinan gagal dan memutuskan sumber baru siap atau belum, tanpa *mask*. Jarak sebaran intensitas juga diukur sebelum dan sesudah CLAHE untuk melihat apakah CLAHE benar-benar mendekatkan sumber-sumber itu."),
    ("item", "(4) Mudah direproduksi. Seluruh eksperimen berjalan di Google Colab dengan GPU T4, dengan kode dan konfigurasi yang terbuka."),
    ("h2", "2.4 Penelitian yang Akan Dilakukan"),
    ("p", "Kami melatih U-Net pada dua sumber dan mengujinya pada sumber ketiga, lalu mengulanginya sampai setiap sumber pernah menjadi sumber uji. Pada tiap putaran, empat skenario kontras dibandingkan: tanpa peningkatan kontras, *histogram equalization* global, CLAHE bawaan, dan CLAHE hasil tuning. Performa diukur dengan enam ukuran dan diuji secara statistik, lalu skor alarm dicocokkan dengan kegagalan yang sebenarnya. Shenzhen dan Montgomery berfokus pada TB, sedangkan Darwin berasal dari koleksi rontgen COVID-19 dan pneumonia viral, sehingga Darwin berperan sebagai uji tekanan pada sumber yang sangat berbeda. Istilah lintas rumah sakit di sini berarti lintas sumber data dari institusi dan alat yang berbeda. Klaim penelitian ini terbatas pada kesiapan segmentasi paru lintas sumber, bukan pada diagnosis TB. Luarannya berupa artikel dan *notebook* yang dapat dijalankan ulang."),
]

RUMUSAN = [
    ("RM1.", "Seberapa besar penurunan performa U-Net (Dice, cakupan paru, dan tingkat kegagalan) ketika diuji pada sumber data yang tidak dipakai saat pelatihan, dibandingkan dengan pengujian pada sumber yang sama?"),
    ("RM2.", "Apakah CLAHE, baik dengan parameter bawaan maupun hasil tuning, mengubah performa dan tingkat kegagalan segmentasi pada sumber data asing dibandingkan dengan citra tanpa peningkatan kontras dan *histogram equalization* global, dan bagaimana *clip limit* serta ukuran grid memengaruhinya?"),
    ("RM3.", "Apakah skor pergeseran per citra yang dihitung tanpa *mask* dapat memperingatkan citra yang gagal, dan cukupkah skor itu untuk memutuskan siap atau belum siapnya model di sumber data baru?"),
]

TUJUAN = [
    ("T1.", "Mengukur celah generalisasi, yaitu selisih performa antara pengujian pada sumber yang sama dan pada sumber asing untuk setiap skenario kontras."),
    ("T2.", "Membandingkan performa dan tingkat kegagalan pada sumber asing antara empat skenario kontras, menguji perbedaannya secara statistik, dan menentukan kombinasi *clip limit* dan ukuran grid terbaik pada data validasi."),
    ("T3.", "Menguji apakah skor pergeseran per citra mampu memperingatkan citra yang gagal, dan merumuskan aturan kesiapan yang dapat dipakai tanpa *mask*."),
]

METODE = [
    ("h2", "5.1 Gambaran Umum"),
    ("p", "Penelitian ini adalah eksperimen komputasi dengan satu alur yang sama untuk semua skenario (Gambar 1). Hanya perlakuan kontras yang diubah, sehingga perbedaan hasil dapat dikaitkan dengan kontras. Semua aturan dikunci sebelum pelatihan dan ditulis di repositori proyek."),
    ("fig", {"file": "gambar1_alur.png", "caption": "Gambar 1. Alur penelitian", "width": 15.0}),
    ("h2", "5.2 Data"),
    ("p", "Data berasal dari koleksi *Chest X-ray dataset for lung segmentation* (Danilov dkk., 2022; Mendeley Data, CC BY 4.0, https://doi.org/10.17632/8gf9vpkhgy.2) yang memuat citra dan *mask* paru biner dari tiga sumber (Tabel 1). Darwin dikurangi menjadi 600 sampel acak (*seed* 42) agar tidak mendominasi. *Mask* dalam koleksi membawa sedikit noise anti-aliasing (sekitar 0,1% piksel bernilai 1 sampai 254), jadi semuanya dibinerkan pada ambang 127. Kami memeriksa 20 sampel per sumber secara visual dan mencari citra duplikat antar sumber dengan *perceptual hash*. Duplikat lintas sumber dibuang dari sumber uji."),
    ("table", {
        "caption": "Tabel 1. Sumber data yang dipakai",
        "header": ["Sumber", "Dipakai", "Asal", "Peran"],
        "rows": [
            ["Shenzhen", "566", "Shenzhen No.3 People's Hospital, Cina", "Inti skrining TB"],
            ["Montgomery", "138", "Program TB Montgomery County, AS", "Inti skrining TB"],
            ["Darwin", "600 dari 6.106", "Koleksi COVID-19 dan pneumonia viral; alat dan resolusi beragam, termasuk rontgen portabel", "Uji tekanan"],
        ],
        "widths": [2.5, 2.7, 6.8, 3.0],
        "note": "Sumber: deskripsi koleksi (Danilov dkk., 2022). Deskripsi menyebut 139 citra Montgomery, berkas yang diunduh berisi 138 pasang citra dan *mask*.",
    }),
    ("h2", "5.3 Skema Pembagian Data"),
    ("p", "Kami memakai skema *leave-one-source-out* (Gambar 2). Tiap sumber dibagi acak menjadi 80% latih, 10% validasi, dan 10% uji dalam sumber. Pada tiap lipatan, dua sumber menjadi data latih dan validasi, sedangkan sumber ketiga seluruhnya menjadi uji sumber asing. Sumber asing tidak dipakai untuk validasi, tuning, maupun menentukan ambang alarm. Satu model menghasilkan dua skor, dan selisihnya kami sebut celah generalisasi. Metadata pasien tidak tersedia, jadi satu pasien bisa muncul di lebih dari satu bagian pada pembagian dalam sumber. Risiko ini tidak menyentuh uji sumber asing yang menjadi dasar kesimpulan utama."),
    ("fig", {"file": "gambar2_loso.png", "caption": "Gambar 2. Skema leave-one-source-out", "width": 14.0}),
    ("h2", "5.4 Pra-pemrosesan"),
    ("p", "Semua citra diubah ke *grayscale* dan ukuran 256×256 (interpolasi area untuk citra karena citra Darwin bisa sampai 5600×4700 piksel, *nearest neighbor* untuk *mask*). Setelah itu dibuat empat skenario (Tabel 2). Augmentasi hanya geometris (rotasi sampai 10 derajat, skala 0,9 sampai 1,1, translasi 5%) sehingga tidak bercampur dengan efek kontras."),
    ("table", {
        "caption": "Tabel 2. Skenario peningkatan kontras",
        "header": ["Skenario", "Perlakuan"],
        "rows": [
            ["S0, tanpa peningkatan", "Normalisasi ke rentang [0, 1]."],
            ["S1, HE global", "*Histogram equalization* global, lalu normalisasi."],
            ["S2, CLAHE bawaan", "*Clip limit* 2,0 dan grid 8×8."],
            ["S3, CLAHE tuning", "Pencarian *clip limit* {1, 2, 3, 4} dan grid {4×4, 8×8, 16×16}, 12 kombinasi. Tiap kombinasi dilatih 20 epoch dengan satu *seed* di setiap lipatan, dan yang Dice validasinya tertinggi dipakai untuk pelatihan penuh. Data validasi hanya berasal dari dua sumber latih."],
        ],
        "widths": [4.0, 11.0],
        "note": "",
    }),
    ("h2", "5.5 Model dan Pelatihan"),
    ("p", "Model utama adalah U-Net empat tingkat (32 sampai 256 filter, *bottleneck* 512) dengan *batch normalization* dan *skip connection* berupa konkatenasi. *Loss* berupa BCE ditambah Dice *loss*. Optimizer Adam dengan *learning rate* 0,001 yang dibagi dua bila Dice validasi tidak membaik selama 5 epoch, *batch size* 16, maksimal 60 epoch, dan *early stopping* setelah 10 epoch. Konfigurasinya sengaja sederhana dan sama untuk semua skenario, karena U-Net yang konsisten sudah menjadi pembanding kuat (Isensee dkk., 2021). Setiap kombinasi skenario dan lipatan dilatih dengan tiga *seed* (11, 22, 33), total 36 pelatihan penuh. Sebagai pemeriksaan tambahan, Attention U-Net (Mustapha dkk., 2025) dilatih pada S0 dan skenario CLAHE terbaik dengan satu *seed*."),
    ("h2", "5.6 Pasca-pemrosesan dan Evaluasi"),
    ("p", "Keluaran model diberi ambang 0,5, dipertahankan dua komponen terbesar, ditutup dengan operasi *closing* 5×5, lalu lubangnya diisi. Evaluasi dilakukan per citra pada 256×256 (Tabel 3). Memasangkan metrik tumpang tindih dengan metrik jarak batas mengikuti *Metrics Reloaded* (Maier-Hein dkk., 2024). Tiga ukuran terakhir ditambahkan karena lebih dekat ke kebutuhan skrining. Skor tiap citra dirata-ratakan lintas *seed*."),
    ("table", {
        "caption": "Tabel 3. Ukuran evaluasi per citra",
        "header": ["Ukuran", "Arti"],
        "rows": [
            ["Dice, IoU", "Tumpang tindih prediksi dan *mask* sebenarnya."],
            ["HD95 (piksel)", "Kesalahan batas paru."],
            ["Cakupan paru", "Bagian paru sebenarnya yang tertangkap. Bagian yang hilang tidak diperiksa."],
            ["Galat luas (%)", "Selisih luas prediksi terhadap luas sebenarnya."],
            ["Tingkat kegagalan", "Persentase citra dengan Dice di bawah 0,90 (kepekaan di 0,85 dan 0,93)."],
        ],
        "widths": [3.8, 11.2],
        "note": "",
    }),
    ("h2", "5.7 Alarm Pergeseran dan Aturan Kesiapan"),
    ("p", "Setiap citra diberi dua skor tanpa *mask*. Skor jarak intensitas adalah jarak Wasserstein satu dimensi antara histogram citra itu dan histogram gabungan citra latih pada skenario yang sama. Skor ketidakpastian adalah rata-rata entropi peta probabilitas model. Ambang alarm adalah persentil ke-95 skor pada citra validasi sumber latih. Tingkat alarm adalah persentase citra beralarm pada satu sumber. Dengan *mask*, sumber dinyatakan siap bila tingkat kegagalan paling tinggi 5% dan Dice median turun paling banyak 0,02 dari uji dalam sumber. Tanpa *mask*, sumber dinyatakan siap bila tingkat alarm paling tinggi 10%. Angka 5% dan 10% adalah konvensi awal yang dikunci sebelum melihat hasil. Alarm dievaluasi dengan korelasi Spearman dan AUROC per citra, tabel kecocokan siap dan alarm pada 12 kasus asing ditambah 12 kontrol dalam sumber, serta 1.000 sampel acak berisi 50 citra per sumber untuk melihat berapa citra tanpa *mask* yang cukup. Jarak intensitas antar sumber juga dihitung sebelum dan sesudah tiap skenario."),
    ("h2", "5.8 Analisis Statistik"),
    ("p", "Perbandingan S1 sampai S3 terhadap S0 pada setiap sumber asing memakai uji Wilcoxon berpasangan dengan koreksi Holm atas 9 uji, ditambah selisih rata-rata Dice beserta selang kepercayaan 95% hasil *bootstrap* (10.000 pengambilan ulang). Selisih dianggap bermakna bila p terkoreksi di bawah 0,05 dan besarnya minimal 0,01 Dice. Karena hanya ada tiga sumber, hubungan jarak intensitas dengan celah generalisasi dilaporkan secara deskriptif tanpa uji korelasi."),
    ("h2", "5.9 Lingkungan Kerja"),
    ("p", "Implementasi memakai Python, PyTorch, OpenCV, SciPy, dan Albumentations di Google Colab dengan GPU T4. *Seed* dikunci, *checkpoint* disimpan ke Google Drive, dan kode serta konfigurasi diunggah ke repositori GitHub."),
]

REFS = [
    "Ahmad, M. S. Z., Aziz, N. A. A., Lim, H. S., Ghazali, A. K., & Latiff, A. A. (2025). Impact of image enhancement using contrast-limited adaptive histogram equalization (CLAHE), anisotropic diffusion, and histogram equalization on spine X-ray segmentation with U-Net, Mask R-CNN, and transfer learning. *Algorithms, 18*(12), 796. https://doi.org/10.3390/a18120796",
    "Azhar, A. N. bin, Sani, N. S., & Wei, L. L. X. (2025). Enhancing COVID-19 detection in X-ray images through deep learning models with different image preprocessing techniques. *International Journal of Advanced Computer Science and Applications, 16*(1). https://doi.org/10.14569/IJACSA.2025.0160161",
    "Bassi, P. R. A. S., & Attux, R. (2022). Covid-19 detection using chest X-rays: Is lung segmentation important for generalization? *Research on Biomedical Engineering, 38*(4), 1121-1139. https://doi.org/10.1007/s42600-022-00242-y",
    "Danilov, V. V., Litmanovich, D., Proutski, A., Kirpich, A., Nefaridze, D., Karpovsky, A., & Gankin, Y. (2022). Automatic scoring of COVID-19 severity in X-ray imaging based on a novel deep learning workflow. *Scientific Reports, 12*, 12791. https://doi.org/10.1038/s41598-022-15013-z",
    "Din, S., Shoaib, M., & Serpedin, E. (2025). CXR-Seg: A novel deep learning network for lung segmentation from chest X-ray images. *Bioengineering, 12*(2), 167. https://doi.org/10.3390/bioengineering12020167",
    "González, C., Gotkowski, K., Fuchs, M., Bucher, A., Dadras, A., Fischbach, R., Kaltenborn, I. J., & Mukhopadhyay, A. (2022). Distance-based detection of out-of-distribution silent failures for Covid-19 lung lesion segmentation. *Medical Image Analysis, 82*, 102596. https://doi.org/10.1016/j.media.2022.102596",
    "Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods, 18*(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z",
    "Mabina, I. F., Sari, C. A., & Rachmawanto, E. H. (2025). Lung segmentation in X-ray images of tuberculosis patients using U-Net with CLAHE preprocessing. *Journal of Applied Informatics and Computing, 9*(4), 1346-1353. https://doi.org/10.30871/jaic.v9i4.9869",
    "Maier-Hein, L., Reinke, A., Godau, P., Tizabi, M. D., Buettner, F., Christodoulou, E., Glocker, B., Isensee, F., dkk. (2024). Metrics reloaded: Recommendations for image analysis validation. *Nature Methods, 21*(2), 195-212. https://doi.org/10.1038/s41592-023-02151-z",
    "Mohd-Isa, W.-N., Joseph, J., Hashim, N., & Salih, N. (2021). Enhancement of digitized X-ray films using contrast-limited adaptive histogram equalization (CLAHE). *F1000Research, 10*, 1051. https://doi.org/10.12688/f1000research.73236.1",
    "Mustapha, B., Zhou, Y., Nawel, B., Chunyan, S., & Zhitao, X. (2025). Optimized attention U-Net for enhanced lung and area of infection segmentation in chest X-rays and CT scans. *Journal of Radiation Research and Applied Sciences, 18*(3), 101650. https://doi.org/10.1016/j.jrras.2025.101650",
    "Qin, Z. Z., Van der Walt, M., Moyo, S., Ismail, F., Maribe, P., Denkinger, C. M., Zaidi, S., Barrett, R., Mvusi, L., Mkhondo, N., Zuma, K., Manda, S., Koeppel, L., Mthiyane, T., & Creswell, J. (2024). Computer-aided detection of tuberculosis from chest radiographs in a tuberculosis prevalence survey in South Africa: External validation and modelled impacts of commercially available artificial intelligence software. *The Lancet Digital Health, 6*(9), e605-e613. https://doi.org/10.1016/S2589-7500(24)00118-3",
    "Singthongchai, J., & Wangkhamhan, T. (2025). Adaptive normalization enhances the generalization of deep learning model in chest X-ray classification. *Journal of Imaging, 12*(1), 14. https://doi.org/10.3390/jimaging12010014",
    "Wang, Y., Guo, Y., Wang, Z., Yu, L., Yan, Y., & Gu, Z. (2024). Enhancing semantic segmentation in chest X-ray images through image preprocessing: ps-KDE for pixel-wise substitution by kernel density estimation. *PLoS ONE, 19*(6), e0299623. https://doi.org/10.1371/journal.pone.0299623",
    "World Health Organization. (2024). *Global tuberculosis report 2024*. World Health Organization. https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/tb-reports/global-tuberculosis-report-2024",
    "Xue, Z., Yang, F., Rajaraman, S., Zamzmi, G., & Antani, S. (2023). Cross dataset analysis of domain shift in CXR lung region detection. *Diagnostics, 13*(6), 1068. https://doi.org/10.3390/diagnostics13061068",
    "Zhou, K., Liu, Z., Qiao, Y., Xiang, T., & Loy, C. C. (2023). Domain generalization: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(4), 4396-4415. https://doi.org/10.1109/TPAMI.2022.3195549",
]

# ----------------------------------------------------------------------------- helpers


def words(text: str) -> int:
    return len(re.findall(r"\S+", text.replace("*", "")))


def add_runs(par, text: str, size: float | None = None, bold: bool = False) -> None:
    for i, chunk in enumerate(text.split("*")):
        if not chunk:
            continue
        run = par.add_run(chunk)
        run.italic = i % 2 == 1
        run.bold = bold
        run.font.name = FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        if size:
            run.font.size = Pt(size)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, indent=None, before=0, after=0, hanging=None, left=None, keep_next=False):
    pf = par.paragraph_format
    par.alignment = align
    pf.line_spacing = spacing
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if indent is not None:
        pf.first_line_indent = Cm(indent)
    if hanging is not None:
        pf.left_indent = Cm(hanging)
        pf.first_line_indent = Cm(-hanging)
    if left is not None:
        pf.left_indent = Cm(left)
    pf.keep_with_next = keep_next


def set_cell_border(cell, **kwargs) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, (sz, val) in kwargs.items():
        el = borders.find(qn(f"w:{edge}"))
        if el is None:
            el = OxmlElement(f"w:{edge}")
            borders.append(el)
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), "000000")


def add_page_number(section) -> None:
    footer = section.footer
    footer.is_linked_to_previous = False
    par = footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    run.font.name = FONT
    run.font.size = Pt(12)


def build() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(3.0))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = FONT, Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    for name, size in (("Heading 1", 12), ("Heading 2", 12)):
        st = doc.styles[name]
        st.font.name, st.font.size, st.font.bold, st.font.italic = FONT, Pt(size), True, False
        st.font.color.rgb = None
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.element.rPr.rFonts.set(qn("w:ascii"), FONT)
        st.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)

    # --- cover
    sec.different_first_page_header_footer = False
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, before=6)
    add_runs(p, TITLE.upper(), 14, True)
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, before=14)
    add_runs(p, "PROPOSAL PROYEK AKHIR\nMATA KULIAH KOMPUTASI BIOMEDIK", 12, True)
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=18, after=18)
    p.add_run().add_picture(str(FIG / "logo_uns.png"), width=Cm(5.2))
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, before=6, after=6)
    add_runs(p, "Disusun oleh:", 12)
    for name, nim in AUTHORS:
        p = doc.add_paragraph()
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=8)
        add_runs(p, name, 12, True)
        p.add_run().add_break()
        add_runs(p, nim, 11, True)
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.15, before=96)
    add_runs(p, "PROGRAM STUDI INFORMATIKA\nFAKULTAS TEKNOLOGI INFORMASI DAN SAINS DATA\nUNIVERSITAS SEBELAS MARET\nSURAKARTA\n2026", 12, True)

    # --- body section with page numbers
    body = doc.add_section(WD_SECTION.NEW_PAGE)
    body.footer.is_linked_to_previous = False
    add_page_number(body)
    sec.footer.is_linked_to_previous = False

    counts = {"judul": words(TITLE), "latar": 0, "metode": 0}

    def heading(text, level):
        p = doc.add_paragraph(style=f"Heading {level}")
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.5, before=6 if level == 1 else 4, keep_next=True)
        add_runs(p, text, 12, True)

    def render(items, counter):
        for kind, payload in items:
            if kind == "h2":
                heading(payload, 2)
            elif kind == "p":
                p = doc.add_paragraph()
                fmt(p, indent=1.0)
                add_runs(p, payload, 12)
                if counter:
                    counts[counter] += words(payload)
            elif kind == "item":
                p = doc.add_paragraph()
                fmt(p, hanging=1.0, left=1.0)
                p.paragraph_format.first_line_indent = Cm(-0.8)
                add_runs(p, payload, 12)
                if counter:
                    counts[counter] += words(payload)
            elif kind == "fig":
                p = doc.add_paragraph()
                fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=6, keep_next=True)
                p.add_run().add_picture(str(FIG / payload["file"]), width=Cm(payload["width"]))
                c = doc.add_paragraph()
                fmt(c, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=2, after=8)
                add_runs(c, payload["caption"], 10)
            elif kind == "table":
                cap = doc.add_paragraph()
                fmt(cap, WD_ALIGN_PARAGRAPH.CENTER, 1.0, before=6, after=3, keep_next=True)
                add_runs(cap, payload["caption"], 10)
                header, rows, widths = payload["header"], payload["rows"], payload["widths"]
                t = doc.add_table(rows=1 + len(rows), cols=len(header))
                t.alignment = WD_TABLE_ALIGNMENT.CENTER
                t.autofit = False
                tbl_pr = t._tbl.tblPr
                layout = OxmlElement("w:tblLayout")
                layout.set(qn("w:type"), "fixed")
                tbl_pr.append(layout)
                for c_i, wcm in enumerate(widths):
                    t.columns[c_i].width = Cm(wcm)
                for r_i, row in enumerate([header] + rows):
                    trPr = t.rows[r_i]._tr.get_or_add_trPr()
                    trPr.append(OxmlElement("w:cantSplit"))
                    for c_i, text in enumerate(row):
                        cell = t.cell(r_i, c_i)
                        cell.width = Cm(widths[c_i])
                        par = cell.paragraphs[0]
                        fmt(par, WD_ALIGN_PARAGRAPH.LEFT, 1.0, before=2, after=2, keep_next=(r_i < len(rows)))
                        add_runs(par, text, 10, bold=(r_i == 0))
                        edges = {}
                        if r_i == 0:
                            edges = {"top": (12, "single"), "bottom": (6, "single")}
                        if r_i == len(rows):
                            edges["bottom"] = (12, "single")
                        set_cell_border(cell, **edges)
                if counter:
                    counts[counter] += words(" ".join(header)) + sum(words(" ".join(r)) for r in rows)
                    counts[counter] += words(payload["caption"])
                if payload["note"]:
                    n = doc.add_paragraph()
                    fmt(n, WD_ALIGN_PARAGRAPH.LEFT, 1.0, before=3, after=8)
                    add_runs(n, payload["note"], 9)
                    if counter:
                        counts[counter] += words(payload["note"])
                else:
                    sp = doc.add_paragraph()
                    fmt(sp, WD_ALIGN_PARAGRAPH.LEFT, 1.0, after=4)

    heading("1. JUDUL", 1)
    p = doc.add_paragraph()
    fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.5, indent=0)
    add_runs(p, TITLE, 12, True)
    heading("2. LATAR BELAKANG", 1)
    render(LATAR, "latar")
    heading("3. RUMUSAN MASALAH", 1)
    for tag, text in RUMUSAN:
        p = doc.add_paragraph()
        fmt(p, hanging=1.1)
        add_runs(p, tag + " ", 12, True)
        add_runs(p, text, 12)
    heading("4. TUJUAN", 1)
    p = doc.add_paragraph()
    fmt(p, indent=1.0)
    add_runs(p, "Setiap tujuan menjawab rumusan masalah dengan nomor yang sama.", 12)
    for tag, text in TUJUAN:
        p = doc.add_paragraph()
        fmt(p, hanging=1.1)
        add_runs(p, tag + " ", 12, True)
        add_runs(p, text, 12)
    heading("5. METODE", 1)
    render(METODE, "metode")
    heading("6. REFERENSI", 1)
    for ref in REFS:
        p = doc.add_paragraph()
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.15, after=6, hanging=1.0)
        add_runs(p, ref, 12)

    doc.save(OUT)
    print(f"wrote {OUT.name}")
    print(f"judul : {counts['judul']} kata (maks 14)")
    print(f"latar : {counts['latar']} kata (maks 1000)")
    print(f"metode: {counts['metode']} kata (maks 1000, termasuk isi tabel dan keterangan)")
    print(f"referensi: {len(REFS)}")


if __name__ == "__main__":
    build()
