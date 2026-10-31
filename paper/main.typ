#import "@preview/juti:0.1.1"
#import "setup.typ": setup

#let authors = juti.init-authors((
  (
    name: "Abya",
    institution-ref: 0,
    contribution-refs: (0, 1, 2, 3, 4, 5, 7, 8, 10),
  ),
))

#let institutions = (
  (
    name: "Peneliti Independen",
    address: "Indonesia",
  ),
)

#show: juti.template.with(
  title: "Kalibrasi Sensitivitas TB pada Radiografi Toraks di Situs Baru dengan Prosedur Sertifikasi atau Penolakan",
  authors: authors,
  corresponding-ref: 0,
  corresponding-email: "abyalaxx@gmail.com",
  institutions: institutions,
  abstract: [
    *Latar belakang:* Ambang skor triase tuberkulosis (TB) pada radiografi toraks yang ditetapkan di satu situs belum tentu mempertahankan sensitivitas di situs lain. Konfirmasi mikrobiologis yang selektif juga dapat menghasilkan sampel positif yang tidak mewakili populasi sasaran.

    *Metode:* Kami mengimplementasikan sertifikat eksak berbasis statistik urut kasus positif target (Tier A) untuk sensitivitas minimal 90% dengan probabilitas kegagalan 5%. Prosedur menolak menerbitkan ambang jika jumlah kasus positif kurang dari 29. Fitur dari encoder RAD-DINO dan ResNet50-A1 yang dibekukan dipasangkan dengan kepala regresi logistik dan perceptron multilapis (MLP) yang dilatih hanya pada situs sumber. Evaluasi menggunakan pembagian resmi data TBX11K untuk kalibrasi dan pengujian, serta analisis sekunder dengan Shenzhen sebagai situs sasaran. Kami juga melakukan kontrol pembagian acak tingkat-gambar antarsitus, simulasi verifikasi selektif dengan batas Clopper–Pearson (CP), pembanding Hoeffding berbobot probabilitas terbalik (IPW), dan analisis ukuran lesi pada 68 citra positif beranotasi.

    *Hasil:* Pada validasi TBX11K, ambang sumber RAD-DINO menghasilkan sensitivitas/spesifisitas 91,5%/64,3% untuk regresi logistik dan 95,7%/57,5% untuk MLP. Pada Tier A TBX11K dengan 29 kasus positif untuk kalibrasi, sensitivitas rata-rata masing-masing kepala adalah 96,0% dan 97,0%, dengan spesifisitas 41,8% dan 39,3%. Pada Shenzhen, sensitivitas ambang sumber berada di bawah 90% untuk kedua kepala (79,2% untuk regresi logistik dan 77,4% untuk MLP). Untuk MLP dengan 100 kasus kalibrasi, sensitivitas dan spesifisitas rata-rata Tier A adalah 92,1% dan 62,9%, dengan frekuensi pelanggaran empiris 4,9%. Dalam simulasi verifikasi selektif dengan probabilitas verifikasi kasus berskor rendah 0,10, kalibrasi naif melanggar sasaran sensitivitas pada seluruh 100 pengulangan untuk kedua kepala. Sertifikat empat batas menerbitkan ambang pada 4% pengulangan regresi logistik dan 7% pengulangan MLP, tanpa pelanggaran teramati di antara hasil yang diterbitkan. Pembanding IPW–Hoeffding menolak seluruh ambang. Analisis ukuran lesi memberikan interval kepercayaan lebar dan tidak menunjukkan pola yang meyakinkan.

    *Analisis tambahan:* Audit 10 seed dan 1.000 nested resampling menemukan violation Tier A 4,7-6,7% pada n+=29 pada pool citra terbatas; pemetaan ekstrem 140 label obsolete/unresolved mengubah violation RAD-DINO logistik pada n+=200 dari 0,9% menjadi 8,5%. Dua seed ablation LoRA menghasilkan violation finite-pool 1,2% dan 17,7% pada n+=200 meskipun B0 serupa; perbedaan ini tidak memvalidasi atau menyangkal PAC populasi. Hasil ini bersifat image-level dan tidak memvalidasi asumsi patient-i.i.d.

*Kesimpulan:* Implementasi ini memperjelas batas minimum bukti positif untuk sertifikasi dan memperlihatkan konsekuensi utilitas dari verifikasi selektif. Seluruh hasil merupakan analisis eksploratif tingkat-gambar; keterkaitan pasien belum diverifikasi. Hasil empiris tidak membuktikan asumsi PAC maupun validitas klinis prosedur.
  ],
  keywords: (
    "tuberkulosis",
    "radiografi toraks",
    "kalibrasi ambang",
    "pengendalian risiko konformal",
    "verifikasi selektif",
    "pergeseran domain",
  ),
  meta: (
    received: "Belum diserahkan",
    revised: "-",
    accepted: "-",
    online: "-",
    doi: none,
  ),
  bib: bibliography("references.bib"),
  ..setup,
)

= Pendahuluan

Skor deteksi berbantuan komputer pada radiografi toraks digunakan untuk mendukung skrining dan triase tuberkulosis (TB). Ambang keputusan yang mencapai sensitivitas tertentu pada kohort pengembangan belum tentu mempertahankan sensitivitas tersebut di situs baru dengan karakteristik akuisisi, populasi, atau spektrum penyakit non-TB yang berbeda. Dalam skrining lapangan, verifikasi mikrobiologis juga dapat bergantung pada gejala, skor, atau pertimbangan klinisi. Jika peserta dengan skor lebih rendah lebih jarang diverifikasi, kasus positif yang teramati akan terseleksi dan tidak mewakili populasi sasaran.

Evaluasi multisitus radiografi toraks dan meta-analisis akurasi diagnostik berbasis data individual menunjukkan perlunya menilai kinerja ambang pada populasi yang berbeda @qin2019 @tavaziva2022. Sebuah preprint arXiv eksploratori tahun 2026 juga melaporkan bahwa ambang TBX11K yang dipilih untuk sensitivitas 95% mencapai sensitivitas titik tersebut hanya pada 4 dari 16 evaluasi target @akhtar2026. Penelitian ini mengaudit prosedur ambang berbasis sertifikasi atau penolakan dan verifikasi selektif; penelusuran kami belum lengkap dan tidak menjadi dasar klaim prioritas.

Pengendalian risiko konformal dan koreksi pergeseran kovariat menyediakan dasar bagi kalibrasi dengan kendali risiko hingga sampel terbatas @bates2021 @zecchin2025 @almeida2025 @angelopoulos2025ltt. Runner penelitian membandingkan conformal risk control (CRC) dan importance-weighted learn-then-test (IW-LTT) dengan bobot kepadatan satu serta mengevaluasi ambang pada holdout independen. W-CRC berbobot satu identik secara aljabar dengan CRC, sehingga tidak dilaporkan sebagai baris metode terpisah. Baseline ini tidak mengoreksi pergeseran situs. Rasio kepadatan kondisional positif serta batas bobot valid yang diperlukan untuk analisis shift nyata belum tersedia. RAD-DINO merupakan encoder citra toraks yang dilatih secara swa-supervisi @perezgarcia2025.

Penelitian ini mengkaji alur *sertifikasi atau penolakan*: ambang dikalibrasi menggunakan skor kasus TB positif dari situs sasaran dan diterbitkan hanya jika kriteria sensitivitas kondisional kelas dengan jaminan hingga sampel terpenuhi. Untuk verifikasi selektif, kami membandingkan sertifikat konservatif empat batas dengan metode Hoeffding linear berbobot dalam kebijakan simulasi yang probabilitas verifikasinya diketahui. Penelitian ini tidak mengajukan teorema baru. Kontribusinya adalah implementasi dan audit terhadap asumsi, perilaku penolakan, serta pertukaran utilitas empiris.

Pertanyaan penelitian meliputi: (1) bagaimana sensitivitas, spesifisitas, dan refusal berubah pada kalibrasi positif target berulang; (2) bagaimana hasil pembagian acak campuran situs berbeda dari evaluasi dengan situs yang ditahan sebagai target; (3) bagaimana prosedur naif dan konservatif bekerja ketika verifikasi label referensi pada skor rendah bersifat selektif.

= Metode

== Sumber data, kebijakan label, dan unit analisis

Kohort terolah terdiri atas 9.060 citra: 138 citra Montgomery, 662 citra Shenzhen, dan 8.260 rekaman biner TBX11K. Daftar resmi pelatihan dan validasi TBX11K memuat 8.400 citra; 139 citra dengan anotasi TB yang hanya usang serta satu rekaman TB yang belum terselesaikan dikeluarkan. Label tingkat citra Montgomery dan Shenzhen mengikuti konvensi sufiks nama berkas yang didokumentasikan National Library of Medicine (NLM) @montgomery_cxr_dataset @shenzhen_cxr_dataset. Pada TBX11K, label positif TB aktif diberikan hanya jika terdapat sedikitnya satu anotasi lesi ActiveTuberculosis @liu2023revisiting. Penelitian ini menggunakan daftar resmi pelatihan dan validasi serta tidak menggunakan partisi uji yang tidak berlabel maupun daftar campuran situs.

Protokol utama menggunakan Montgomery dan Shenzhen sebagai situs sumber, kasus positif pada bagian pelatihan TBX11K untuk kalibrasi target, dan bagian validasi TBX11K sebagai evaluasi. Protokol sekunder menahan Shenzhen sebagai target, melatih model pada Montgomery dan bagian pelatihan resmi TBX11K, lalu membagi Shenzhen secara terstratifikasi pada tingkat citra dengan rasio 50:50 untuk kalibrasi dan evaluasi. Masing-masing bagian Shenzhen berisi 168 citra positif dan 163 citra negatif.

Seluruh radiografi dipertahankan tanpa pemotongan lapang paru atau *region of interest* (ROI). Citra dikonversi ke skala keabuan. Intensitas 8-bit dipertahankan, sedangkan citra dengan kedalaman bit lebih tinggi dipetakan ke uint8 menggunakan jendela persentil 0,5–99,5 per citra. Citra kemudian diubah ukurannya menjadi 224 × 224 piksel dengan interpolasi Lanczos dan direplikasi ke tiga kanal. Resolusi cache ini digunakan untuk ResNet50, sedangkan masukan RAD-DINO diproses pada resolusi 518 × 518 piksel. Berkas ROI Montgomery dan Shenzhen hanya menandai lesi positif dan bukan inventaris label lengkap. Manifest mencatat satu baris per citra dan patient_id kosong karena tidak ada crosswalk yang terverifikasi. Unit analisis dan pembagian yang didukung aset saat ini adalah tingkat-gambar; pemisahan pasien belum dibuktikan.

== Encoder dan aturan keputusan pada situs sumber

Penelitian menggunakan encoder ImageNet ResNet50-A1 dan RAD-DINO yang dibekukan pada revisi yang telah dipatok. RAD-DINO menggunakan prosesor beresolusi 518 piksel dan embedding token CLS. Ekstraksi embedding dilakukan pada CUDA dengan inferensi presisi 16-bit (FP16), kemudian disimpan sebagai float32. Regresi logistik dan MLP satu lapis tersembunyi dengan 128 unit dilatih hanya menggunakan citra pelatihan dari situs sumber.

Data validasi sumber digunakan untuk memilih ambang empiris dengan sensitivitas teramati minimal 90% pada kasus positif. Ambang ini merupakan baseline empiris situs sumber, bukan sertifikat untuk situs sasaran; selanjutnya disebut B0. Untuk mengukur informasi situs pada embedding RAD-DINO, kami melatih probe asal situs menggunakan validasi silang terstratifikasi lima lipatan. Probe ini menilai apakah embedding mengandung informasi tentang situs, tetapi tidak menetapkan penyebabnya atau membuktikan adanya *shortcut learning* kausal.

== Sertifikat eksak Tier A

Misalkan $s(x)$ meningkat seiring bukti adanya TB aktif dan prediksi positif diberikan ketika $s(x) >= t$. Untuk populasi sasaran $P_T$, sensitivitas dan FNR pada ambang $t$ didefinisikan sebagai

#align(center)[
  $ "Sens"_T(t) = P_T(s(X) >= t | Y = 1), quad
    "FNR"_T(t) = 1 - "Sens"_T(t). $
]

Toleransi *false-negative rate* (FNR) ditetapkan sebesar $alpha = 0.10$ dan probabilitas kegagalan sebesar $delta = 0.05$. Diberikan $n^+$ skor positif target untuk kalibrasi, $S_1^+, ..., S_(n^+)^+$, indeks terbesar yang memenuhi batas binomial didefinisikan sebagai

#align(center)[
  $ k^*(n^+) = max { k in {0, ..., n^+ - 1} :
    P("Bin"(n^+, alpha) <= k) <= delta }. $
]

Jika himpunan tersebut kosong, prosedur mengeluarkan keputusan *REFUSE* (menolak menerbitkan ambang). Jika tidak, dengan statistik urut menaik $S_(1)^+ <= ... <= S_(n^+)^+$, ambang yang diterbitkan adalah

#align(center)[
  $ hat(t) = S_(k^*(n^+) + 1)^+. $
]

Dengan asumsi observasi positif target untuk kalibrasi merupakan sampel i.i.d. yang representatif, model dan skor telah ditetapkan, label referensi akurat, serta distribusi tidak berubah antara kalibrasi dan penggunaan, pernyataan hingga sampelnya adalah

#align(center)[
  $ P_(D_"cal")["Sens"_T(hat(t)) >= 1 - alpha] >= 1 - delta. $
]

Untuk $k = 0$, kelayakan mensyaratkan $(1 - alpha)^(n^+) <= delta$. Karena itu, ukuran sampel positif minimum adalah

#align(center)[
  $ n_"min"^+ = ceil(ln(delta) / ln(1 - alpha)) = 29. $
]

Jaminan ini berlaku untuk sensitivitas populasi di bawah asumsi tersebut; jaminan ini bukan jaminan per pasien. Pernyataan ini merupakan jaminan PAC (*probably approximately correct*). Karena keterkaitan identitas pasien terverifikasi tidak tersedia, analisis data nyata mengaudit algoritme dan kinerjanya pada data tahan-uji, tetapi tidak memvalidasi interpretasi PAC pada populasi pasien. Prosedur statistik urut positif target ini selanjutnya disebut Tier A.

== Pembanding CRC dan IW-LTT

Untuk setiap kepala klasifikasi pada studi primer dan sekunder, runner mengkalibrasi marginal CRC dan IW-LTT pada kasus positif target di bagian kalibrasi, lalu menghitung sensitivitas dan spesifisitas ambang yang diterbitkan pada holdout target independen. Perbandingan menggunakan bobot kepadatan satu dan batas atas bobot satu, sehingga mewakili baseline kasus positif exchangeable. W-CRC pada bobot kalibrasi dan titik uji satu identik dengan CRC, sehingga tidak disajikan sebagai metode atau baris hasil terpisah. Baseline ini tidak mengoreksi pergeseran kovariat antarsitus. Analisis berbobot yang mengoreksi shift memerlukan rasio kepadatan target/sumber kondisional pada TB positif yang dapat dipertanggungjawabkan dan batas atas rasio yang valid; keduanya belum tersedia. Status sertifikat, ambang, diagnostik metode, serta metrik holdout dicatat pada `risk_control_comparison` di setiap `results.json`. Kisi 25 ambang IW-LTT ditentukan dari kuantil unik 0,01 through 0,99 skor validasi sumber dan ditetapkan independen dari sampel kalibrasi target. Nilai p WSR merata-ratakan 100 urutan acak dengan seed 42.

== Kebijakan dan sertifikat verifikasi selektif

Pada simulasi TBX11K bagian pelatihan, skor referensi tetap $c$ digunakan untuk menentukan proses verifikasi. Semua rekaman dengan $s_i >= c$ diverifikasi, sedangkan rekaman dengan skor di bawah $c$ diverifikasi secara independen dengan probabilitas $q$. Probabilitas verifikasi yang diketahui adalah

#align(center)[
  $ pi_i = P(V_i = 1 | s_i) =
    cases(1, & s_i >= c, q, & s_i < c). $
]

Label yang tidak diverifikasi disembunyikan dari prosedur kalibrasi. Nilai $c$ ditetapkan menggunakan ambang validasi sumber. Sebelum label simulasi diperiksa, kisi ambang kandidat ditentukan dari kuantil skor validasi sumber pada 25 titik probabilitas antara 0,01 dan 0,99. Kandidat diuji berurutan dari ambang yang lebih sensitif ke yang kurang sensitif. Satu nilai kegagalan $delta = 0.05$ berlaku untuk seluruh urutan tetap yang telah ditentukan. Karena FNR populasi tidak menurun ketika ambang naik dan aturan hanya menerbitkan awalan kandidat yang lolos, setiap penerbitan yang tidak aman mensyaratkan kandidat tidak aman pertama juga lolos batas empat komponen yang berlaku secara titik; argumen ini mengendalikan false issuance tanpa membagi $delta$ menurut jumlah kandidat. Nilai probabilitas verifikasi skor rendah yang diaudit adalah $q in {0.10, 0.25, 0.50, 1.00}$.

Sertifikat bias verifikasi empat batas disebut VB. Nilai $delta$ dibagi ke empat ekor dengan $eta = delta / 4$. Pada ambang kandidat $t$, batas atas Clopper–Pearson untuk jumlah kasus positif yang terlewat pada kelompok skor tinggi dan rendah masing-masing dinyatakan sebagai $U_H(t)$ dan $U_L(t)$. Komponen kelompok skor rendah diskalakan dengan $1/q$. Batas bawah CP untuk total kasus positif pada kedua kelompok masing-masing dinyatakan sebagai $L_H$ dan $L_L$, dengan komponen skor rendah juga diskalakan dengan $1/q$. Batas atas FNR yang digunakan adalah

#align(center)[
  $ U_"FNR"(t) =
    frac(U_H(t) + U_L(t) / q, L_H + L_L / q). $
]

Jika batas bawah penyebut bernilai nol, batas FNR dianggap tak hingga. Ambang lolos jika $U_"FNR"(t) <= alpha$. Aturan urutan tetap menerbitkan ambang terbesar pada awalan kandidat yang seluruhnya lolos; jika kandidat pertama gagal, prosedur menolak menerbitkan ambang. Untuk kasus terlewat dengan skor di bawah $c$, implementasi menggunakan segmen rendah $s_i < min(t, c)$; seluruh rekaman dengan skor sekurang-kurangnya $c$ harus teramati.

Metode linear menguji satu fungsi IPW untuk setiap ambang tetap. Dengan $Z_i(t) = V_i Y_i (1(s_i < t) - alpha) / pi_i$, batas atas yang dihitung adalah

#align(center)[
  $ overline(Z)(t) +
    sqrt((ln(1 / delta) sum_(i=1)^N pi_i^(-2)) / (2 N^2)). $
]

Ambang dinyatakan lolos jika batas atas tersebut tidak lebih dari nol. Implementasi berhenti pada kandidat pertama yang gagal. Saat probabilitas verifikasi skor rendah $q$ kecil, suku batas kepercayaan bertambah sebanding dengan $q^(-2)$ dan dapat melampaui margin empiris negatif pada ukuran skrining yang tersedia. Karena itu, penolakan yang berulang adalah hasil negatif konservatif, bukan bukti kegagalan validitas metode ini. Metode ini bukan klaim bahwa Hoeffding selalu lebih ketat atau bermanfaat secara empiris.

== Kontrol, analisis sekunder, dan ringkasan ketidakpastian

Pada kontrol spektrum negatif, N1 berarti kelompok negatif yang hanya mencakup citra sehat, sedangkan N2 mencakup citra sehat dan penyakit non-TB. Kontrol pembagian acak menggunakan pembagian acak terstratifikasi tingkat citra yang memungkinkan setiap situs muncul pada bagian pelatihan, kalibrasi, validasi, dan evaluasi. Keterkaitan pasien belum diverifikasi. Kontrol ini bukan ukuran kemampuan transfer antarsitus atau validasi tingkat pasien.

Analisis ukuran lesi menggunakan 250 kotak pada ROI CSV proyek untuk Shenzhen. Arsip publik memuat koleksi anotasi terpisah dari dua radiolog, masing-masing untuk 336 citra positif TB; dokumentasi tidak menjelaskan derivasi CSV proyek dari kedua koleksi tersebut. Kotak dikelompokkan berdasarkan citra pada 68 citra positif TB aktif di dalam manifest. Jika satu citra memiliki beberapa kotak, ukuran yang digunakan adalah kotak anotasi terbesar, bukan gabungan geometris seluruh kotak. Ukuran lesi dihitung sebagai luas kotak dibagi luas citra penuh. Citra dibagi menjadi tiga tertil dengan ukuran mendekati sama. Ukuran ini merupakan proksi radiografis, bukan ukuran keparahan klinis, dan anotasi tidak mencakup seluruh kasus positif. Sensitivitas tiap strata dilaporkan dengan interval kepercayaan binomial eksak CP 95%.

Untuk Tier A, kasus positif diambil berulang kali tanpa pengembalian dari kumpulan kalibrasi yang terbatas. Analisis utama dan kontrol pembagian acak menggunakan 1.000 pengulangan; simulasi verifikasi selektif menggunakan 100 pengulangan. Frekuensi pelanggaran empiris dihitung terhadap data tahan-uji target yang terbatas. Interval CP untuk frekuensi tersebut menggambarkan ketidakpastian Monte Carlo, bukan ketidakpastian akibat pengambilan sampel dari populasi pasien tak terbatas.

#figure(
  image("figures/study_workflow.svg"),
  caption: [Alur penelitian dan batas asumsi sertifikasi.],
) <fig:workflow>

== Protokol ablation LoRA

LoRA RAD-DINO dievaluasi sebagai ablation penyetelan halus pada data sumber. LoRA diterapkan pada proyeksi atensi query dan value dengan rank 8, alpha 16, dropout 0,05, *autocast* CUDA FP16, dan kepala klasifikasi yang dilatih pada data sumber. Pelatihan hanya menggunakan citra pelatihan Montgomery dan Shenzhen. Bagian validasi sumber digunakan untuk menetapkan ambang baseline; kasus positif pada bagian pelatihan resmi TBX11K disisihkan untuk kalibrasi Tier A, sedangkan bagian validasi TBX11K digunakan untuk evaluasi.

Konfigurasi menggunakan tiga epoch, AdamW dengan laju pembelajaran $10^(-4)$ dan *weight decay* 0,01, ukuran batch 1, akumulasi gradien 4, serta dua seed independen, 42 dan 43. Tiap run memakai 639 citra pelatihan sumber dan 161 citra validasi sumber. Sebanyak 295.681 parameter dapat dilatih pada adapter dan kepala klasifikasi. Adapter, kepala klasifikasi, manifest pembagian sumber, dan hasil JSON disimpan terpisah.

= Hasil

== Kohort dan pipeline

Kohort 9.060 citra lolos pemeriksaan integritas cache. Seluruh citra dalam cache RAD-DINO 518 piksel dapat didekode dan sesuai dengan nilai SHA-256 yang tercatat. Audit terhadap 882.775 rekaman daftar pelatihan RAD-DINO tidak menemukan kecocokan ID citra, stem nama berkas, atau nama berkas dengan manifest lokal. Audit metadata ini tidak dapat menyingkirkan kemungkinan citra duplikat yang diganti namanya.

== Kinerja ambang sumber dan hasil Tier A

Tabel berikut merangkum hasil ambang sumber B0. Kinerja pada TBX11K dan Shenzhen merupakan estimasi eksploratif tingkat-gambar; keterkaitan pasien belum diverifikasi.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([*Protokol dan target*], [*Kepala*], [*Sensitivitas*], [*Spesifisitas*]),
    [TBX11K, RAD-DINO], [Regresi logistik], [91,5%], [64,3%],
    [TBX11K, RAD-DINO], [MLP], [95,7%], [57,5%],
    [TBX11K, ResNet50-A1], [Regresi logistik], [84,1%], [33,3%],
    [TBX11K, ResNet50-A1], [MLP], [88,4%], [31,0%],
    [Shenzhen, RAD-DINO], [Regresi logistik], [79,2%], [84,0%],
    [Shenzhen, RAD-DINO], [MLP], [77,4%], [89,6%],
    [Pembagian acak campuran situs, RAD-DINO], [Regresi logistik], [81,5%], [99,3%],
    [Pembagian acak campuran situs, RAD-DINO], [MLP], [85,8%], [99,2%],
    [TBX11K, LoRA RAD-DINO], [Kepala linear], [95,7%], [88,8%],
  ),
  caption: [Kinerja ambang sumber pada protokol utama, sekunder, kontrol, dan ablation. TBX11K memiliki 164 kasus positif dan 1.600 negatif pada validasi; Shenzhen memiliki 168 kasus positif dan 163 negatif pada bagian evaluasi. Pembagian acak campuran situs bukan evaluasi transportabilitas.],
) <tab:b0>

== Perbandingan CRC-family

Pada holdout primer TBX11K RAD-DINO, CRC menghasilkan sensitivitas/spesifisitas 90,9%/70,2% untuk regresi logistik dan 91,5%/81,5% untuk MLP. W-CRC berbobot kalibrasi dan titik uji satu identik dengan CRC, sehingga tidak dilaporkan sebagai hasil terpisah. IW-LTT berbobot satu memilih ambang yang lebih rendah: sensitivitas/spesifisitas 94,5%/51,8% untuk regresi logistik dan 93,3%/68,0% untuk MLP. Nilai p IW-LTT pada ambang terpilih masing-masing 0,00000577 dan 0,00918, dengan $delta = 0.05$. Angka ini adalah hasil empiris pada holdout di bawah bobot satu, bukan bukti koreksi pergeseran situs. Pada holdout Shenzhen, CRC menghasilkan sensitivitas/spesifisitas 86,3%/68,1% untuk regresi logistik dan 85,7%/76,1% untuk MLP. IW-LTT menghasilkan 97,0%/31,3% dan 94,0%/51,5%, dengan nilai p 0,00784 dan 0,000113. Kalibrasi memakai kasus positif target sehingga belum menggantikan analisis transport berbobot dari situs sumber ke target.

Pada analisis Tier A utama, kohort kalibrasi TBX11K berisi 496 kasus positif; data tahan-uji terdiri atas 164 kasus positif dan 1.600 kasus negatif. Untuk $n^+ = 29$, sensitivitas rata-rata regresi logistik dan MLP masing-masing 96,0% dan 97,0%, sedangkan spesifisitasnya 41,8% dan 39,3%. Untuk $n^+ = 200$, sensitivitas dan spesifisitas masing-masing 93,2%/58,8% pada regresi logistik serta 94,5%/65,7% pada MLP.

#figure(
  image("figures/latest_results.svg"),
  caption: [Ringkasan hasil agregat berdasarkan berkas JSON eksperimen.],
) <fig:results>

Peluang empiris spesifisitas tersertifikasi mencapai 70% pada ukuran kalibrasi $n^+ = 29, 46, 61, 100, 200$ berturut-turut adalah 6,1%, 4,0%, 5,1%, 2,7%, dan 2,3% untuk regresi logistik. Untuk MLP nilainya 11,7%, 15,7%, 19,4%, 14,9%, dan 33,2%.

Pada analisis Shenzhen, ambang sumber tidak mencapai sensitivitas 90%: sensitivitas/spesifisitas regresi logistik adalah 79,2%/84,0%, sedangkan MLP 77,4%/89,6%. Karena hanya tersedia 168 kasus positif untuk kalibrasi, metode menolak menerbitkan ambang pada $n^+ = 200$. Pada $n^+ = 100$, sensitivitas dan spesifisitas rata-rata MLP adalah 92,1% dan 62,9%, dengan frekuensi pelanggaran empiris 4,9%; regresi logistik menghasilkan 93,8% dan 39,9%, dengan frekuensi pelanggaran 28,8%. Pada ukuran ini, peluang spesifisitas mencapai 70% adalah 4,9% untuk MLP dan 0,6% untuk regresi logistik. Dua frekuensi 4,9% tersebut merupakan ringkasan resampling yang berbeda: satu menyatakan pelanggaran sensitivitas dan yang lain spesifisitas sekurang-kurangnya 70%.

Pada Shenzhen, peluang spesifisitas mencapai 70% untuk MLP pada $n^+ = 29, 46, 61, 100$ masing-masing adalah 10,4%, 10,3%, 10,3%, dan 4,9%; untuk regresi logistik nilainya 4,0%, 1,6%, 0,9%, dan 0,6%. Seluruh angka tersebut merupakan estimasi dari 1.000 resampling tingkat gambar; keterkaitan pasien belum diverifikasi. Nilai rinci, interval frekuensi pelanggaran, dan frekuensi penolakan tercatat dalam berkas hasil eksperimen.

Probe asal situs mencapai akurasi 99,3%, akurasi seimbang 98,6%, dan luas area di bawah kurva ROC satu-lawan-semua makro 99,9% pada embedding RAD-DINO. Hasil ini menunjukkan bahwa embedding mengandung informasi situs, tetapi tidak mengidentifikasi sumber atau mekanisme kausalnya. Empat probe ablasi praproses dengan alias pixel identik dikeluarkan mempertahankan akurasi seimbang tinggi: ResNet50-A1 window-all-224 98,4%, letterbox-224 98,7%, RAD-DINO window-all-518 98,3%, dan letterbox-518 99,2%. Sinyal asal situs tetap mudah didekode; hasil ini tidak membuktikan shortcut TB kausal.

Spesifisitas pada ambang sumber regresi logistik adalah 85,6% terhadap N1 (negatif sehat saja) dan 64,3% terhadap N2 (negatif sehat serta penyakit non-TB). Untuk MLP, spesifisitasnya 76,8% pada N1 dan 57,5% pada N2. Sensitivitas tidak berubah dalam masing-masing kepala karena kumpulan positif tahan-uji yang digunakan sama. Hasil ini menunjukkan bahwa spesifisitas bergantung pada komposisi kelompok negatif.

Kontrol pembagian acak campuran situs menghasilkan spesifisitas ambang sumber sekitar 99%, jauh di atas nilai pada evaluasi TBX11K yang menahan situs sebagai target (57,5–64,3%). Perbedaan ini konsisten dengan evaluasi campuran situs yang lebih mudah, tetapi komposisi pembagian berbeda dan keterkaitan identitas antarsitus tidak diketahui. Karena itu, selisih tersebut bukan estimasi kausal pergeseran situs.

== Audit robustness multi-seed dan resampling nested

Sepuluh seed sumber (42-51) menghasilkan sensitivitas B0 rata-rata RAD-DINO 92,6%/94,6% pada TBX11K dan 82,3%/85,2% pada Shenzhen untuk logistik/MLP. Untuk ResNet50-A1, rata-ratanya 87,6%/87,1% pada TBX11K dan 77,4%/71,8% pada Shenzhen. Audit nested 1.000 pengulangan membagi pool positif target secara acak menjadi kalibrasi dan evaluasi yang disjoint (660 positif TBX11K, 336 Shenzhen sebelum pembagian). Violation Tier A pada n+=29 berkisar 4,7-6,7% dan pada n+=100 2,9-6,9% lintas encoder, target, dan head; nilai binomial teoretis masing-masing 4,71% dan 2,37%. Pool ini image-level, pasien belum terverifikasi, sehingga frekuensi empiris tersebut tidak membuktikan validitas PAC pasien-i.i.d. CRC memiliki violation per-pengulangan 21,8-48,6%; angka ini bukan type-I error karena CRC mengontrol risiko marginal. IW-LTT hanya 20 pengulangan dan 10 ordering per sel dan tetap preliminary.

Simulasi null WSR menghasilkan 50/2.000 false rejection (2,5%; CP95 1,86-3,28%). Cross-fitted ESS diagnostik adalah 415/660 untuk TBX11K dan 43/336 untuk Shenzhen pada ResNet50-A1, serta 447/660 dan 69/336 pada RAD-DINO. Rasio tersebut bukan density ratio terverifikasi atau bobot valid.

Audit label ekstrem memetakan 140 temuan obsolete/unresolved TBX11K sebagai semua aktif atau semua non-aktif. Pada n+=200, violation RAD-DINO logistik berubah dari 0,9% menjadi 8,5%; hasil ini menunjukkan sensitivitas terhadap kebijakan pemetaan, bukan kebenaran biologis salah satu label. Hasil lengkap tersimpan pada JSON robustness dan label-sensitivity per encoder.

== Simulasi verifikasi selektif

Simulasi menggunakan label lengkap rekaman pelatihan TBX11K hanya untuk membentuk kebijakan verifikasi selektif, dengan 100 pengulangan untuk setiap kebijakan. Pada $q = 0.10$, Tier A naif menerbitkan ambang pada seluruh pengulangan dan melanggar sasaran sensitivitas pada 100% pengulangan untuk kedua kepala. Pada $q = 0.25$, frekuensi pelanggaran naif adalah 100% untuk regresi logistik dan 77% untuk MLP.

Sertifikat VB empat batas tidak menunjukkan pelanggaran pada ambang yang diterbitkan. Pada $q = 0.10$, tingkat penerbitan ambang adalah 4% untuk regresi logistik dan 7% untuk MLP; spesifisitas rata-rata pada sejumlah kecil pengulangan yang diterbitkan tersebut adalah 23,1% dan 10,5%. Pada $q = 0.25$, VB menerbitkan ambang pada seluruh pengulangan untuk kedua kepala tanpa pelanggaran teramati. Metode IPW–Hoeffding menolak semua ambang pada seluruh simulasi yang diaudit. Tidak adanya pelanggaran teramati tidak membuktikan batas kegagalan populasi nominal 5%, terutama karena jumlah pengulangannya terbatas.

Pada $q = 1$, gerbang utilitas pengetatan yang telah ditetapkan mensyaratkan spesifisitas rata-rata VB tidak lebih rendah lebih dari 3 poin persentase dibandingkan Tier A naif. Gerbang ini tidak terpenuhi: spesifisitas VB dan Tier A naif masing-masing 51,8% dan 63,6% untuk regresi logistik, serta 51,1% dan 73,7% untuk MLP. Dengan demikian, VB bersifat konservatif pada audit ini, tetapi tidak lolos gerbang utilitas yang ditetapkan.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([*Probabilitas $q$*], [*Metode*], [*Tingkat penerbitan*], [*Pelanggaran teramati*]),
    [0,10], [Tier A naif (logistik / MLP)], [100% / 100%], [100% / 100%],
    [0,10], [Sertifikat VB (logistik / MLP)], [4% / 7%], [0% / 0%],
    [0,25], [Tier A naif (logistik / MLP)], [100% / 100%], [100% / 77%],
    [0,25], [Sertifikat VB (logistik / MLP)], [100% / 100%], [0% / 0%],
  ),
  caption: [Hasil simulasi verifikasi selektif pada rekaman pelatihan TBX11K yang berlabel lengkap. Label digunakan hanya untuk menyimulasikan kebijakan verifikasi.],
) <tab:selective>

Sweep bootstrap budget memvariasikan N=1.000-6.496, prevalensi 0,05-0,40, dan q=0,10-1,00 dengan 100 pengulangan per sel. Tier A naif terbit pada seluruh sel dengan 2-100% violation; VB menerbitkan 44/64 sel dengan 0-2% violation di antara sertifikat yang terbit; IPW-Hoeffding menerbitkan 9/64 sel tanpa violation teramati. Simulasi menggunakan skor kelas-kondisional empiris TBX11K yang dideduplikasi, bukan kohort prospektif atau bukti PAC populasi.

== Analisis ukuran lesi

Tabel berikut menyajikan sensitivitas pada tiga tertil luas ROI beserta interval kepercayaan binomial eksak CP 95% dalam tanda kurung.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([*Kepala*], [*Tertil kecil*], [*Tertil sedang*], [*Tertil besar*]),
    [Regresi logistik], [20/23 = 87,0% (66,4–97,2)], [20/23 = 87,0% (66,4–97,2)], [19/22 = 86,4% (65,1–97,1)],
    [MLP], [22/23 = 95,7% (78,1–99,9)], [21/23 = 91,3% (72,0–98,9)], [20/22 = 90,9% (70,8–98,9)],
  ),
  caption: [Sensitivitas pada kasus positif dengan anotasi ROI Shenzhen.],
) <tab:roi>

Interval kepercayaan lebar dan saling tumpang tindih. Sensitivitas regresi logistik relatif datar menurut luas anotasi, sedangkan sensitivitas MLP menurun dari tertil kecil ke besar. Subset positif yang kecil ini tidak dapat memastikan ketiadaan pengaruh ukuran lesi dan tidak mewakili citra positif yang tidak memiliki anotasi.

== Ablation LoRA

Pelatihan pada perangkat RTX 3050 dengan memori 6 GB selesai tanpa kehabisan memori. *Binary cross-entropy* pelatihan turun dari 0,6013 menjadi 0,2472 dan 0,1222 selama tiga epoch. Ambang empiris sumber adalah 0,9352, dengan sensitivitas dan spesifisitas validasi sumber 91,3% dan 98,8%. Pada validasi TBX11K, ambang tersebut menghasilkan sensitivitas 95,7% (157/164) dan spesifisitas 88,8% (1.420/1.600).

Dengan kalibrasi positif target Tier A, rata-rata sensitivitas/spesifisitas adalah 96,3%/73,9% pada $n^+ = 29$, 95,4%/83,1% pada 46, 95,0%/87,5% pada 61, 94,9%/89,7% pada 100, dan 93,8%/94,0% pada 200. Tabel berikut memuat frekuensi pelanggaran empiris dan interval CP 95% untuk frekuensi Monte Carlo.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([*$n^+$*], [*Sensitivitas rata-rata*], [*Spesifisitas rata-rata*], [*Pelanggaran (jumlah/1.000; interval CP 95%)*]),
    [29], [96,3%], [73,9%], [4,6% (46/1.000; 3,4–6,1%)],
    [46], [95,4%], [83,1%], [5,6% (56/1.000; 4,3–7,2%)],
    [61], [95,0%], [87,5%], [3,8% (38/1.000; 2,7–5,2%)],
    [100], [94,9%], [89,7%], [1,5% (15/1.000; 0,8–2,5%)],
    [200], [93,8%], [94,0%], [1,2% (12/1.000; 0,6–2,1%)],
  ),
  caption: [Hasil Tier A untuk ablation LoRA pada validasi TBX11K.],
) <tab:lora>

Pada $n^+ = 46$, estimasi frekuensi pelanggaran pada data tahan-uji terbatas melebihi 5%. Frekuensi audit tersebut tidak membuktikan maupun menyangkal pernyataan PAC populasi. Memori CUDA maksimum yang dialokasikan adalah 502.656.512 byte. Seed 43 menghasilkan B0 sensitivity/specificity 95,1%/89,9% dan Tier A n+=200 91,1%/94,5%, dengan violation finite-pool 17,7% (177/1.000; CP95 15,4-20,2%), dibanding 1,2% pada seed 42. Perbedaan lintas seed ini menggambarkan variasi pada audit citra terbatas; patient linkage belum diverifikasi, sehingga hasil tidak memvalidasi atau menyangkal PAC populasi. Ablation ini tidak membuktikan peningkatan yang signifikan secara statistik ataupun transfer pada populasi pasien.

= Pembahasan

Aturan statistik urut eksak memiliki batas penolakan yang jelas: pada parameter $(alpha, delta) = (0.10, 0.05)$, kurang dari 29 unit positif target yang independen dan representatif tidak cukup untuk menerbitkan sertifikat. Ketika ukuran kalibrasi bertambah, sensitivitas rata-rata turun mendekati target dan spesifisitas meningkat; pada ukuran minimum, sensitivitas yang lebih tinggi disertai spesifisitas yang lebih rendah. Pola ini merupakan deskripsi pada data tahan-uji TBX11K yang terbatas. Pengambilan ulang subset memakai kembali kumpulan citra yang sama dan tidak membuktikan validitas PAC pada populasi pasien.

Di Shenzhen, ambang sumber tidak mencapai sasaran sensitivitas 90% untuk regresi logistik maupun MLP, walaupun spesifisitasnya masing-masing 84,0% dan 89,6%. Pada $n^+ = 100$, Tier A meningkatkan sensitivitas rata-rata MLP menjadi 92,1%, dengan spesifisitas 62,9% dan frekuensi pelanggaran empiris 4,9%; frekuensi pelanggaran regresi logistik adalah 28,8%. Spesifisitas sekitar 40–63% berarti sekitar 37–60 dari setiap 100 citra non-TB pada data tahan-uji akan ditandai untuk tindak lanjut. Hal ini dapat menambah kebutuhan pemeriksaan konfirmasi, tetapi data tahan-uji tingkat gambar ini tidak mengestimasi beban rujukan atau nilai prediktif pada populasi penerapan.

Pada perbandingan B0 utama di TBX11K, kedua kepala RAD-DINO memiliki sensitivitas dan spesifisitas teramati lebih tinggi daripada kepala ResNet50-A1 yang bersesuaian: 91,5%/64,3% dan 95,7%/57,5% dibandingkan dengan 84,1%/33,3% dan 88,4%/31,0%. Perbandingan ini bersifat deskriptif dan tidak didasarkan pada pengujian keunggulan statistik. Kontrol spektrum negatif menunjukkan kenaikan spesifisitas MLP dari 57,5% menjadi 76,8% ketika hanya citra sehat digunakan sebagai negatif; untuk regresi logistik, kenaikannya dari 64,3% menjadi 85,6%. Kumpulan positif yang digunakan sama. Karena itu, pelaporan spesifisitas perlu menyebutkan komposisi kelompok negatif.

Spesifisitas sekitar 99% pada kontrol pembagian acak campuran situs jauh lebih tinggi daripada hasil sekitar 58–64% pada evaluasi TBX11K yang menahan situs sebagai target. Perbedaan ini sejalan dengan pembagian campuran situs yang lebih mudah, tetapi perbedaan komposisi data dan ketidakpastian keterkaitan identitas mencegah interpretasi sebagai estimasi kausal pergeseran situs. Probe asal situs dan kontrol spektrum negatif menunjukkan bahwa satu metrik agregat perlu ditafsirkan dengan hati-hati.

Kalibrasi naif pada kasus positif yang diverifikasi secara selektif menghasilkan kinerja buruk dalam simulasi. Sertifikat empat batas mengurangi utilitas demi konservatisme dan tidak lolos gerbang spesifisitas yang telah ditetapkan; metode Hoeffding menolak seluruh ambang. Temuan ini mendukung pelaporan penolakan secara eksplisit serta perancangan verifikasi yang lebih baik, tetapi tidak menunjukkan bahwa batas VB yang dievaluasi sudah memadai untuk penerapan. Simulasi menggunakan label lengkap untuk membentuk proses seleksi dan tidak menggantikan studi verifikasi prospektif.

Analisis ROI bersifat eksploratif. Luas kotak hanya proksi ukuran radiografis, anotasi mencakup sebagian citra positif, dan tiap strata berisi 22–23 kasus. Karena itu, hasil ini tidak mendukung klaim tentang keparahan. Ablation LoRA juga belum dapat digunakan untuk klaim pemilihan model berdasarkan hasil validasi target atau klaim keunggulan klinis.

== Keterbatasan

Pertama, konfirmasi satu citra; keterkaitan pasien belum diverifikasi pada ketiga dataset mendukung pembagian data tingkat gambar, tetapi tidak membuktikan pengambilan sampel i.i.d. atau menyingkirkan citra duplikat lintas dataset sumber. Daftar pelatihan publik RAD-DINO tidak menyediakan berkas citra atau hash piksel, sehingga audit tidak menguji duplikasi piksel dengan nama berkas yang berbeda. Kedua, standar label referensi dan spektrum kohort hanya merupakan proksi terhadap konfirmasi klinis dan populasi penerapan.

Ketiga, verifikasi selektif disimulasikan dan tidak diamati secara prospektif. Keempat, dokumentasi publik belum menjelaskan apakah tabel ROI konsensus Shenzhen menggabungkan kedua set anotasi asal. Kelima, manifest tidak menyediakan strata usia, jenis kelamin, atau riwayat TB sehingga stratifikasi Mondrian tidak dapat dievaluasi. Terakhir, dokumentasi proyek yang tersedia belum mencatat status persetujuan atau pengecualian etik, ketentuan dataset, dan izin penggunaan data sebagai selesai; kewenangan untuk mendistribusikan artefak turunan juga belum ditetapkan. Artefak tersebut tidak didistribusikan bersama naskah ini.

= Kesimpulan

Penelitian ini mengimplementasikan ambang sensitivitas eksak berbasis kasus positif target dengan keputusan sertifikasi atau penolakan, mengaudit pertukarannya pada data tingkat gambar, dan membandingkannya dengan prosedur verifikasi selektif. Untuk toleransi FNR 10% dan probabilitas kegagalan 5%, batas minimum 29 kasus positif mengikuti aturan eksak. Namun, analisis empiris belum membuktikan seluruh asumsi distribusi dan pelabelan yang diperlukan. Hasil mendukung evaluasi prospektif dengan protokol verifikasi yang ditetapkan sebelumnya sebelum menafsirkan prosedur sebagai bukti kinerja klinis.

#set heading(numbering: none)

= Pernyataan kontribusi kepengarangan CRediT

#juti.credits(authors)

= Pernyataan konflik kepentingan

Penulis menyatakan tidak memiliki kepentingan finansial yang bersaing atau hubungan pribadi yang diketahui yang dapat memengaruhi karya ini.

= Ucapan terima kasih

Penelitian ini tidak menerima pendanaan eksternal.

= Ketersediaan data

Analisis menggunakan dataset publik yang tunduk pada ketentuan masing-masing. Citra mentah, embedding turunan, vektor skor, dan informasi yang dapat mengidentifikasi pasien tidak disertakan dalam repositori ini. Manifest pembagian data, konfigurasi, dan kode disimpan secara lokal untuk mendukung reproduksibilitas. Distribusi artefak turunan tidak dilakukan.

= Penggunaan kecerdasan buatan generatif

Kecerdasan buatan generatif digunakan untuk membantu memperjelas draf dan mengonversi naskah ke Typst. Penulis meninjau naskah akhir dan bertanggung jawab atas isinya.
