# Metodologi Review & Cleanup Codebase — Catatan Kerja yang Bisa Dipakai Ulang

Dokumen ini mencatat **cara kerja** (bukan sekadar hasil) dari sesi review &
cleanup kode `08_AI TRADING` (~11.700 baris Python di `src/`, `scripts/`,
`tests/`, plus `streamlit_app.py` 5.370 baris). Tujuannya supaya kalau Anda
membangun proyek AI baru di masa depan dan butuh proses serupa (review kode,
audit kualitas, cari bug, bersih-bersih codebase besar), Anda tinggal minta
"pakai metodologi kayak di 08_AI TRADING" dan asisten AI (versi mana pun)
punya panduan konkret untuk diikuti — bukan menebak-nebak dari nol.

---

## 0. Konteks singkat

Permintaan awal: *"review & bersihkan kode"* untuk proyek yang sudah berjalan
~1 bulan, punya riwayat git, dan sedang dipakai (bukan proyek baru kosong).
Tantangannya: codebase besar (belasan ribu baris), saya (asisten AI) belum
pernah melihatnya sebelumnya, dan ini adalah sistem yang hasilnya dipakai
untuk keputusan trading sungguhan — jadi kesalahan dalam "membersihkan" bisa
berakibat nyata, bukan cuma kosmetik.

---

## 1. Urutan langkah yang dipakai

### Langkah 1 — Orientasi dulu, jangan langsung ubah apa pun

Sebelum menyentuh satu baris kode pun:
- Lihat isi folder (`ls`), baca dokumentasi yang ada (README/QUICK_START).
- Cek `git status` dan `git log` — apakah ada perubahan yang belum di-commit?
  Apakah ini proyek aktif (commit rutin) atau sudah lama tidak disentuh?
- Ukur skala: jumlah baris per file/folder (`wc -l`), supaya tahu apakah ini
  tugas "baca 5 file" atau "butuh strategi pembagian kerja".

**Kenapa penting:** kalau langsung lompat ke "perbaiki ini itu" tanpa orientasi,
risiko besar salah paham konteks — misalnya mengira sesuatu adalah bug padahal
itu keputusan desain yang disengaja, atau menghapus perubahan yang belum
di-commit milik proses lain (di proyek ini ternyata ada workflow harian
otomatis yang menghasilkan ratusan file data belum ter-commit — itu HARUS
dikenali dan TIDAK disentuh).

### Langkah 2 — Klarifikasi fokus SEBELUM menyelam

Codebase besar + permintaan umum ("review & cleanup") = ambigu. Daripada
menebak, tanya langsung ke user dengan pilihan konkret (bug fix vs fitur baru
vs review kualitas vs lainnya) — sekali tanya di awal jauh lebih murah
daripada mengerjakan hal yang salah selama berjam-jam.

### Langkah 3 — Untuk codebase besar: pecah jadi domain, review PARALEL

Alih-alih membaca 11.700 baris secara berurutan (lambat, dan konteks jenuh
sebelum sampai ujung), proyek dipecah jadi 4 domain logis yang saling lepas:
1. Model ML & logika trading (`src/models/`, `src/trading/`)
2. Data pipeline & infrastruktur (`src/data_pipeline/`, `src/utils/`,
   `src/database/`, `src/nlp/`)
3. UI (satu file besar `streamlit_app.py`, ditangani sendiri karena ukurannya)
4. Entry point, script CLI, dan test (`run_analysis.py`, `scripts/`, `tests/`)

Tiap domain di-review oleh **subagent terpisah berjalan paralel** (lihat
bagian 2 di bawah untuk pola instruksinya). Empat subagent selesai dalam waktu
yang kira-kira sama dengan SATU domain dikerjakan sendiri-sendiri secara
berurutan — karena jalan bersamaan, bukan bergiliran.

**Kapan pola ini masuk akal:** proyek besar (~>2000 baris, banyak file),
domain yang jelas terpisah (tidak semua saling bergantung erat), dan Anda
butuh cakupan MENYELURUH bukan cuma jawaban cepat untuk 1 pertanyaan spesifik.
Untuk tugas kecil/spesifik, ini oversized — cukup baca file yang relevan
langsung.

### Langkah 4 — Instruksi ke tiap subagent harus SPESIFIK, bukan "cek kualitas kode"

Instruksi yang dipakai (pola umum, isi file-nya beda per domain):
- **Daftar file eksplisit** yang jadi tanggung jawab agent itu.
- **Kategori temuan yang diminta**, sama untuk semua agent supaya hasilnya
  bisa disatukan: (1) duplikasi logika, (2) fungsi terlalu kompleks, (3) kode
  mati (dead code), (4) pola tidak konsisten antar file serupa, (5) red flag
  kebenaran (correctness) yang kelihatan sambil lalu — TANPA diminta berburu
  bug secara aktif (beda fokus dari audit keamanan/bug-hunting).
- **Format laporan wajib**: file + nomor baris + satu kalimat masalah + satu
  kalimat saran. Ini krusial — tanpa format ini, laporan jadi esai panjang
  yang susah ditindaklanjuti.
- **Batas jumlah temuan** ("~15-25 temuan paling berdampak", bukan daftar
  nitpick tanpa akhir) — supaya agent memprioritaskan, bukan mencatat semua
  hal sekecil apa pun.

### Langkah 5 — Sintesis semua laporan jadi kategori, lalu TANYA prioritas

Setelah semua subagent selesai, hasil disatukan jadi kategori lintas-domain:
- **Bug nyata** (mengubah hasil/output — di sini: 6 bug ditemukan, termasuk 1
  yang mencemari fitur model dan 1 yang bisa memecah riwayat prediksi ticker).
- **Dead code besar** (764+ baris kode yang terbukti tidak pernah jalan).
- **Duplikasi mekanis** (pola yang sama ditulis ulang di banyak file — aman
  dibereskan tanpa mengubah perilaku).
- **Inkonsistensi arsitektur** (butuh keputusan desain, bukan sekadar "rapikan
  saja" — misal dua sistem pelacakan data paralel yang tidak saling kenal).
- **Gap testing**.

Baru SETELAH kategori ini jelas, saya tanya ke user urutan prioritas
(pakai pilihan konkret, bukan pertanyaan terbuka) — karena "bug dulu vs
cleanup dulu vs sekaligus vs laporan saja dulu" adalah keputusan RISIKO yang
harus diambil pemilik proyek, bukan diasumsikan sepihak oleh AI.

### Langkah 6 — Sebelum mengubah kode: verifikasi ulang temuan secara independen

Laporan dari subagent (atau dari mana pun) **tidak langsung dipercaya dan
dieksekusi**. Untuk tiap temuan yang mau ditindaklanjuti:
- Baca ulang kode di sekitarnya sendiri — apakah kesimpulan agent benar-benar
  tepat, atau ada konteks yang terlewat?
- Untuk klaim "dead code"/"tidak pernah dipakai": `grep` ULANG setiap simbol
  yang mau dihapus di SELURUH file, cek betul-betul tidak ada pemakaian lain
  yang terlewat (bukan cuma percaya baris yang disebut laporan).
- Untuk bug yang perbaikannya mengubah hasil: cek dulu bagaimana kode itu
  dipakai downstream, supaya perbaikan tidak menimbulkan efek samping baru.

Contoh nyata dari sesi ini: satu temuan bilang `PriceProjector` "sebagian"
dipakai di luar kode mati; setelah di-grep ulang manual, ternyata SEMUA
pemakaiannya ada di dalam blok kode mati — laporan awal agak meleset, dan
verifikasi ulang inilah yang mencegah kesalahan (dalam kasus ini untungnya
tidak fatal, tapi prinsipnya: jangan hapus berdasarkan laporan tanpa
verifikasi sendiri).

### Langkah 7 — Untuk penghapusan besar (ratusan baris): pakai tool yang tepat

Menghapus ratusan baris kode dengan tool edit berbasis "cari-ganti teks
persis" itu berisiko (harus mengetik ulang ratusan baris sebagai teks
pencarian — rawan salah ketik/salah transkripsi). Untuk kasus "hapus baris N
sampai M", command line (`sed`, `head`) adalah tool yang tepat karena bekerja
berdasarkan NOMOR BARIS, bukan mencocokkan isi teks:
```bash
head -n 4754 file.py > file_new.py && mv file_new.py file.py   # potong sampai baris tertentu
sed -i '2660,2803d' file.py                                     # hapus rentang baris tertentu
```
Tetap **verifikasi batas rentang dulu** (baca beberapa baris sebelum & sesudah
titik potong) sebelum menjalankan perintah — supaya tidak memotong di tengah
blok kode yang masih dipakai.

### Langkah 8 — Setelah tiap perubahan: verifikasi, bukan asumsi "pasti benar"

Urutan verifikasi yang dipakai tiap kali selesai mengubah kode:
1. **Cek sintaks** (`python -m py_compile file.py`) — cepat, menangkap
   kesalahan struktural.
2. **Jalankan test suite** (`pytest`) — menangkap regresi pada bagian yang
   sudah ada testnya.
3. **Smoke test runtime sungguhan** — untuk aplikasi Streamlit di sesi ini,
   itu berarti benar-benar menjalankan `streamlit run` dan mengambil
   (`curl`) halamannya, bukan cuma percaya "sintaksnya valid jadi pasti
   jalan". Banyak bug hanya muncul saat kode benar-benar dieksekusi.
4. **`grep` ulang** simbol-simbol yang dihapus, pastikan nol referensi
   menggantung tersisa di seluruh file.

Prinsip: **"lolos syntax check" ≠ "aman"**. Ketiga lapis verifikasi di atas
menangkap jenis kesalahan yang berbeda-beda.

### Langkah 9 — Manfaatkan git sebagai jaring pengaman, jangan bikin backup manual berlebihan

Kalau file sudah ter-`git`-track, riwayat commit sebelumnya SUDAH JADI backup
— tidak perlu bikin salinan manual (`file.py.bak`) di sebelahnya (itu cuma
menambah sampah). Cukup pastikan file benar sudah tercatat di commit
sebelumnya (`git log -1 -- file.py`) sebelum mengandalkan ini.

### Langkah 10 — Lacak progres secara eksplisit, terutama untuk kerja lintas-agent/lintas-waktu

Dengan 4 subagent berjalan paralel dan proses yang berlangsung lama, daftar
tugas (todo list) di-update di SETIAP titik transisi (agent mana yang sudah
selesai, langkah apa yang sedang dikerjakan) — supaya kalau sesi terputus di
tengah jalan, status pekerjaan tetap jelas (baik untuk saya sendiri melanjutkan,
maupun untuk user yang mengecek progres).

### Langkah 11 — Checkpoint ke user secara berkala, bukan cuma di akhir

Setiap subagent selesai, user diberi ringkasan singkat (bukan dump laporan
mentah). Setelah bug selesai diperbaiki + diverifikasi, ada checkpoint
laporan sebelum lanjut ke cleanup mekanis yang lebih luas. Ini memberi
kesempatan user meng-*course-correct* di tengah jalan, bukan baru tahu di
akhir kalau arahnya meleset dari yang diinginkan.

---

## 2. Template instruksi subagent yang dipakai (bisa disalin ulang)

```
Anda mereview bagian dari codebase [NAMA PROYEK] di [PATH]. Konteks: [1-2
kalimat tentang proyek & levelnya kematangannya — proyek baru vs proyek
aktif/production]. User minta review kualitas kode: duplikasi, simplifikasi,
efisiensi, konsistensi — BUKAN audit keamanan/bug-hunting mendalam (tetap
catat kalau kebetulan lihat sesuatu yang jelas salah).

Scope: [daftar file/folder eksplisit]. Jangan ubah apa pun, ini riset/review
saja.

Untuk tiap file, baca penuh. Lalu laporkan:
1. Duplikasi logika antar file
2. Fungsi yang terlalu kompleks/panjang
3. Kode mati (dead code)
4. Pola tidak konsisten antar file serupa
5. Red flag kebenaran yang terlihat sambil lalu

Laporkan sebagai daftar terstruktur per kategori, tiap butir: path file +
nomor baris + satu kalimat masalah + satu kalimat saran. Faktual dan
spesifik — hindari komentar vague tanpa contoh konkret. Prioritaskan ~15-25
temuan paling berdampak, bukan daftar nitpick tanpa akhir.
```

Untuk file tunggal yang sangat besar (seperti `streamlit_app.py` di sini),
tambahkan instruksi eksplisit: *"baca seluruh file dalam beberapa bagian
(pakai offset/limit), jangan cuma baca awal file"* — supaya agent tidak
berhenti membaca sebelum sampai ke bagian akhir file yang panjang.

---

## 3. Prinsip umum (berlaku lintas proyek, tidak spesifik ke trading)

1. **Pisahkan "mengubah perilaku" dari "tidak mengubah perilaku".** Bug fix
   mengubah hasil (butuh penilaian kasus per kasus + izin eksplisit kalau
   sistemnya "hidup"); cleanup mekanis (hapus duplikasi, hapus dead code)
   idealnya TIDAK mengubah hasil — beri tahu user mana yang masuk kategori
   mana, jangan campur aduk tanpa penjelasan.
2. **Verifikasi > percaya laporan**, termasuk laporan dari subagent AI lain.
   Laporan adalah titik awal investigasi, bukan kebenaran final.
3. **Ukur risiko sebelum bertindak.** Bug yang mempengaruhi hasil model/uang
   sungguhan (di sini: prediksi trading) butuh kehati-hatian lebih besar
   daripada duplikasi kode yang murni kosmetik — meski dua-duanya sama-sama
   "temuan review kode".
4. **Untuk perubahan besar/berisiko, tanya dulu** — bahkan kalau secara teknis
   aman (misal dead code, bisa di-restore lewat git). Biaya bertanya jauh
   lebih murah daripada biaya mengerjakan hal yang ternyata tidak diinginkan.
5. **Tunjukkan bukti kerja** (jumlah baris berubah, hasil test, hasil smoke
   test) di ringkasan akhir — bukan cuma klaim "sudah saya perbaiki".

---

## 4. Cara memakai ulang metodologi ini untuk proyek AI berikutnya

Kalau nanti membangun proyek AI baru dan butuh proses serupa, cukup minta:
*"pakai metodologi di `METODOLOGI_REVIEW_DAN_CLEANUP_KODE.md` dari proyek
AI Trading"* — lalu sesuaikan bagian 2 (template instruksi subagent) dengan
domain proyek baru itu (ganti daftar file & konteks proyek, kategori temuan
di bagian 1-5 biasanya tetap relevan apa pun jenis proyeknya).
