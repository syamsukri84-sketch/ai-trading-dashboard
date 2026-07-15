# Status Proyek AI Trading — Baca Ini Dulu

**Terakhir diperbarui**: 2026-07-14
**Tujuan file ini**: supaya siapa pun (Anda, atau AI assistant lain di VSCode -- Copilot,
Cursor, Claude, dll) bisa langsung paham cara kerja sistem ini dan apa yang sudah
dikerjakan TANPA harus membaca ulang seluruh ~14.600 baris kode dari nol. Baca file
ini dulu sebelum menyentuh kode.

**PERINGATAN PENTING**: `SYSTEM_PROMPT_TRADING_APP.md` dan `QUICK_START_GUIDE.md` di
folder yang sama **SUDAH USANG** (ditulis Juni 2026, fase perencanaan awal). Dokumen itu
menggambarkan arsitektur FastAPI+SQLite+IsolationForest sebagai inti sistem -- itu **TIDAK
LAGI BENAR**. Arsitektur nyata yang jalan sekarang dijelaskan di file ini. Jangan pakai
kedua file itu sebagai acuan arsitektur; boleh dibaca untuk konteks sejarah saja.

---

## 1. Apa proyek ini

Dashboard AI untuk membantu keputusan trading saham Indonesia (IDX), dipakai sendiri oleh
pemilik proyek (bukan produk multi-user). Entry point yang SUNGGUHAN dipakai:
**`streamlit_app.py`** (jalankan lewat `BUKA_AI_TRADING.bat` atau
`streamlit run streamlit_app.py`). Data diperbarui otomatis tiap sore hari bursa lewat
Windows Task Scheduler (`scripts/daily_global_workflow_cli.py`).

## 2. Cara kerja sistem (arsitektur nyata, bukan yang di dokumen lama)

```
Harga OHLCV (Yahoo Finance/yahooquery) + Indeks ^JKSE
        |
        v
Feature Engineering (src/data_pipeline/feature_engineer.py)
  -> feat_rsi_14, feat_macd, feat_atr_14, feat_bb_width, feat_obv, feat_beta_60,
     feat_corr_60, feat_ihsg_*, dll (lihat generate_features untuk daftar lengkap)
        |
        v
Model-model PARALEL per saham (run_analysis.py, src/models/):
  - DirectionClassifier x4 (LightGBM/XGBoost/RandomForest/Logistic) -> arah H+1
  - Direction-Ensemble -> gabungan ke-4 di atas, dibobot rekam jejak akurasi historis
    (src/trading/reliability_ensemble.py)
  - PriceProjector (XGBoost/LightGBM) -> return% H+3/H+5/H+10
  - LSTMPriceProjector -> sama tapi lihat urutan 20 hari (sequence), bukan snapshot
  - GARCHModel -> volatilitas & VaR (info risiko, BUKAN dipakai untuk stop-loss;
    stop-loss aktual pakai ATR)
  - IsolationForestModel -> deteksi hari anomali (bukan arah/harga)
  - GlobalDirectionModel/GlobalPriceModel (src/models/global_models.py) -> versi
    dilatih dari SEMUA saham sekaligus (feat_ticker_id sebagai pembeda), dipakai
    workflow harian otomatis karena lebih murah dilatih daripada per-saham satu-satu
        |
        v
Walk-Forward Validation (src/models/walk_forward.py)
  -> selalu dibandingkan ke baseline tebak-mayoritas/return-nol pada fold yang SAMA
  -> sejak 2026-07-12: JUGA dikoreksi Benjamini-Hochberg FDR lintas seluruh ticker
     (screen_genuine_edge.py) supaya ambang effect-size tidak meloloskan false-positive
     murni dari varians sampel
        |
        v
Trust Gate (src/utils/accuracy_tracker.py: get_model_trust_audit)
  -> butuh LOLOS DUA SUMBER: (1) live track record, (2) walk-forward genuine edge
  -> compute_unified_trust_badge() -> badge "Terverifikasi Ganda" di UI
        |
        v
streamlit_app.py -- Beranda / Ringkasan Harian / Ranking Prediksi / Akurasi Model / dst.
```

**Pipeline KEDUA yang terpisah dan TIDAK terhubung** ke atas: `fastapi_app.py` + `routes.py`
+ `src/database/` (isolation forest + conformal predictor + SQLite). Ini eksperimental,
tidak pernah divalidasi walk-forward, dan `streamlit_app.py` **tidak pernah memanggilnya**.
Sudah diberi warning docstring di kedua file itu (2026-07-12). Jangan bingung menganggap ini
bagian dari sistem produksi.

## 3. TEMUAN PALING PENTING -- baca sebelum mempercayai sinyal apa pun

Screening walk-forward penuh terhadap 265 ticker (`data/edge_screening_status.json`):
**0 dari 265 ticker punya edge nyata di H+1/H+3/H+5** (rata-rata H+1 sekitar -2.2pp
DI BAWAH baseline tebak-mayoritas). H+10 cuma 3/265 lolos ambang effect-size, dan itu pun
kemungkinan besar noise (dikonfirmasi lewat koreksi FDR yang baru ditambahkan). Ini
konsisten dengan literatur keuangan bahwa gerakan harga jangka pendek mendekati random walk.

**Implikasi untuk kerja lanjutan**: jangan buru-buru menambah fitur/model baru untuk
"mengejar akurasi" -- itu sudah dicoba (regularisasi, model lebih sederhana, window
training lebih besar) dan tidak menciptakan edge yang tidak ada di data. Prioritas yang
lebih masuk akal: (a) reframing target (mis. model "layak ditindaklanjuti atau tidak" alih-
alih arah), atau (b) pindah ke prescriptive analytics (position sizing, portfolio) yang
tidak butuh mengalahkan pasar dulu untuk berguna. Lihat `ROADMAP_PRESCRIPTIVE_ANALYTICS.md`.

## 4. Apa yang SUDAH dikerjakan (ringkasan kronologis)

### Sesi optimasi model (awal Juli 2026) -- lihat `LAPORAN_OPTIMASI_MODEL_JULI_2026.md`
- Fix bug `^JKSE` gagal download (suffix `.JK` salah tambah ke simbol indeks).
- Dedup kode NLP sentiment + tambah held-out evaluation akurasi.
- `walk_forward.py` mulai selalu hitung baseline naif (sebelumnya tidak ada).
- Regularisasi hyperparameter classifier arah (LightGBM/XGBoost/RandomForest).
- Fix bug trust audit produksi (`walk_forward_score` yang isinya bukan walk-forward asli).

### Cognitive Dashboard (2026-07-12) -- lihat `ROADMAP_COGNITIVE_DASHBOARD.md`
- **A1 SELESAI**: feedback eksplisit (tombol Ikuti/Lewati/Berguna + log CSV).
- **A3 SELESAI**: lapisan personalisasi (`src/trading/personalization.py`) -- skor
  personal per ticker, mute/unmute, SENGAJA terpisah dari model prediksi (lihat
  prinsip desain di bagian 6 di bawah).
- **B3 SELESAI**: riwayat regime pasar (`src/trading/market_regime.py`,
  `data/regime_history.csv`).
- **A2, A4, B1, B2 BELUM dikerjakan** -- lihat bagian 5 untuk rekomendasi sequencing.

### Audit menyeluruh + perbaikan (2026-07-12, sesi yang sama dengan file ini ditulis)
4 subagent baca penuh seluruh codebase, temuan lengkap ada di memory assistant
(`project_codebase_audit_2026_07`). Yang SUDAH diperbaiki hari ini:
- **Gap trust UI**: tab Ranking Prediksi & Rekomendasi Model per Saham sekarang
  membawa kolom status trust/edge (sebelumnya bisa menampilkan ticker "JANGAN DIIKUTI"
  di posisi teratas tanpa peringatan).
- **Mismatch kalibrasi**: walk-forward sekarang menguji model TERKALIBRASI
  (`DirectionClassifier.build_walk_forward_estimator()`), identik dengan yang
  dideploy ke prediksi live -- sebelumnya walk-forward diam-diam menguji estimator
  mentah pra-kalibrasi.
- **Koreksi multiple-testing (FDR)**: `apply_fdr_correction()` baru di
  `walk_forward.py`, dipakai `screen_genuine_edge.py` lintas seluruh ticker per
  horizon. `has_genuine_edge_*` sekarang butuh effect-size DAN signifikansi FDR.
- **Reliabilitas workflow harian**: step 2-4 di `daily_global_workflow_cli.py`
  dibungkus try/except (selalu tulis status FAILED, bukan crash senyap), log Task
  Scheduler di-redirect ke `logs/*.log`, dashboard Beranda alert kalau run terakhir
  gagal/basi (`get_latest_daily_workflow_run()`).
- **Atomic write**: `src/utils/atomic_io.py` baru, dipakai di 6 file tracking CSV
  (mencegah korupsi kalau proses terhenti paksa di tengah tulis).
- **Dataset sentiment divendor**: `data/sentiment/external/financial_news_clean.csv`
  (1320 baris) -- tidak lagi bergantung folder `../tugas_nlp_ai_trading/` di luar proyek.
- **Dead code dihapus**: `src/models/backtest_engine.py`,
  `src/data_pipeline/lq45_downloader.py`, `check_data.py`, 3 folder kosong
  (`src/api`, `src/backtesting`, `src/frontend`).
- **Bug KeyError diperbaiki** di `routes.py:147`; `fastapi_app.py`/`routes.py`
  diberi warning docstring "experimental, tidak divalidasi".
- **Test baru**: 18 di `tests/test_run_analysis.py` (fungsi dedup/skip harian yang
  sebelumnya 0% tercover), 8 di `test_prediction_upgrades.py` (FDR+kalibrasi), 4 di
  `test_atomic_io.py`. Total test **118 -> 143 lulus**.
- Verifikasi: `py_compile` bersih, `AppTest` (eksekusi penuh dashboard tanpa browser)
  0 exception, smoke run nyata `daily_global_workflow_cli.py` & `screen_genuine_edge.py`.

**SEMUA perubahan di atas MASIH BELUM DI-COMMIT ke git per tanggal file ini ditulis**
(lihat bagian 7). `git log` terakhir masih di commit `7f4c28b`.

### Efisiensi komputasi LSTM (2026-07-13)
Temuan penting: `LEGACY_MODELS_ENABLED = False` (streamlit_app.py:56) -- SELURUH tombol
"Analisis Manual"/"Job Background"/"Retrain" per-saham (yang memanggil `run_full_analysis`,
termasuk training LSTM) **sudah dinonaktifkan di UI** sejak sebelum sesi ini, digantikan
Global Model. Satu-satunya jalur `run_full_analysis` yang masih aktif: fitur **Backfill**
(sudah lama punya pola `include_lstm` opt-in, default mati) dan pemanggilan langsung
`python scripts/background_analysis_job.py` dari terminal (tidak melalui gate UI).

Perbaikan yang dilakukan (mengikuti pola `backfill_include_lstm` yang sudah ada):
- `run_full_analysis` sekarang punya parameter `include_lstm: bool = False` (dulu LSTM
  selalu jalan, dilatih 2x per panggilan, tanpa opsi mati). Kalau `include_lstm=False`,
  `lstm_projection`/`next_day_lstm_projection` jadi `None` dan seluruh downstream
  (log_prediction, print, summary dict) menangani `None` dengan aman.
- **Cache reuse**: `_load_cached_lstm_if_trained_today()` (run_analysis.py) mengecek
  `data/models/model_registry.json` -- kalau sudah ada artifact LSTM untuk
  ticker+horizon+purpose dengan `trained_until_date` == tanggal data terkini, model
  dimuat ulang lewat `load_model_artifact()`, TIDAK dilatih ulang dari nol. Diverifikasi
  nyata: 2x panggilan `run_full_analysis(include_lstm=True)` pada BBCA menghasilkan
  `lstm_projected_return_pct` identik bit-for-bit DAN `saved_at`/`training_run_id` di
  registry tidak berubah -- bukti cache benar-benar dipakai, bukan retrain diam-diam.
- `scripts/background_analysis_job.py` dapat flag `--include-lstm` (default OFF).
- UI: checkbox "Sertakan LSTM" ditambahkan di section "Job Background Analisis" dan
  "Update Harga + analisis ulang otomatis" (streamlit_app.py) -- WALAUPUN section-section
  ini sendiri saat ini masih di-disable oleh `LEGACY_MODELS_ENABLED=False`, checkbox tetap
  di-wire dengan benar supaya otomatis berfungsi kalau flag itu diaktifkan lagi nanti.
- Belum dikerjakan (disebutkan sebagai langkah lanjutan, belum diminta user): validasi
  walk-forward untuk LSTM (perlu fungsi walk-forward terpisah karena interface LSTM beda
  dari model sklearn-style; reuse `_paired_one_sided_pvalue`/`apply_fdr_correction`),
  dan paralelisasi `screen_genuine_edge.py` (screening mingguan ~1 jam, saat ini sekuensial
  padahal tiap ticker independen -- peluang percepatan terbesar yang belum digarap).

### Simplifikasi UI dashboard (2026-07-13)
Tujuan: kurangi information overload supaya hasil lebih mudah dipahami. Temuan awal:
`display_mode` (Pemula/Trader/Audit) ternyata cuma menyederhanakan 1 tabel di 1 tab
(Ringkasan Harian) -- 5 dari 6 tab utama tampil sama beratnya ke semua mode. Perbaikan:
- **Mode Pemula sekarang benar-benar menyembunyikan** (bukan cuma teks help yang
  mengklaim begitu): sub-tab "🔧 Detail Teknis Lainnya" (6 nested sub-tab di Akurasi
  Model) dan tab "Ranking Mentah (Riset)" -- keduanya diganti pesan singkat "buka mode
  Trader/Audit kalau perlu" untuk Pemula, isi lengkap tetap ada & jalan normal di
  Trader/Audit (diverifikasi lewat AppTest ketiga mode).
- **Tab "Ranking Prediksi" diganti nama jadi "Ranking Mentah (Riset)"** + sub-tab
  "Rekomendasi Model per Saham" jadi "Rekomendasi Model per Saham (Riset)" -- menandai
  eksplisit bahwa ini BELUM tergate trust/edge, beda dari "Ringkasan Harian" yang sudah.
  Urutan sub-tab Akurasi Model ditukar: "Model Trust Audit" (sudah tergate) sekarang
  jadi sub-tab PERTAMA, bukan lagi "Rekomendasi Model per Saham" (akurasi mentah).
- **UI legacy mati DIHAPUS dari tampilan** (bukan cuma `disabled=True`, yang sebelumnya
  membuat control tetap tampil abu-abu tanpa fungsi): section "8. Job Background
  Analisis", "9. Audit Kelengkapan Analisis Setelah Update", checkbox "Jalankan analisis
  ulang otomatis setelah update data", tombol "Paksa Analisis Ulang dari Data Lokal" --
  semua dibungkus `if LEGACY_MODELS_ENABLED:` sehingga hilang total selama flag itu
  False. Tombol "Retrain Lama Dinonaktifkan" (permanen `disabled=True`, bukan lewat flag)
  dihapus sepenuhnya, kolom Tombol Cepat After Market jadi 3 (dari 4, semua actionable).
- **Alat riset perbandingan sentiment engine dipindah keluar dashboard**: tombol
  "Bandingkan Semua Engine Sentimen (Robust, 5 Split)" (TF-IDF vs IndoBERT dkk, unduh
  model 500MB-1GB, ~beberapa menit) dihapus dari streamlit_app.py, jadi script baru
  `scripts/compare_sentiment_engines_cli.py` (`python scripts/compare_sentiment_engines_cli.py
  [--dataset PATH] [--output PATH]`) -- fungsi backend (`compare_sentiment_engines_repeated`)
  tidak berubah, cuma UI trigger-nya pindah dari dashboard produksi ke CLI manual.
  Fitur "Bangun Dataset Sentimen Lokal" (operasional, dipakai workflow harian) TETAP di
  dashboard, tidak ikut dipindah.
- Verifikasi: py_compile bersih, 143 test tetap lulus, AppTest 0 exception di ketiga mode
  (Pemula/Trader/Audit) dijalankan terpisah.
- Belum dikerjakan (di luar scope sesi ini, kandidat lanjutan kalau mau lebih jauh):
  mode-gating untuk section maintenance lain di Workflow Harian (Audit OHLC, Backfill),
  dan penyesuaian default tampilan multi-horizon (H+1/H+3/H+5/H+10 semua tampil sekaligus
  di banyak tempat, padahal cuma H+1 yang jadi horizon keputusan utama).

### Validasi walk-forward LSTM + paralelisasi screening (2026-07-14)
Dua item yang sebelumnya cuma direkomendasikan, sekarang dikerjakan:

**Validasi walk-forward LSTM** -- LSTM sebelumnya SATU-SATUNYA model yang tidak pernah
divalidasi walk-forward (interface `train(df)`/`predict(df)` tidak kompatibel dengan
`walk_forward_return_validation` yang menerima X/y datar). Ditambahkan
`walk_forward_sequence_model_validation()` (src/models/walk_forward.py) -- metodologi
IDENTIK (purge gap, baseline zero/mean return, p-value berpasangan per-fold via
`_paired_one_sided_pvalue`), tapi tiap fold memanggil `model.train(train_df, epochs=...)`
lalu `model.predict(window)` berulang per hari test (LSTM cuma prediksi baris terakhir
dari df yang diberikan, beda dari model sklearn-style yang bisa predict banyak baris
sekaligus). Script baru `scripts/validate_lstm_walk_forward.py`
(`python scripts/validate_lstm_walk_forward.py [--tickers ...] [--limit N] [--horizon 3]
[--epochs 3]`) -- SENGAJA TERPISAH dari screen_genuine_edge.py karena LSTM jauh lebih
mahal dilatih ulang per fold; jangan jadikan bagian screening mingguan rutin.
- Diverifikasi data nyata: BBCA H+3 (32 fold, epochs=3) selesai 10.4 detik. Smoke test
  4 ticker (BBCA/BBRI/TLKM/ASII) selesai 25.3 detik total.
- **Hasil**: LSTM juga TIDAK punya edge nyata (edge negatif di semua ticker yang dites,
  p-value jauh dari signifikan) -- konsisten dengan temuan model lain di proyek ini.
  Jangan buru-buru simpulkan LSTM "harus dipertahankan" tanpa dasar ini.
- 4 test baru di `tests/test_walk_forward_sequence.py` (data sintetis, lookback kecil
  supaya cepat, ~30 detik total).

**Paralelisasi `screen_genuine_edge.py`** -- sebelumnya sekuensial murni (~1 jam untuk
265 ticker), padahal tiap ticker independen (baca CSV sendiri, latih model sendiri,
tidak ada file bersama yang ditulis selama screening). Sekarang pakai
`ProcessPoolExecutor` (flag baru `--workers N`, default 0 = otomatis pakai semua core
CPU; `--workers 1` = jalur sekuensial lama, dipertahankan untuk debug/mesin 1 core).
Worker top-level `_screen_ticker_worker()` membuat `DataLoader`/`FeatureEngineer` fresh
di tiap proses (wajib untuk Windows spawn mode, tidak mewarisi state proses induk).
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS` di-set ke 1 di awal file (pola yang sama
seperti `scripts/sync_mongodb_cli.py`) supaya tiap proses tidak oversubscribe CPU
dengan thread internal BLAS/LightGBM-nya sendiri.
- **Diverifikasi ketat**: 8 ticker dijalankan sequential (`--workers 1`, 156.8 detik)
  vs parallel (auto, 46.0 detik) -- **hasil di-diff field-by-field, 100% identik**,
  speedup 3.41x (di mesin 12-core, dibatasi jumlah ticker=8 bukan jumlah core; untuk
  screening penuh 265 ticker potensi speedup mendekati jumlah core mesin).
- Koreksi FDR tetap dijalankan SETELAH semua ticker selesai (baik sequential maupun
  parallel) -- urutan penyelesaian ticker berbeda-beda saat paralel, tapi ini tidak
  memengaruhi korektnas karena FDR correction tidak bergantung urutan.

Verifikasi akhir sesi ini: py_compile bersih semua file, **147 test lulus** (dari 143,
+4 baru).

### Perbaikan launcher .bat (2026-07-14)
`BUKA_AI_TRADING.bat`/`BUKA_AI_TRADING_ONLINE_FREE.bat` gagal total dengan error
"Fatal error in launcher" -- ternyata `.venv\Scripts\streamlit.exe` di proyek ini adalah
launcher stub yang path Python tertanamnya menunjuk ke proyek LAIN
(`D:\PROYEK_PYTHON\GEMINI AI1\.venv\Scripts\python.exe`, kemungkinan `.venv` sempat
dibuat/dicopy tidak bersih). Diperbaiki: kedua .bat sekarang panggil
`.venv\Scripts\python.exe -m streamlit run ...` (bypass launcher exe yang rusak) --
pola yang sudah terbukti selalu berhasil sepanjang sesi-sesi sebelumnya. Dikonfirmasi
seluruh `.bat`/`.ps1` lain di proyek sudah memanggil `python.exe` langsung, jadi cuma
2 file ini yang perlu diperbaiki.

### Fitur baru: Cari Kata Kunci Sentimen Otomatis via Gemini API (2026-07-14)
User minta alur baru di tab Sentimen Pasar -- SEBELUM mengisi field "Ticker untuk
pencarian berita"/"Query pencarian" (sub-tab Ambil Berita Otomatis), AI mencari dulu
ticker mana yang relevan untuk sentimen pasar Indonesia saat ini, lalu isi otomatis
kedua field itu. User awalnya minta ini lewat link `gemini.google.com/app/...` --
**dikoreksi**: itu sesi chat pribadi berbasis login, bukan API, tidak bisa
diintegrasikan programatik (dikonfirmasi via WebFetch: perlu login). Setelah
klarifikasi ke user (`AskUserQuestion`), diputuskan pakai **Gemini API resmi**
(`google-genai`, endpoint `generateContent`, BUKAN chat UI) dengan universe ticker =
seluruh `config/stocks.yaml` apa adanya (tidak ada data "Papan Utama" resmi di proyek
ini, user pilih pakai yang ada).

**Modul baru `src/nlp/gemini_keyword_search.py`**:
- `get_gemini_config()`/`check_gemini_status()` -- pola sama seperti
  `mongo_store.py` untuk MongoDB: integrasi eksternal opsional, degradasi anggun
  kalau `GEMINI_API_KEY` belum diset (fitur nonaktif, sisa dashboard tetap jalan).
- `suggest_sentiment_keywords(tickers, top_n)` -- minta Gemini (model
  `gemini-flash-latest`, alias yang otomatis ikut versi terbaru Google supaya tidak
  basi seperti `gemini-2.5-flash` yang sudah dideprecate untuk API key baru per
  2026-07-14) mencari **dengan Google Search grounding** (`types.Tool(google_search=...)`)
  ticker paling relevan dari universe yang diberikan + saran query pencarian berita.
- **Fallback jujur kalau kuota grounding habis**: di tier gratis, kuota Google Search
  grounding jauh lebih ketat daripada kuota generate_content biasa (dikonfirmasi
  langsung dengan API key nyata milik user: generate_content polos berhasil, dengan
  grounding tool kena 429 berkali-kali). Kalau grounding gagal karena kuota, otomatis
  MUNDUR ke generate_content tanpa pencarian live -- hasil tetap dikembalikan tapi
  field `grounded=False` + `warning` eksplisit bilang ini BUKAN hasil pencarian
  langsung. Tidak pernah gagal total kalau kuota grounding habis.
- Parsing respons pakai regex longgar (`TICKER: X | QUERY: Y | ALASAN: Z`), robust
  terhadap respons kosong/tidak sesuai format -- selalu `raw_response` disertakan untuk
  debug manual.

**UI** (`streamlit_app.py`, sub-tab "Ambil Berita Otomatis"): section baru "1. Cari
Kata Kunci Sentimen Otomatis (Gemini AI)" SEBELUM "2. Ambil Berita untuk Ticker
Terpilih". Tiap saran ticker dari Gemini punya tombol "Pakai" yang mengisi
`st.session_state["auto_news_ticker"]`/`["auto_news_query"]` lalu `st.rerun()` --
memicu field Ticker/Query di langkah 2 terisi otomatis. Fitur digate oleh
`sentiment_online_enabled` yang sudah ada (`AI_TRADING_EXTERNAL_SERVICES=true` +
mode online aktif), konsisten dengan fitur online lain.

**Konfigurasi**: `GEMINI_API_KEY`+`GEMINI_MODEL` ditambahkan ke `.env` (asli, sudah
terisi API key milik user -- **aman**, `.env` sudah gitignored, dikonfirmasi tidak
ter-stage git) dan `.env.example` (placeholder). `AI_TRADING_EXTERNAL_SERVICES`
diset `true` di `.env` supaya fitur online (termasuk ini) aktif. Dependency
`google-genai>=0.3.0` ditambah ke `requirements.txt`, terinstal di `.venv`.

**Diverifikasi dengan API key NYATA milik user** (bukan cuma mock): dipanggil
langsung berkali-kali selama development -- berhasil penuh dengan grounding,
berhasil dengan fallback (kuota grounding habis), dan gagal graceful (kuota + respons
kosong) -- kode menangani semua kasus tanpa crash. **Diverifikasi interaktif via
AppTest**: klik tombol "Cari Kata Kunci" -> saran tersimpan di session_state -> klik
"Pakai" -> `auto_news_ticker`/`auto_news_query` terisi otomatis (dikonfirmasi
`ADRO`/`"ADRO saham dividen restrukturisasi"` di satu run nyata). Sempat ketemu
warning kosmetik Streamlit ("value= sekaligus session_state key") saat auto-fill --
diperbaiki dengan pola `st.session_state.setdefault(...)` alih-alih `value=` langsung.

**PENTING -- catatan konsumsi kuota**: API key yang dipakai untuk verifikasi kemungkinan
sudah banyak terpakai kuota gratisnya selama development sesi ini (banyak percobaan
manual + AppTest interaktif). Kalau fitur ini langsung menunjukkan "kuota habis" saat
pertama dipakai user, itu wajar -- tunggu reset kuota harian atau cek
https://ai.dev/rate-limit.

**Test baru**: 23 test di `tests/test_gemini_keyword_search.py`, SEMUA pakai mock
(`unittest.mock.patch("google.genai.Client")`), tidak ada satupun yang memanggil API
sungguhan -- cepat (~2 detik) dan tidak menghabiskan kuota. Mencakup: parsing
respons, deteksi error kuota, fallback grounding->non-grounding, error non-kuota
(tidak fallback), respons kosong, respons tidak terparsing, top_n, strip suffix .JK.

Verifikasi akhir: py_compile bersih, **170 test lulus** (dari 147, +23 baru), AppTest
0 exception (dengan dan tanpa GEMINI_API_KEY).

## 5. Apa yang BELUM dikerjakan -- prioritas & alasan

### Prescriptive Analytics -- BELUM DIMULAI SAMA SEKALI
Lihat `ROADMAP_PRESCRIPTIVE_ANALYTICS.md`. Position sizing Kelly/risk-parity, optimasi
portofolio lintas-ticker, backtest strategi penuh (bukan cuma akurasi arah), simulator
what-if, rebalancing. Aturan wajib: batasi rekomendasi prescriptive HANYA untuk ticker
"✅ Terverifikasi Ganda".

### Cognitive Dashboard lanjutan -- direkomendasikan JANGAN buru-buru (2026-07-12)
- **B1 (drift detection formal)**: `data/regime_history.csv` baru mulai terisi hari
  ini -- tunggu minimal ~60 hari bursa sebelum rolling-window drift metric bermakna.
  Kalau nanti dikerjakan, mulai dari perbandingan rolling sederhana, JANGAN langsung
  pakai ADWIN/DDM.
- **A4 (bandit personalisasi)**: dipertimbangkan ulang bentuknya, bukan cuma ditunda.
  Bandit (Thompson Sampling dkk.) butuh volume interaksi besar untuk konvergen --
  dashboard ini SATU user dengan klik feedback sporadis, jadi bandit sungguhan
  kemungkinan besar tidak pernah konvergen bermakna. Skor statis yang sudah ada
  (`compute_personal_scores`) kemungkinan sudah cukup untuk skala ini.
- **B2 (model per-regime)**: turunkan prioritas jauh atau drop. Riwayat harga per
  ticker cuma ~950-1000 baris; dipecah 3 regime jadi ~300-400 baris/model regime --
  pas-pasan/kurang dari kebutuhan walk-forward (252 baris/fold), dan regime CRASH
  (jarang terjadi) kemungkinan tidak akan pernah punya data cukup. Risiko nyata:
  bikin model LEBIH lemah, bukan lebih baik.
- **A2 (feedback implisit)**: effort sedang, manfaat lebih kecil dari A1, prioritas rendah.

### Test coverage lanjutan
`run_full_analysis`/`run_backfill_analysis` sendiri (orchestrator inti) baru punya test
untuk fungsi dedup murninya (`_prediction_exists` dkk) -- belum ada integration test
untuk fungsi orchestrator utuh (butuh fixture data + mocking, effort lebih besar).

### Posisi di beberapa kerangka maturity dashboard (dibahas 2026-07-12, belum ada dokumennya)
- **Single Pane of Glass**: parsial. Sudah satu aplikasi + trust badge yang mensintesis
  2 sumber jadi 1 indikator + expander "Detail Per Saham" per-ticker. Belum penuh karena:
  MongoDB cuma backup satu-arah (CSV->Mongo, BUKAN sumber baca dashboard -- perlu koreksi
  mental model kalau ada asumsi sebaliknya), dan pipeline FastAPI terpisah/tak terlihat.
- **Augmented Analytics** (narasi otomatis dari data): sebagian. "Alasan Utama" di
  `build_daily_decision_board` sudah genuine narasi dinamis. XAI/SHAP dan ringkasan
  sentimen masih angka/tabel mentah atau template tetap, belum dirangkai jadi kalimat.
- **Cognitive Dashboard**: lihat di atas (A1/A3/B3 selesai = instrumented, tapi A4/B1/B2
  yang belum = bagian yang bikin genuinely "belajar", bukan cuma "tercatat").
- **Shneiderman's Mantra** (Overview first / Zoom & filter / Details-on-demand): overview
  & filter kuat (tab Beranda pertama + filter sidebar+per-tab). Details-on-demand ada
  tapi lewat selectbox pilih-ulang, bukan klik langsung baris/titik data (Streamlit
  `on_select`/`selection_mode` belum dipakai sama sekali di `streamlit_app.py`).

## 6. Prinsip desain yang WAJIB dipegang (jangan dilanggar tanpa sadar)

1. **Umpan balik pengguna TIDAK BOLEH memengaruhi bobot model prediksi** -- hanya boleh
   memengaruhi lapisan personalisasi/ranking/presentasi. Lihat penjelasan lengkap di
   `ROADMAP_COGNITIVE_DASHBOARD.md` bagian atas.
2. **`EDGE_THRESHOLD_PCT` (3.0pp) + koreksi FDR** adalah syarat GANDA untuk status
   "genuine edge" -- jangan longgarkan salah satu demi menambah jumlah sinyal yang
   ditampilkan.
3. **Model yang divalidasi walk-forward HARUS identik dengan model yang dideploy**
   (termasuk wrapping kalibrasi) -- kalau mengubah hyperparameter produksi, factory
   walk-forward harus ikut berubah otomatis (lihat komentar di `run_analysis.py`
   sekitar baris 720-740).
4. **Prescriptive analytics HANYA untuk ticker "✅ Terverifikasi Ganda"** -- jangan
   longgarkan demi menambah cakupan rekomendasi.
5. **File tracking CSV ditulis via `atomic_write_csv`** (`src/utils/atomic_io.py`),
   bukan `.to_csv()` langsung, untuk file yang ditulis ulang penuh tiap update.

## 7. Status git & cara verifikasi sebelum lanjut kerja

- Branch `main`, commit terakhir `7f4c28b`. **Banyak perubahan uncommitted** (termasuk
  seluruh pekerjaan sesi 2026-07-12 di atas) -- cek `git status` sebelum mengasumsikan
  apa yang "resmi" masuk kode.
- Sebelum mengubah kode lebih lanjut, jalankan baseline ini dulu:
  ```
  python -m pytest -q                    # harus 143 lulus (per 2026-07-12)
  python -m py_compile streamlit_app.py run_analysis.py
  ```
- Smoke test dashboard tanpa perlu browser (lebih reliable dari `curl`, karena benar-
  benar mengeksekusi seluruh script termasuk semua tab, bukan cuma shell HTML):
  ```python
  from streamlit.testing.v1 import AppTest
  at = AppTest.from_file('streamlit_app.py', default_timeout=180)
  at.run()
  print(len(at.exception))  # harus 0
  ```

## 8. Peta dokumen lain di folder ini

| File | Isi | Status |
|---|---|---|
| `LAPORAN_OPTIMASI_MODEL_JULI_2026.md` | Laporan 4 fase optimasi model Juli 2026 | Akurat, historis |
| `ROADMAP_COGNITIVE_DASHBOARD.md` | Rencana A1-A4/B1-B3, checklist [x]/[ ] | Akurat, sebagian sudah [x] |
| `ROADMAP_PRESCRIPTIVE_ANALYTICS.md` | Rencana position sizing/portofolio, belum mulai | Akurat, semua masih [ ] |
| `METODOLOGI_REVIEW_DAN_CLEANUP_KODE.md` | Cara kerja review-codebase-besar yang dipakai (subagent paralel dsb) | Reusable untuk proyek lain juga |
| `SYSTEM_PROMPT_TRADING_APP.md` | Spek awal (Juni 2026) | **USANG, jangan dipakai untuk arsitektur** |
| `QUICK_START_GUIDE.md` | Setup awal Copilot (Juni 2026) | **USANG, arsitektur sudah berubah** |

## 9. Catatan untuk AI assistant yang membaca file ini

- File ini snapshot per **2026-07-12**. Sebelum menindaklanjuti klaim spesifik (nomor
  baris, nama fungsi, status test), **verifikasi ulang ke kode saat ini** -- kode bisa
  berubah sejak file ini ditulis.
  - Nomor baris di prinsip #3 (bagian 6) dan nama fungsi lain kemungkinan bisa
    sedikit bergeser kalau ada edit lain di antaranya.
- Kalau user minta "lanjutkan pekerjaan", tanyakan dulu item mana yang dimaksud dari
  daftar bagian 5 -- jangan asumsikan urutan otomatis tanpa konfirmasi, karena beberapa
  item (B1, B2) sengaja direkomendasikan DITUNDA, bukan dikerjakan berikutnya secara
  default.
- Metodologi kerja yang sudah terbukti dipakai & disukai di proyek ini (lihat
  `METODOLOGI_REVIEW_DAN_CLEANUP_KODE.md`): orientasi dulu (git status/log) -> klarifikasi
  fokus ke user -> untuk codebase besar, pecah jadi domain & review paralel -> verifikasi
  ulang temuan sebelum eksekusi -> py_compile + pytest + smoke test tiap selesai perubahan
  -> checkpoint ke user secara berkala, jangan cuma di akhir.
