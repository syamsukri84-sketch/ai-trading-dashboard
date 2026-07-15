# Laporan Optimasi Model AI Trading (Juli 2026)

Laporan ini merangkum hasil 4 fase optimasi yang diminta: (1) Data-Centric Tuning,
(2) NLP Tuning, (3) Algorithmic Tuning, (4) Strategy & Backtest Tuning.
Ditulis jujur berdasarkan bukti (pengujian walk-forward, test suite, dan
pembacaan kode langsung) -- termasuk bagian yang **tidak** berhasil, karena itu
sama pentingnya untuk diketahui sebelum modal sungguhan dipertaruhkan pada
sinyal model ini.

## Ringkasan Eksekutif

- **Bug nyata ditemukan & diperbaiki** di 3 tempat berbeda: pipeline data
  (`^JKSE.JK` tidak pernah berhasil terunduh), infrastruktur evaluasi
  (`walk_forward.py` tidak pernah membandingkan ke baseline naif), dan gating
  kepercayaan produksi (`accuracy_tracker.py` punya kolom bernama
  "walk_forward_score" yang isinya BUKAN hasil walk-forward, cuma akurasi
  mentah -- lolos audit trust tanpa pernah dibandingkan ke baseline).
- **Regularisasi hyperparameter classifier arah TERBUKTI membantu** (walk-forward
  tervalidasi): jarak rata-rata terhadap baseline membaik dari sekitar -5.8pp
  menjadi -2.3pp/-2.9pp pada sampel 12 ticker.
- **Temuan yang tidak menyenangkan, tapi jujur**: sampai laporan ini ditulis,
  TIDAK ADA konfigurasi yang diuji (regularisasi, model lebih sederhana,
  window training lebih besar) yang menghasilkan edge POSITIF nyata di atas
  baseline naif untuk prediksi arah H+1, pada seluruh 24 ticker yang
  diskrining. Ini konsisten dengan literatur keuangan bahwa pergerakan harga
  1-hari sangat mendekati random walk -- bukan sesuatu yang bisa "diperbaiki"
  lewat rekayasa model semata.
- Confidence dashboard, sentiment engine, dan trust-audit sekarang melaporkan
  angka yang bisa dipertanggungjawabkan (dibandingkan ke baseline / held-out
  test), bukan angka mentah yang bisa menyesatkan.

## Fase 1: Optimasi Fitur dan Data (Data-Centric Tuning)

**Bug ditemukan**: `src/data_pipeline/auto_updater.py` menambahkan suffix
`.JK` ke SEMUA simbol termasuk indeks `^JKSE`, menghasilkan `^JKSE.JK` yang
tidak dikenali Yahoo Finance (`404 Quote not found`). Dikonfirmasi langsung
lewat reproduksi: sebelum perbaikan, `run_auto_updater(tickers=['^JKSE'])`
selalu gagal; sesudah perbaikan, 954 baris data berhasil terunduh.

**Dampak**: seluruh fitur relative-strength/korelasi/beta-terhadap-market yang
bergantung pada data IHSG sebelumnya SELALU jatuh ke nilai default netral
(karena data tidak pernah ada). Sekarang fitur ini punya data sungguhan.

**Perubahan**:
- `src/data_pipeline/auto_updater.py`: deteksi `ticker.startswith("^")` sebelum
  menambahkan `.JK`.
- `scripts/daily_global_workflow_cli.py`, `scripts/update_prices_cli.py`:
  fetch `^JKSE` terpisah dari daftar ticker saham (supaya index tidak ikut
  masuk ke loop training/prediksi per-saham).

**Catatan jujur**: fitur ini punya data yang benar sekarang, tapi belum ada
pengujian terpisah yang mengisolasi seberapa besar kontribusinya ke akurasi
akhir -- itu tercampur dalam hasil walk-forward Fase 3 di bawah, bukan diukur
sendiri secara ablatif.

## Fase 2: Optimasi Model Bahasa (NLP Tuning)

**Duplikasi kode**: `_analyze_text_ml` dan `_predict_with_model` di
`src/nlp/sentiment_analyzer.py` isinya nyaris identik. Sekarang `_analyze_text_ml`
memanggil `_predict_with_model` -- satu sumber logika, bukan dua yang bisa
diverge diam-diam.

**Validasi akurasi ditambahkan**: sebelumnya `get_sentiment_engine_status()`
hanya melaporkan `"model_available": True` tanpa angka performa apa pun --
dashboard bilang "model tersedia" tapi tidak pernah bilang "seberapa bisa
dipercaya". Sekarang ada evaluasi held-out (split 75/25 stratified, terpisah
dari model produksi yang dilatih di 100% data) yang menghasilkan
`accuracy_pct` dan F1 per kelas, ditampilkan di dashboard.

**Catatan jujur -- ini bagian penting**: dataset sentimen cuma 215 baris dan
timpang (positive=109, neutral=88, **negative=18**). Split 75/25 berarti test
set kelas negative cuma berisi sekitar 4-5 baris -- F1 untuk kelas itu akan
sangat tidak stabil dan TIDAK BOLEH dipercaya sama seperti kelas lain yang
datanya lebih banyak. Ini sudah ditulis eksplisit sebagai caption di
dashboard, bukan disembunyikan.

**Belum dikerjakan (di luar scope sesi ini)**: hipotesis "kontrarian Indonesia"
(berita positif -> bearish, berita negatif -> bullish) yang dipakai
`build_trading_sentiment_summary` belum pernah di-backtest terhadap return
aktual. Ini masih asumsi heuristik, bukan hasil pengujian.

## Fase 3: Optimasi Model Prediktif (Algorithmic Tuning)

**Infrastruktur baseline** (prasyarat sebelum tuning apa pun bisa dipercaya):
`src/models/walk_forward.py` sebelumnya hanya melaporkan akurasi mentah, yang
ternyata MENYESATKAN -- classifier arah H+1 ditemukan kolaps ke tebakan kelas
mayoritas (predict_proba mampat di kisaran 0.40-0.47, tidak pernah melewati
ambang 0.5), tapi akurasinya bisa kelihatan tinggi murni karena base rate
periode test kebetulan searah. Sekarang setiap validasi walk-forward SELALU
menghitung baseline naif (tebak kelas mayoritas dari data training / return
nol / rata-rata return training) pada fold yang identik, dan melaporkan
`edge_vs_baseline_pct` -- angka yang jujur menunjukkan skill nyata, bukan
kebetulan. Hasil ini disurface ke output `run_analysis.py` dengan ambang
`EDGE_THRESHOLD_PCT=3.0` dan flag `has_genuine_edge_h1/h3/h5/h10`.

**Hyperparameter classifier arah diregularisasi**: `n_estimators`, `num_leaves`/
`max_depth`, `min_child_samples`/`min_child_weight`, `subsample`,
`colsample_bytree`, `reg_alpha`/`reg_lambda` diperketat pada
`DirectionClassifier` dan `GlobalDirectionModel` (LightGBM, XGBoost, dan
RandomForest). **Tervalidasi lewat walk-forward** pada sampel 12 ticker: jarak
rata-rata terhadap baseline membaik dari sekitar -5.8pp menjadi -2.3pp
(pengujian terisolasi) dan -2.9pp (kelas produksi penuh).

**Diuji juga (sesuai arahan Anda) dan hasilnya negatif**: model yang lebih
sederhana (logistic regression) TIDAK mengungguli tree model teregulasi;
window training lebih besar tidak menciptakan edge baru yang berarti.
Regularisasi mengurangi overfitting pada noise, tapi tidak bisa menciptakan
sinyal yang memang tidak ada di data.

**Temuan akhir yang jujur harus disampaikan**: setelah semua perbaikan di
atas, screening pada 24 ticker sampel menunjukkan **0 dari 24** yang mencapai
edge positif nyata (`edge_vs_baseline_pct >= 3.0pp`) untuk horizon H+1. Sinyal
di horizon lebih panjang (H+5/H+10) secara korelasi univariat terlihat lebih
kuat pada eksplorasi awal, tapi belum diverifikasi lewat walk-forward penuh
seperti H+1 -- ini kandidat pekerjaan lanjutan, bukan klaim yang sudah
dibuktikan.

## Fase 4: Optimasi Aturan Eksekusi (Strategy & Backtest Tuning)

**Temuan penting**: `src/trading/decision_support.py`
(`calculate_ai_confidence_score`, `build_decision_support`, `build_trade_gate`)
adalah **kode mati** -- tidak dipanggil di mana pun pada jalur produksi
(`streamlit_app.py`, `run_analysis.py`), hanya dipakai di test. Gating
kepercayaan yang SUNGGUHAN, yang benar-benar memengaruhi apa yang dilihat
pengguna di dashboard, ada di `src/utils/accuracy_tracker.py` fungsi
`get_model_trust_audit()`, dikonsumsi oleh `build_daily_decision_board()` di
`streamlit_app.py`.

**Bug ditemukan di sana**: audit punya kolom bernama `walk_forward_score`
yang, meski namanya begitu, isinya cuma salinan `direction_accuracy_pct`
mentah -- BUKAN hasil validasi walk-forward apa pun. Dan `beats_baseline`
murni berdasarkan `profit_factor >= 1.2`, tanpa pernah membandingkan akurasi
model terhadap tebakan "arah mayoritas" pada periode evaluasi yang sama.
Artinya model yang kebetulan selalu benar karena market kebetulan searah bisa
lolos status "LAYAK DIPERCAYA" tanpa skill nyata -- pola bug yang sama persis
dengan yang ditemukan di walk-forward backtest, tapi kali ini di jalur live
production.

**Perbaikan**: `get_model_trading_leaderboard()` sekarang menghitung
`baseline_majority_accuracy_pct` (akurasi tebak-mayoritas dari arah AKTUAL
yang terjadi pada window evaluasi yang sama) dan `edge_vs_baseline_pct`,
konsisten secara metodologi dengan `walk_forward.py`. `get_model_trust_audit()`
sekarang mensyaratkan `edge_vs_baseline_pct >= 3.0pp` (ambang sama dengan
`run_analysis.py`) sebelum status "LAYAK DIPERCAYA" diberikan. Kolom lama yang
menyesatkan (`walk_forward_score`) dihapus, diganti kolom yang namanya sesuai
isinya.

**Dibuktikan lewat test regresi baru**
(`test_model_trust_audit_flags_illusory_accuracy_without_baseline_edge`):
skenario akurasi 100% tapi tanpa edge nyata sekarang benar dikategorikan
"JANGAN DIIKUTI", bukan "LAYAK DIPERCAYA" seperti sebelumnya.

**Belum dikerjakan**: `decision_support.py` sendiri tidak diubah karena
memang tidak terpakai -- mengubahnya tidak akan berdampak ke pengguna. Kalau
ke depan mau dipakai (mis. untuk position sizing otomatis di UI), perlu
disambungkan dulu ke `streamlit_app.py`/`run_analysis.py`, dan idealnya dibuat
membaca `edge_vs_baseline_pct` yang sama seperti trust audit, bukan skor
heuristik terpisah.

## Validasi Akhir

- `py_compile` bersih untuk semua file yang diubah.
- Test suite penuh: **67 test lulus**, termasuk 3 test baru (2 di sentiment
  holdout evaluation, 1 di trust-audit illusory-accuracy).
- Smoke test runtime (`streamlit run` + HTTP request): HTTP 200, tidak ada
  exception, untuk perubahan UI trust-audit maupun sentiment holdout.

## Yang Belum Dikerjakan / Kandidat Lanjutan (belum diminta secara eksplisit)

1. Screening genuine edge pada seluruh ~270 ticker di `config/stocks.yaml`
   (baru 24 yang diuji).
2. Validasi walk-forward penuh untuk horizon H+5/H+10 (baru korelasi
   univariat, belum walk-forward tervalidasi seperti H+1).
3. Backtest hipotesis "kontrarian Indonesia" pada sentiment terhadap return
   aktual.
4. Vendor dataset sentiment praktikum secara lokal (saat ini bergantung pada
   folder `../tugas_nlp_ai_trading/` di luar proyek ini).

## Daftar File yang Diubah (sesi ini)

- `src/data_pipeline/auto_updater.py`
- `scripts/daily_global_workflow_cli.py`, `scripts/update_prices_cli.py`
- `src/models/walk_forward.py`
- `src/models/direction_classifier.py`, `src/models/global_models.py`
- `run_analysis.py`
- `src/nlp/sentiment_analyzer.py`
- `src/utils/accuracy_tracker.py`
- `streamlit_app.py`
- `tests/test_prediction_upgrades.py`, `tests/test_sentiment_analyzer.py`,
  `tests/test_accuracy_tracker.py`
