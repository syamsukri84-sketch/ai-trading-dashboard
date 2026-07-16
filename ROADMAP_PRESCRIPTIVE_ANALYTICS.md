# Roadmap: Prescriptive Analytics untuk AI Trading Dashboard

Catatan rencana kerja -- **belum dikerjakan**, disimpan supaya bisa dilanjutkan
kapan saja tanpa mengulang diskusi dari awal. Ditulis 2026-07-12, setelah 4 fase
optimasi model + perapian dashboard (XAI, genuine-edge screening) selesai.

## Konteks: posisi dashboard saat ini di tangga analitik

| Tingkat | Contoh yang SUDAH ada | Status |
|---|---|---|
| Descriptive ("apa yang terjadi") | Rekap akurasi harian, leaderboard, drill-down | Sudah ada |
| Diagnostic ("kenapa terjadi") | SHAP/XAI (`src/explainability/shap_explainer.py`), walk-forward vs baseline edge | Sudah ada |
| Predictive ("apa yang akan terjadi") | DirectionClassifier, PriceProjector, LSTM, GARCH | Sudah ada (inti sistem) |
| **Prescriptive** ("apa yang harus dilakukan, seberapa optimal") | -- | **Belum ada** |

Yang ada sekarang (`build_daily_decision_board`, `calculate_position_sizing` di
`src/trading/decision_support.py`) adalah **heuristik** (aturan if-else), bukan
**optimasi** (menghitung aksi/alokasi terbaik dari fungsi tujuan eksplisit
dengan kendala). Itu beda mendasar antara "rules-based decision support" dan
"prescriptive analytics" sesungguhnya.

## Peringatan penting sebelum mulai

Prescriptive analytics MENGOPTIMALKAN berdasarkan prediksi yang sudah ada.
Temuan sesi optimasi (Juli 2026): sebagian besar ticker H+1 **belum terbukti**
punya edge nyata vs baseline naif (lihat `data/edge_screening_status.json` dan
tab "Walk-Forward Genuine Edge"). Membangun optimizer canggih di atas sinyal
yang belum tervalidasi hanya menghasilkan alokasi modal yang percaya diri
terhadap noise -- kelihatan lebih ilmiah padahal dasarnya sama lemahnya.

**Aturan wajib**: batasi semua rekomendasi prescriptive HANYA untuk ticker
berstatus "✅ Terverifikasi Ganda" (badge dari `compute_unified_trust_badge`
di `streamlit_app.py`), bukan seluruh universe saham. Jangan longgarkan aturan
ini demi menambah jumlah rekomendasi yang bisa ditampilkan.

## Daftar pekerjaan (urutan prioritas)

### 1. Position sizing berbasis optimasi (bukan % tetap)
- [ ] Ganti/lengkapi `calculate_position_sizing` (saat ini risiko X% tetap per
      trade) dengan Kelly Criterion atau risk-parity yang memperhitungkan edge
      riil (`edge_vs_baseline_pct` dari walk-forward) dan volatilitas per ticker.
- [ ] Validasi lewat backtest: apakah sizing baru menghasilkan expected growth
      lebih baik dari fixed-percent, bukan cuma "kelihatan lebih canggih".
- Effort: sedang. Perluasan alami dari kode yang sudah ada.

### 2. Optimasi portofolio lintas-ticker
- [ ] Hitung matriks korelasi/kovariansi return ANTAR-TICKER (bukan cuma
      `feat_beta_60`/`feat_corr_60` yang saat ini hanya vs IHSG).
- [ ] Definisikan fungsi tujuan eksplisit (mis. maksimalkan expected Sharpe
      ratio portofolio, bukan return individual per ticker).
- [ ] Tambah kendala: modal maksimum, jumlah posisi maksimum, eksposur
      maksimum per sektor/likuiditas tier.
- [ ] Pertimbangkan solver optimasi (mis. `scipy.optimize`, `cvxpy`) untuk
      alokasi -- bukan aturan if-else manual.
- Effort: besar. Ini pekerjaan arsitektur baru, bukan ekstensi kode lama.

### 3. Backtest STRATEGI penuh (bukan cuma akurasi prediksi)
- [ ] Walk-forward yang ada sekarang menguji "apakah arah/prediksi benar",
      BUKAN "apakah mengikuti rekomendasi ini profit setelah biaya transaksi".
- [ ] Simulasikan entry/exit/sizing lengkap dengan asumsi fee broker & slippage
      riil IDX.
- [ ] Ukur metrik strategi: Sharpe, max drawdown, win rate, profit factor --
      bukan cuma direction_accuracy_pct.
- Effort: sedang-besar. Bergantung pada item #4 (data biaya transaksi).

### 4. Simulator "what-if" / skenario
- [ ] "Kalau saya entry sekarang vs tunggu N hari" berdasarkan interval
      prediksi (bukan cuma titik estimasi tunggal).
- [ ] Perlu confidence interval / distribusi prediksi, bukan cuma nilai
      tunggal -- mungkin perlu conformal prediction atau quantile regression
      di layer model (belum ada saat ini).
- Effort: besar. Bergantung pada peningkatan model untuk menghasilkan interval,
  bukan cuma titik.

### 5. Rebalancing-aware (opsional, effort besar)
- [ ] Sistem perlu tahu portofolio Anda SAAT INI (posisi terbuka, kas) supaya
      bisa menyarankan REBALANCING, bukan cuma sinyal beli baru.
- [ ] Perlu form/input manual atau integrasi broker untuk data posisi real.

## Kebutuhan data

**Sudah tersedia** (bisa langsung dipakai untuk item di atas):
- Harga/volume historis (`data/raw/*.csv`)
- Fitur teknikal per ticker (`FeatureEngineer`)
- Log akurasi & prediksi (`data/tracking/accuracy_log.csv`, `predictions_log.csv`)
- Status genuine-edge per ticker (`data/edge_screening_status.json`)
- Fitur beta/korelasi ticker-vs-IHSG (belum ticker-vs-ticker)

**Perlu ditambahkan**:
- [ ] **Skema biaya transaksi IDX**: fee broker, estimasi slippage. Tanpa ini,
      "profit" yang diprescribe cuma teori di atas kertas.
- [ ] **Data portofolio pengguna saat ini**: posisi terbuka, kas tersedia.
      Tanpa ini sistem tidak bisa menyarankan rebalancing, cuma sinyal baru.
- [ ] **Matriks korelasi/kovariansi antar-ticker**: dihitung dari data harga
      yang sudah ada, tapi belum dihitung lintas-ticker.
- [ ] **Parameter toleransi risiko pengguna**: maks drawdown yang bisa
      diterima, horizon investasi, maks % modal per posisi -- lebih rinci
      dari `risk_per_trade_pct` yang ada sekarang.
- [ ] **(Opsional) Data fundamental/sektor**: kalau prescriptive juga mau
      menghormati batasan syariah/sektor/kapitalisasi, bukan cuma sinyal
      teknikal murni.
- [ ] **(Opsional) Kalender aksi korporasi**: stock split, dividen, rights
      issue -- supaya tidak ada sinyal palsu di sekitar tanggal-tanggal itu.

## Item tambahan diselamatkan dari `GEMINI AI` (proyek dipensiunkan 2026-07-17)

`GEMINI AI` (folder terpisah di workspace) awalnya direncanakan sebagai
trading app kedua, tapi arsitektur intinya (Isolation Forest + LightGBM/
XGBoost untuk arah harga) duplikat dari yang sudah diuji dan tidak
menunjukkan edge di proyek ini -- jadi dipensiunkan sebelum sempat
diimplementasi (src/ masih 100% kosong di sana). Tapi spek proyek itu
(`GEMINI AI/Dokumen Rujukan Ketat.md`, sebenarnya file .docx berlabel
.md) punya blueprint SIAP PAKAI untuk 3 hal yang persis mengisi gap di
atas -- dikutip di sini supaya tidak perlu dirancang dari nol.

### 6. Fitur cross-sectional ranking (fondasi murah untuk item #2)
- [ ] Tambahkan fitur peringkat ANTAR-ticker pada tanggal yang sama
      (bukan per-ticker seperti fitur teknikal yang sudah ada):
      `rank_return_1d`, `rank_return_5d`, `rank_volume_zscore`,
      `rank_volatility_20`, `rank_momentum_20`. Dihitung dengan
      `groupby("timestamp")`, kontras dengan fitur teknikal biasa yang
      pakai `groupby("kode_saham")`.
- Sumber blueprint: `GEMINI AI/Dokumen Rujukan Ketat.md` bagian 11.
- Effort: kecil. Prasyarat murah/cepat sebelum menghitung matriks
  korelasi/kovariansi penuh antar-ticker yang diminta item #2 di atas.

### 7. Modul portofolio Top-K + Inverse Volatility Weighting (v1 sebelum solver penuh)
- [ ] Bangun versi awal position sizing/portfolio construction: pilih
      Top-K ticker (mis. K=10), bobot dihitung `raw_weight_i =
      1/volatility_i` lalu dinormalisasi (`weight_i = raw_weight_i /
      sum(raw_weight)`), batas bobot maksimal per saham (mis. 15%),
      filter anomaly score, rebalancing berkala (mis. mingguan).
- Sumber blueprint: `GEMINI AI/Dokumen Rujukan Ketat.md` bagian 21.
- **PENYESUAIAN WAJIB, bukan opsional**: spek asli memilih Top-K
  langsung dari `priority_score`/probabilitas mentah TANPA gate
  kepercayaan apa pun. Itu melanggar Prinsip Desain #4 proyek ini
  (lihat `STATUS_PROYEK_AI_TRADING.md` bagian 6): "Prescriptive
  analytics HANYA untuk ticker Terverifikasi Ganda". Saat
  mengimplementasikan modul ini, **filter trust gate
  (`compute_unified_trust_badge`) wajib diterapkan SEBELUM Top-K
  selection**, bukan dipakai apa adanya seperti spek aslinya -- kalau
  hasil filter itu kosong (lihat catatan gating di item #1), modul ini
  belum bisa jalan sampai ada ticker yang lolos.
- Effort: sedang. Bisa jadi v1 yang lebih murah sebelum solver
  `scipy.optimize`/`cvxpy` di item #2 (optimasi portofolio penuh).

### 8. Modul backtesting strategi lengkap (mengisi gap item #3 langsung)
- [ ] Backtest kronologis dengan biaya transaksi eksplisit, modal awal,
      rebalancing periodik, dibandingkan ke benchmark equal-weight DAN
      buy-and-hold sederhana (bukan cuma dibandingkan ke baseline arah
      seperti walk-forward yang sudah ada).
- [ ] Metrik: cumulative return, annualized return, annualized
      volatility, maximum drawdown, Sharpe ratio sederhana, win rate,
      turnover, jumlah transaksi.
- Sumber blueprint: `GEMINI AI/Dokumen Rujukan Ketat.md` bagian 22 & 35
  -- parameter awal siap pakai sebagai starting point (Top-K=10,
  rebalancing mingguan, biaya transaksi 0.15% per transaksi, modal awal
  Rp100.000.000, max weight 15%/saham, anomaly threshold 75) --
  sesuaikan ke biaya transaksi IDX riil sebelum dipercaya angkanya.
- Effort: sedang-besar (estimasi sama seperti item #3 di atas -- ini
  detail implementasi tambahan untuk item yang sama, bukan item
  terpisah dari segi cakupan kerja).

## Cara memakai ulang dokumen ini

Kalau mau lanjut kerjakan salah satu item, cukup rujuk nomornya (mis. "kerjakan
#1 dan #3 dari roadmap prescriptive analytics") -- konteks penuh sudah ada di
sini, tidak perlu dijelaskan ulang dari awal.
