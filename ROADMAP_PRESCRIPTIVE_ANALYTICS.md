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

## Cara memakai ulang dokumen ini

Kalau mau lanjut kerjakan salah satu item, cukup rujuk nomornya (mis. "kerjakan
#1 dan #3 dari roadmap prescriptive analytics") -- konteks penuh sudah ada di
sini, tidak perlu dijelaskan ulang dari awal.
