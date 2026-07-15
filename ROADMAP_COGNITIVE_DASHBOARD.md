# Roadmap: Menutup Gap "Cognitive Dashboard" (Umpan Balik Pengguna + Adaptasi Pasar)

Catatan rencana kerja -- **belum dikerjakan**, disimpan supaya bisa dilanjutkan
kapan saja. Ditulis 2026-07-12, menindaklanjuti diagnosis jujur: dashboard saat
ini TIDAK punya loop belajar dari umpan balik pengguna sama sekali, dan adaptasi
ke konteks pasar masih berbasis aturan terjadwal, bukan pembelajaran kognitif
sesungguhnya.

## Prinsip desain yang WAJIB dipegang (baca ini dulu sebelum mengerjakan apa pun)

**Umpan balik pengguna TIDAK BOLEH dicampur ke model prediksi arah/harga.**

Model prediksi (`DirectionClassifier`, `PriceProjector`, dst.) divalidasi lewat
walk-forward vs baseline naif -- itu ground truth objektif (harga aktual).
"Suka/tidak suka" pengguna terhadap sebuah rekomendasi BUKAN indikator apakah
rekomendasi itu benar secara statistik -- pengguna bisa saja tidak suka sinyal
yang sebenarnya profitable, atau suka sinyal yang ternyata rugi. Kalau umpan
balik subjektif ini dicampur ke bobot model prediksi, seluruh kerja keras
validasi walk-forward & genuine-edge screening (lihat
`ROADMAP_PRESCRIPTIVE_ANALYTICS.md` dan sesi optimasi Juli 2026) jadi rusak.

**Aturan**: umpan balik pengguna hanya boleh memengaruhi **lapisan
personalisasi/ranking/presentasi** (ticker mana yang ditonjolkan, ambang
personal, watchlist personal) -- TIDAK PERNAH mengubah angka
`edge_vs_baseline_pct`, hyperparameter model, atau status genuine-edge itu
sendiri. Dua lapisan ini harus tetap terpisah secara arsitektur.

## Bagian A: Loop Umpan Balik Pengguna (Reinforcement dari Interaksi)

### A1. Tangkap umpan balik eksplisit -- **SELESAI (2026-07-12)**
- [x] Tombol ✅ Saya Ikuti / ⏭ Saya Lewati / 👍 Berguna / 👎 Tidak Berguna
      ditambahkan di expander "🔍 Detail Per Saham" (tab Ringkasan Harian) --
      BUKAN di tiap baris tabel utama (Streamlit tidak praktis untuk tombol
      per-baris di dataframe besar), tapi di panel detail ticker yang sudah
      ada dari Fase E. Fungsional setara: tetap per-ticker, per-rekomendasi.
- [x] `data/tracking/user_feedback_log.csv` -- kolom: timestamp, ticker,
      signal_shown, action (IKUTI/LEWATI/BERGUNA/TIDAK_BERGUNA), note.
- [x] Modul baru `src/utils/user_feedback.py`: `log_user_feedback()`,
      `load_user_feedback()`, `get_feedback_summary_by_ticker()` (yang
      terakhir ini disiapkan untuk A3, belum dipakai di UI).
- [x] Riwayat feedback per ticker ditampilkan langsung di expander yang sama.
- [x] Test: `tests/test_user_feedback.py` (4 test, semua lulus). Sempat
      menemukan bug late-binding default-argument Python (fungsi
      `load_user_feedback()` dipanggil tanpa argumen di dalam
      `log_user_feedback` memakai default parameter yang di-bind saat modul
      di-import, bukan saat dipanggil -- menyebabkan tiap panggilan menimpa
      log alih-alih menambah). Diperbaiki dengan memanggil
      `load_user_feedback(USER_FEEDBACK_FILE)` eksplisit.
- Effort aktual: kecil-sedang, sesuai perkiraan.

### A2. Tangkap umpan balik implisit (opsional, lebih rumit)
- [ ] Lacak ticker mana yang benar-benar dibuka detailnya (expander "🔍 Detail
      Per Saham" yang sudah ada) via `st.session_state` -- proxy minat.
- [ ] Streamlit tidak native melacak dwell-time/klik detail; perlu
      instrumentasi manual per komponen, bukan analytics otomatis.
- Effort: sedang. Manfaat lebih kecil dari A1, prioritas lebih rendah.

### A3. Lapisan personalisasi (BUKAN model prediksi) -- **SELESAI (2026-07-12)**
- [x] Modul baru `src/trading/personalization.py`: baca
      `user_feedback_log.csv` lewat `get_feedback_summary_by_ticker`, hitung
      `compute_personal_scores()` per ticker (rentang -1.0 s.d. 1.0, 0.0 kalau
      belum ada feedback).
- [x] `data/user_profile.json`: `load_user_profile`/`save_user_profile`/
      `mute_ticker`/`unmute_ticker` -- simpan `muted_tickers` (list) dan
      `personal_risk_tolerance` (placeholder, belum dipakai di UI).
- [x] `apply_personalization(board_df)` menambah kolom "Skor Personal" dan
      "Dimute" ke papan keputusan TANPA mengubah kolom Sinyal/Confidence/edge
      yang sudah ada (diverifikasi via test bahwa df asli tidak termutasi).
- [x] UI: kolom "Skor Personal"/"Dimute" muncul di mode Trader/Audit (bukan
      Pemula, supaya tetap sederhana di sana). Checkbox "Sembunyikan ticker
      yang saya mute" (default aktif) di atas tabel Rencana Trading. Tombol
      🔇 Mute / 🔊 Un-mute per ticker di expander "🔍 Detail Per Saham".
- [x] Test: `tests/test_personalization.py` (5 test, semua lulus) -- termasuk
      memverifikasi df asli tidak termutasi dan skor 0.0 untuk ticker tanpa
      feedback.
- [x] Menghindari bug late-binding yang sama seperti di A1 (fungsi modul lain
      dipanggil dengan argumen eksplisit, bukan mengandalkan default
      parameter yang di-bind saat import).
- Effort aktual: sedang, sesuai perkiraan.

### A4. (Lanjutan, effort besar) Bandit untuk variasi presentasi
- [ ] Kalau mau genuinely "belajar dari interaksi": pakai multi-armed bandit
      (mis. Thompson Sampling) untuk memilih variasi tampilan/ambang mana yang
      paling sering diikuti pengguna, lalu adaptif condong ke situ.
- [ ] Ini komponen reinforcement learning yang sesungguhnya, tapi scope-nya
      TETAP di lapisan presentasi (threshold mana yang ditampilkan duluan),
      bukan mengubah model prediksi itu sendiri.
- Effort: besar. Butuh volume interaksi cukup banyak supaya bandit-nya
  bermakna (bukan sekadar noise dari sample kecil).

## Bagian B: Adaptasi ke Konteks Pasar yang Berubah

Yang SUDAH ada (jangan dibangun ulang): deteksi regime CALM/VOLATILE/CRASH
(`decision_support.py`), market breadth REBOUND/BEARISH/MIXED
(`build_daily_decision_board`), kebijakan retrain otomatis saat akurasi
menurun (`evaluate_training_policy`). Ini semua heuristik berbasis aturan,
bukan deteksi statistik formal atau pembelajaran berkelanjutan.

### B1. Deteksi concept drift yang formal
- [ ] Tambah metrik drift eksplisit: bandingkan `edge_vs_baseline_pct` rolling
      (mis. 20 hari terakhir) vs baseline historisnya sendiri -- kalau turun
      signifikan, flag "model mulai usang", terpisah dari sekadar
      `direction_accuracy_pct < threshold` yang sudah ada.
- [ ] Pertimbangkan algoritma drift standar (ADWIN/DDM) kalau mau lebih
      rigorous daripada perbandingan rolling sederhana.
- [ ] Simpan riwayatnya di `data/drift_monitoring/` supaya bisa dilihat trennya
      dari waktu ke waktu, bukan cuma snapshot saat ini.
- Effort: sedang.

### B2. Model per-regime (bukan satu model untuk semua kondisi)
- [ ] Latih model TERPISAH untuk tiap regime (CALM vs VOLATILE vs CRASH),
      bukan satu model dengan regime cuma sebagai fitur/filter heuristik.
- [ ] Perlu cukup data historis per regime untuk masing-masing dilatih --
      regime CRASH biasanya jarang & datanya sedikit, jadi mungkin tidak
      semua ticker bisa dapat model CRASH-spesifik yang reliable.
- Effort: besar. Reset ulang bagian training pipeline.

### B3. Riwayat regime (memori konteks pasar) -- **SELESAI (2026-07-12)**
- [x] Modul baru `src/trading/market_regime.py`: `compute_market_breadth()`
      (logika sama seperti `latest_market_breadth` lama yang ada di dalam
      `build_daily_decision_board` -- disatukan di sini, bukan diduplikasi).
- [x] `log_regime_snapshot()` menyimpan ke `data/regime_history.csv`, dedup
      per tanggal kalender (aman dipanggil berkali-kali di hari yang sama).
- [x] `summarize_regime_streaks()`: menghitung streak regime saat ini +
      rata-rata durasi historis per jenis regime.
- [x] Wired ke DUA tempat: `build_daily_decision_board` (streamlit_app.py, jadi
      tercatat tiap dashboard dibuka) DAN `scripts/daily_global_workflow_cli.py`
      (supaya tercatat juga lewat otomatisasi Task Scheduler walau dashboard
      tidak dibuka).
- [x] UI: caption "📅 Konteks regime pasar: sudah N hari di regime X,
      historisnya rata-rata bertahan ~M hari" di tab Ringkasan Harian.
- [x] Test: `tests/test_market_regime.py` (7 test, semua lulus). Diverifikasi
      juga dengan data proyek asli (bukan cuma data sintetis test).
- Effort aktual: kecil-sedang, sesuai perkiraan.

## Kebutuhan Data

**Sudah tersedia**: log prediksi & akurasi, status genuine-edge, fitur
volatilitas/breadth untuk regime detection, DAN sejak A1/A3/B3 selesai:
**log umpan balik pengguna** (`data/tracking/user_feedback_log.csv`),
**profil pengguna** (`data/user_profile.json`), **riwayat regime harian**
(`data/regime_history.csv`).

**Perlu ditambahkan** (untuk B1/A4/B2 yang belum dikerjakan):
- [ ] **Baseline statistik untuk drift** -- rolling distribusi
      `edge_vs_baseline_pct` per ticker dari waktu ke waktu, perlu dihitung &
      disimpan sistematis (bisa diturunkan dari data yang sudah ada, tapi
      belum diagregasi untuk tujuan ini).
- [ ] **(Asumsi perlu dikonfirmasi) Status single-user vs multi-user** --
      dashboard ini saat ini tidak punya sistem akun/login. Kalau
      personalisasi memang hanya untuk Anda sendiri, tidak perlu identitas
      pengguna formal. Kalau ke depan dipakai banyak orang, ini perlu
      dibangun terpisah (login, isolasi data per user) -- scope besar yang
      belum termasuk roadmap ini.

## Gambaran Cara Kerja Sistem Nanti (Alur)

```
[Model prediksi & trust audit yang SUDAH ADA -- TIDAK BERUBAH]
        |
        v
[BARU] Lapisan Personalisasi (src/trading/personalization.py)
   - baca user_feedback_log.csv + user_profile.json
   - hitung skor personal per ticker (murni untuk URUTAN tampilan)
   - TIDAK PERNAH mengubah angka prediksi/edge/confidence
        |
        v
[Papan Keputusan Harian -- SUDAH ADA, ditambah:]
   - urutan/highlight mengikuti skor personal
   - tombol umpan balik 👍/👎 / "Saya Ikuti" / "Saya Lewati" di tiap baris
        |
        v
[BARU] Feedback Logger -- setiap klik tombol ditulis ke user_feedback_log.csv
        |
        v (loop mingguan/berkala)
[BARU] Job Personalisasi -- hitung ulang skor personal dari akumulasi feedback
        |
        (kembali ke Lapisan Personalisasi)

--- Jalur terpisah: adaptasi pasar (tidak berpotongan dengan feedback pengguna) ---

[Retrain policy & regime detection -- SUDAH ADA]
        |
        v
[BARU] Drift Detector -- pantau tren edge_vs_baseline_pct, flag kalau menurun signifikan
        |
        v
[BARU] Regime History Log -- catat regime harian, agregasi durasi/pola
        |
        v
[Retrain policy -- SUDAH ADA, sekarang juga dipicu oleh flag drift, bukan cuma akurasi mentah]
```

Dua jalur ini (personalisasi pengguna vs adaptasi pasar) SENGAJA dipisah dan
tidak saling memengaruhi -- keduanya "belajar", tapi dari sumber kebenaran
yang berbeda (preferensi subjektif vs harga aktual), dan mencampur keduanya
adalah kesalahan desain yang harus dihindari.

## Urutan Prioritas yang Disarankan

1. ~~**A1** (tangkap feedback eksplisit)~~ -- **SELESAI**.
2. ~~**B3** (riwayat regime)~~ -- **SELESAI**.
3. ~~**A3** (lapisan personalisasi dasar)~~ -- **SELESAI**.
4. **B1** (drift detection formal) -- sekarang giliran ini; `data/regime_history.csv`
   sudah mulai terisi (dari B3) tapi masih perlu waktu untuk punya cukup riwayat
   sebelum perbandingan rolling benar-benar bermakna (bukan cuma 1-2 titik data).
5. **A4 dan B2** -- effort besar, kerjakan belakangan setelah fondasi di atas
   terbukti berguna dan cukup data feedback/regime terkumpul.

**Catatan volume data**: karena B1 butuh rolling window regime/edge yang
lumayan panjang, dan A4 (bandit) butuh volume interaksi feedback yang tidak
sedikit supaya tidak cuma menangkap noise -- jangan buru-buru mengerjakan
keduanya sebelum `data/regime_history.csv` dan `data/tracking/user_feedback_log.csv`
punya cukup baris (kumpulkan dulu lewat pemakaian harian/mingguan).
