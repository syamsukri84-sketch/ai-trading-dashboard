# Catatan Sesi — VaR Multi-Metodologi & Perbaikan Gating BUY

**Tanggal**: 2026-07-17
**Tujuan file ini**: rangkuman kerja satu sesi chat (Cowork) supaya bisa dilanjutkan
langsung di VSCode tanpa perlu re-derive konteks dari awal. Baca `STATUS_PROYEK_AI_TRADING.md`
dulu kalau belum familiar dengan arsitektur proyek ini secara umum — file ini cuma
mencatat apa yang berubah/ditemukan di SESI INI.

---

## 1. Konteks awal sesi (di luar proyek ini, tapi jadi sumber temuan)

Sesi dimulai dari permintaan lain (analisis fundamental & risiko saham LPPF/MDS
Retailing Tbk dari laporan keuangan PDF yang diunggah user, di luar proyek AI
Trading ini) — di situ dibangun skill Cowork terpisah (`analisis-saham-emiten`,
tersimpan di scratchpad Cowork, bukan di proyek ini) dan dilakukan analisis VaR
5 metodologi (historical, parametric normal, Cornish-Fisher, EWMA, Monte Carlo)
terhadap data harga LPPF.

**Temuan metodologis penting dari sesi itu** (dasar kerja di proyek ini):
- Di confidence 95%, Cornish-Fisher konsisten dengan metode lain.
- Di confidence 99%, suku koreksi kurtosis pada ekspansi Cornish-Fisher
  (`(z³-3z)·K/24`) membesar **secara kubik** terhadap z. Begitu kurtosis riil
  (bukan artefak sampel kecil) berada di kisaran moderat-tinggi, suku ini bisa
  melebih-lebihkan VaR jauh di atas historical simulation / Monte Carlo
  bootstrap. Diverifikasi empiris dengan data **asli** `data/raw/LPPF_raw.csv`
  (252 hari trailing s.d. 16 Jul 2026): CF 99% = 4.16%, sementara
  historical/parametric/MC-bootstrap semua di 3.19–3.28%.
- **Rekomendasi**: pakai Cornish-Fisher di confidence <99%, pakai rata-rata
  historical+MC-bootstrap di confidence ≥99% (jangan pakai CF sebagai estimasi
  utama di confidence sedalam itu).

User lalu meminta temuan ini diterapkan ke proyek AI Trading yang sesungguhnya
(`D:\PROYEK_PYTHON\08_AI TRADING`) — itulah pekerjaan yang tercatat di bawah.

---

## 2. Perubahan kode di sesi ini (file per file)

### 2.1 `src/models/var_analysis.py` — **BARU**
Modul fungsi murni (bukan class, tidak butuh training seperti `GARCHModel`).
Isi: `compute_var_suite(returns, confidence_levels, window, horizon_days, ...)`
dan wrapper `compute_var_from_price_df(df, ...)`.

Menghitung 5 metode (historical/parametric/Cornish-Fisher/EWMA/MC-bootstrap)
lalu **otomatis memilih metode direkomendasikan per confidence level**:
- `< 0.99` → `cornish_fisher`
- `>= 0.99` → rata-rata `historical` + `mc_bootstrap` (field
  `CF_UNSTABLE_CONF_THRESHOLD = 0.99` di puncak file kalau mau diubah)

Default window trailing **252 hari** (`DEFAULT_WINDOW`), floor minimum 10
observasi (di bawah itu return `{"error": ...}`), flag `data_terbatas=True`
kalau n < 60 (`MIN_OBS_RELIABLE`).

Docstring modul ini berisi penjelasan matematis lengkap (kenapa CF tidak stabil
di confidence dalam) — baca langsung di file kalau perlu detail.

### 2.2 `run_analysis.py` — diedit, **aditif saja**
- Baris ~12: `from src.models.var_analysis import compute_var_from_price_df`
- Baris ~706-714 (setelah `garch_projection = garch_model.predict(...)`):
  panggil `var_suite = compute_var_from_price_df(df, confidence_levels=(0.95, 0.99), horizon_days=1)`
- Baris ~825-832: tambahan print `[VaR 95%]`/`[VaR 99%]` (setelah print GARCH lama)
- Baris ~905-910: 6 field baru ditambahkan ke dict `summary["analyzed"]`:
  `var95_recommended_pct`, `var95_recommended_method`, `var99_recommended_pct`,
  `var99_recommended_method`, `var_n_obs`, `var_data_terbatas`

**Field lama `garch_volatility_pct`/`garch_var95_pct` TIDAK dihapus** — murni
penambahan, tidak ada breaking change ke konsumen dict ini.

### 2.3 `tests/test_var_analysis.py` — **BARU**
10 test. Salah satunya (`test_compute_var_from_lppf_raw_matches_manually_verified_values`)
adalah **regresi terhadap data asli** `data/raw/LPPF_raw.csv` (bukan data
sintetis) — memverifikasi ulang angka yang sudah dicek manual di sesi
sebelumnya (skew~0.013, kurtosis~2.80, CF 99% jadi outlier vs historical).

**Sudah diverifikasi lulus semua (10/10)** — tapi dijalankan di sandbox Linux
terpisah dengan `numpy`/`scipy`/`pandas`/`pytest` yang di-pip-install manual,
BUKAN lewat `.venv` Windows proyek ini (sandbox tidak bisa eksekusi venv
Windows). **Belum pernah dijalankan lewat `.venv\Scripts\python.exe -m pytest`
yang sesungguhnya** — lihat bagian 4.

### 2.4 `streamlit_app.py` — diedit, **1 blok kode di `build_daily_decision_board`**

**Temuan bug** (bukan cuma penambahan fitur, ini perbaikan pelanggaran aturan
proyek sendiri): sinyal 🟢 BUY (dengan entry/stop-loss/lot **konkret**) bisa
tampil untuk ticker APA PUN, tanpa syarat status "✅ Terverifikasi Ganda",
karena:
- Parameter `genuine_edge_lookup` (dipakai gate "TIDAK ADA EDGE (WALK-FORWARD)")
  cuma terisi kalau user mencentang checkbox opt-in "Hanya proses ticker dengan
  edge nyata" di tab Ringkasan Harian — **default checkbox itu OFF**
  (`value=False`, baris ~2514-2518).
- Ini melanggar Prinsip Desain #4 (`STATUS_PROYEK_AI_TRADING.md` bagian 6):
  "Prescriptive analytics HANYA untuk ticker Terverifikasi Ganda" — karena
  gate itu jadi OPT-IN, bukan berlaku selalu.

**Perbaikan** (di percabangan BUY, sebelumnya sekitar baris ~1605): BUY
sekarang wajib `unified_badge == "✅ Terverifikasi Ganda"` (variabel ini SELALU
terisi benar, tidak tergantung checkbox, karena `edge_lookup_for_badge=full_edge_lookup`
selalu dioper apa adanya di baris ~2531). Kalau syarat teknis BUY terpenuhi
tapi belum Terverifikasi Ganda → turun ke `WATCH` dengan alasan eksplisit di
kolom "Alasan Utama" (bukan actionable, tapi tetap tampil sebagai referensi
pemantauan).

`py_compile` bersih untuk file ini. **AppTest belum dijalankan** (butuh
dependency berat — streamlit, lightgbm, xgboost, arch — yang tidak ada di
sandbox saya) — WAJIB dijalankan sebelum dipercaya, lihat bagian 4.

---

## 3. Temuan lain yang BELUM ditindaklanjuti (sengaja ditunda)

### 3.1 `volatility_lookup` di `build_daily_decision_board` tidak pernah diisi
Parameter ini (dipakai untuk menghitung stop-loss "berbasis volatilitas GARCH")
**selalu `None`** di titik pemanggilan (`decision_board_df = build_daily_decision_board(...)`
baris ~2523, parameter ini tidak dioper sama sekali) — akibatnya SEMUA ticker
sebenarnya memakai stop-loss default flat 3%, BUKAN volatilitas GARCH seperti
klaim komentar kode (baris ~1556-1568 di `streamlit_app.py`).

Supaya `var95_recommended_pct` yang baru bisa dipakai di sini, dibutuhkan
**lapisan penyimpanan baru** — VaR yang dihitung `run_analysis.py` saat ini
cuma hidup di dict `summary["analyzed"]` yang dikembalikan ke caller, TIDAK
pernah ditulis ke CSV/JSON tracking mana pun yang bisa dibaca ulang saat
dashboard di-render. Opsi yang belum diputuskan (butuh keputusan desain,
bukan cuma coding):
- (a) Tambah kolom VaR ke CSV tracking yang sudah ada (mis. lewat
  `log_prediction`/`accuracy_log.csv`), atau
- (b) Bikin file tracking baru khusus VaR per ticker per tanggal (ingat:
  **wajib pakai `atomic_write_csv`** dari `src/utils/atomic_io.py`, sesuai
  Prinsip Desain #5 proyek ini — jangan `.to_csv()` langsung).

### 3.2 Tiga implementasi "jarak stop-loss" yang tidak saling terhubung
Ditemukan saat investigasi di atas, belum direkonsiliasi:
1. **ATR-based** (`src/trading/signal_generator.py::generate_signal`,
   `stop_loss = current_price - 2*atr`) — sudah benar tapi **orphan**, tidak
   pernah dipanggil dari `streamlit_app.py`.
2. **GARCH-volatility-heuristic** (`streamlit_app.py` baris ~1556-1568) — yang
   SEHARUSNYA jalan, tapi lihat 3.1 (input-nya selalu kosong jadi selalu fallback
   ke default 3%).
3. **VaR** (baru, `src/models/var_analysis.py`) — sudah ada nilainya tapi belum
   dipakai di manapun untuk stop-loss (lihat 3.1).

Rekomendasi: konsolidasikan jadi satu sumber kebenaran sebelum menambah
metodologi lain di atasnya — jangan tambah implementasi ke-4.

### 3.3 `src/trading/decision_support.py`: `calculate_ai_confidence_score`/
`build_decision_support`/`build_trade_gate` **tidak pernah dipanggil** di
`streamlit_app.py`. Cuma `calculate_position_sizing` yang dipakai. Kalau mau
memakai skor kepercayaan gabungan yang lebih matang (bukan cuma heuristik
if-else `action` di `build_daily_decision_board`), fungsi-fungsi ini sudah
ada tapi perlu di-wire.

### 3.4 Roadmap prescriptive analytics lebih luas
Lihat `ROADMAP_PRESCRIPTIVE_ANALYTICS.md` — semua item (#1-#8) masih
"belum dikerjakan". Item #1 (position sizing Kelly/risk-parity berbasis edge
riil) adalah kelanjutan paling alami dari kerja sesi ini kalau mau lanjut.
**Ingat gate wajib**: cuma 3-6 dari 270 ticker saat ini lolos status
"Terverifikasi Ganda" (`data/edge_screening_status.json`) — jadi fitur
prescriptive apa pun akan terlihat "kosong" untuk mayoritas ticker, itu
BENAR secara desain, bukan bug.

---

## 4. WAJIB dijalankan sebelum lanjut/percaya perubahan sesi ini

Sandbox Cowork saya **tidak bisa** menjalankan ini (venv Windows + dependency
berat seperti `arch`/`lightgbm`/`xgboost`/`streamlit` tidak ada di sandbox
Linux saya) — jalankan sendiri di VSCode/PowerShell dari root proyek:

```powershell
.venv\Scripts\python.exe -m py_compile streamlit_app.py run_analysis.py
.venv\Scripts\python.exe -m pytest -q
```
Harusnya **180 lulus** (170 baseline per `STATUS_PROYEK_AI_TRADING.md` 2026-07-14
+ 10 test baru `test_var_analysis.py`). Kalau beda, cek dulu apakah baseline
170 itu sendiri masih akurat (banyak perubahan lain yang sudah uncommitted
sebelum sesi ini dimulai — jalankan `git status`/`git diff` untuk pisahkan
perubahan sesi ini dari perubahan lain yang sudah ada).

Lalu smoke test tanpa browser (fokuskan ke tab "Ringkasan Harian" karena itu
yang diedit):
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('streamlit_app.py', default_timeout=180)
at.run()
print(len(at.exception))  # harus 0
```

**Belum di-commit ke git sama sekali** — working tree sudah punya banyak
perubahan uncommitted lain sebelum sesi ini (lihat `git status`), jadi pisahkan
commit kode sesi ini dari perubahan lain, jangan digabung satu commit besar
(sesuai catatan git di `CLAUDE.md` bagian 7).

---

## 5. Kalau mau lanjut sesi berikutnya

Rujuk langsung ke bagian 3 di atas (3.1/3.2/3.3/3.4) — pilih salah satu, tidak
perlu jelaskan ulang konteks, cukup bilang "lanjutkan item 3.1 dari
CATATAN_SESI_VAR_DAN_GATING_2026-07-17.md".
