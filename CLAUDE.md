# CLAUDE.md — 08_AI TRADING

## 1. Apa proyek ini

Dashboard Streamlit untuk membantu keputusan trading saham Indonesia
(IDX), dipakai sendiri oleh pemilik (bukan produk multi-user).
Entry point sungguhan: `streamlit_app.py` (jalankan via
`BUKA_AI_TRADING.bat` atau `.venv\Scripts\python.exe -m streamlit run
streamlit_app.py`). Git repo, branch `main`, remote `origin` =
`ai-trading-dashboard.git`.

## 2. WAJIB dibaca dulu sebelum menyentuh kode

**Baca `STATUS_PROYEK_AI_TRADING.md` di root folder ini DULU** — itu
adalah doc "read this first" yang selalu diupdate pemilik proyek,
berisi arsitektur nyata, temuan riset penting, prinsip desain wajib,
dan status pekerjaan terkini.

**JANGAN jadikan acuan arsitektur** dua file berikut (usang, fase
perencanaan Juni 2026, sudah ditandai usang di STATUS doc sendiri):
- `SYSTEM_PROMPT_TRADING_APP.md`
- `QUICK_START_GUIDE.md`

Kalau STATUS doc dan file lain bertentangan, **STATUS doc yang menang**
(karena itu yang aktif dirawat).

## 3. Temuan riset inti (jangan diabaikan saat kerja model)

0 dari 265 ticker punya edge nyata di H+1/H+3/H+5 (walk-forward +
koreksi FDR, LSTM juga sudah divalidasi dan hasilnya sama). **Jangan
buru-buru menambah fitur/model baru untuk "mengejar akurasi"** — arah
lanjutan yang masuk akal adalah prescriptive analytics (position
sizing/portfolio), bukan mengejar akurasi arah. Lihat
`ROADMAP_PRESCRIPTIVE_ANALYTICS.md`.

## 4. Prinsip desain wajib (jangan dilanggar tanpa sadar user)

1. Umpan balik pengguna **tidak boleh** memengaruhi bobot model
   prediksi — hanya boleh memengaruhi personalisasi/ranking/presentasi.
2. `EDGE_THRESHOLD_PCT` + koreksi FDR adalah syarat **GANDA** untuk
   status "genuine edge" — jangan longgarkan salah satu.
3. Model yang divalidasi walk-forward **harus identik** dengan model
   yang dideploy (termasuk kalibrasi).
4. Prescriptive analytics **hanya** untuk ticker "✅ Terverifikasi
   Ganda".
5. File tracking CSV ditulis via `atomic_write_csv`
   (`src/utils/atomic_io.py`), bukan `.to_csv()` langsung.

## 5. Pipeline kedua yang TIDAK terhubung — jangan bingung

`fastapi_app.py` + `routes.py` + `src/database/` adalah pipeline
eksperimental terpisah (isolation forest + conformal predictor +
SQLite), **tidak pernah divalidasi walk-forward**, dan
`streamlit_app.py` **tidak pernah memanggilnya**. Jangan anggap ini
bagian dari sistem produksi.

## 6. Verifikasi wajib sebelum & sesudah perubahan kode

```
.venv\Scripts\python.exe -m py_compile streamlit_app.py run_analysis.py
.venv\Scripts\python.exe -m pytest -q          # 170 lulus per 2026-07-16
```
Plus smoke test tanpa browser:
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('streamlit_app.py', default_timeout=180)
at.run()
print(len(at.exception))  # harus 0
```

## 7. Catatan git (non-obvious, sering bikin gagal)

- Commit pesan **multi-baris wajib pakai `git commit -F <file>`** —
  di PowerShell 5.1, `-m @'...'@` yang isinya ada tanda kutip akan
  terpecah jadi pathspec dan gagal.
- `data/raw/*.csv` dkk memang **sengaja** di-track git (offline-first
  untuk Streamlit Cloud) — tapi pisahkan commit kode dari commit data,
  jangan digabung satu commit besar.
- Committer identity saat ini: name `unknown`, email
  `syamsukri84@email.com` (typo, harusnya `gmail.com`) — belum
  dibetulkan, jangan otomatis "perbaiki" tanpa konfirmasi user.
- Ada 2 remote: `origin` (repo khusus proyek ini, `ai-trading-dashboard.git`)
  dan `sains` (repo lebih umum `SAINS_DATA.git`, kemungkinan wadah
  tugas/proyek data science lain milik user). Sejak 2026-07-17, branch
  `main` lokal **track `origin/main`** (bukan lagi `sains/ai-trading-dashboard`
  seperti sebelumnya) — jadi `git push` polos sekarang ke `origin`.
  `sains` disimpan di branch **`ai-trading-dashboard`** di repo itu (bukan
  `main`), dan harus di-push manual/eksplisit kalau perlu disinkronkan:
  `git push sains main:ai-trading-dashboard`. Kedua remote terakhir
  disinkronkan penuh di commit `899b2d0` (2026-07-17) — cek ulang kalau
  sudah lama, jangan asumsikan masih sejajar.

## 8. Folder/file yang BUKAN bagian aktif codebase (abaikan)

Semua ini di-gitignore dan merupakan artifact tools/agent lain, bukan
kode proyek:
- `.venv-1/`, `.pytest_tmp*/`, `node_modules/`, `.agents/`,
  `.continue/`, `pdf-forge-exports/`, `*.out`/`*.err`/`*.log`
- Hanya `.venv` (tanpa suffix) yang merupakan environment Python aktif
  proyek ini.
- `.env` berisi secret asli (`GEMINI_API_KEY` dkk) — sudah gitignored,
  **jangan pernah** print isinya ke output atau commit.

## 9. Kalau user minta "lanjutkan pekerjaan"

Tanyakan dulu item mana yang dimaksud dari bagian "Apa yang BELUM
dikerjakan" di STATUS doc — jangan asumsikan urutan otomatis, karena
beberapa item (B1 drift detection, B2 model per-regime) sengaja
direkomendasikan **ditunda**, bukan next-in-line secara default.
