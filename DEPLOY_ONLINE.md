# Deploy Online AI Trading Dashboard

Target app:

```powershell
D:\PROYEK_PYTHON\GEMINI AI1\streamlit_app.py
```

## Pilihan 0: Streamlit Community Cloud

Gunakan opsi ini jika ingin dashboard bisa diakses online tanpa laptop menyala.

Project ini sudah disiapkan untuk Streamlit Cloud dengan:

- `streamlit_app.py` sebagai main file.
- `requirements.txt` untuk dependensi Python.
- `runtime.txt` untuk Python 3.11.
- `.streamlit/config.toml` untuk konfigurasi Streamlit.
- `.gitignore` agar `.env`, `.venv`, `node_modules`, data raw, dan model lokal tidak ikut ter-upload.

Langkah deploy:

1. Upload project ini ke repository GitHub.
2. Buka `https://share.streamlit.io`.
3. Pilih repository AI Trading.
4. Isi main file path:

```text
streamlit_app.py
```

5. Buka menu **Advanced settings** atau **App settings > Secrets**.
6. Isi secrets minimal untuk mode offline penuh:

```toml
AI_TRADING_DASHBOARD_PASSWORD = "password_yang_kuat"
AI_TRADING_OFFLINE_ONLY = "true"
AI_TRADING_DUPLICATE_POLICY = "skip"
AI_TRADING_PREDICTION_RUN_TYPE = "FINAL"
```

Dengan konfigurasi ini, dashboard memakai data lokal yang sudah ikut repository dan tidak mencoba koneksi MongoDB dari sidebar.

Jika nanti ingin mengaktifkan sinkronisasi MongoDB Atlas, ubah `AI_TRADING_OFFLINE_ONLY` menjadi `"false"` dan tambahkan:

```toml
MONGODB_URI = "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DATABASE = "ai_trading"
```

7. Klik Deploy.

Catatan: jangan upload file `.env` ke GitHub. Gunakan Secrets Streamlit Cloud untuk password dan koneksi database.

## Pilihan 1: Akses Online Cepat dari PC Lokal

Gunakan opsi ini untuk demo atau akses pribadi sementara. Dashboard tetap berjalan di PC Anda.

### Sekali klik

Jalankan:

```text
BUKA_AI_TRADING_ONLINE_FREE.bat
```

File ini akan:

- membuka Streamlit di `http://localhost:8502`
- membuka Cloudflare Tunnel jika `cloudflared` tersedia
- menampilkan URL online pada jendela Cloudflare Tunnel

### Manual

1. Jalankan dashboard lokal:

```powershell
cd "D:\PROYEK_PYTHON\GEMINI AI1"
.\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8502 --server.address localhost
```

2. Publikasikan port `8502` memakai tunnel seperti Cloudflare Tunnel atau ngrok.

Contoh ngrok:

```powershell
ngrok http 8502
```

Contoh Cloudflare Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8502
```

3. Buka URL publik yang diberikan tool tunnel.

## Sinkronisasi MongoDB Atlas Free

MongoDB Atlas dipakai sebagai penyimpanan online opsional untuk:

- `predictions`
- `accuracy_logs`
- `sentiment_issues`
- `daily_workflows`
- `training_registry`

Training dan prediksi tetap bisa dilakukan di laptop. Setelah selesai, jalankan sinkronisasi:

```text
SINKRON_MONGODB_ATLAS.bat
```

Atau manual:

```powershell
cd "D:\PROYEK_PYTHON\GEMINI AI1"
.\.venv\Scripts\python.exe scripts\sync_mongodb_cli.py
```

Pastikan file `.env` berisi:

```text
MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=ai_trading
AI_TRADING_DASHBOARD_PASSWORD=password_yang_kuat
```

Jangan commit file `.env` ke GitHub.

## Pilihan 2: VPS + Docker Compose

Gunakan opsi ini untuk dashboard yang lebih stabil dan bisa online terus.

1. Salin folder proyek ke VPS.

2. Buat file `.env` dari contoh:

```bash
cp .env.example .env
```

3. Edit `.env` dan isi password:

```bash
AI_TRADING_DASHBOARD_PASSWORD=password_yang_kuat
AI_TRADING_DUPLICATE_POLICY=skip
AI_TRADING_PREDICTION_RUN_TYPE=FINAL
```

4. Jalankan:

```bash
docker compose up -d --build
```

5. Buka:

```text
http://IP_SERVER:8501
```

Untuk domain HTTPS, pasang reverse proxy seperti Nginx Proxy Manager, Caddy, atau Cloudflare Tunnel di depan port `8501`.

## Keamanan Minimal

- Set `AI_TRADING_DASHBOARD_PASSWORD` sebelum membuka dashboard ke internet.
- Jangan upload file CSV dari sumber tidak dipercaya.
- Jangan buka port `8000` FastAPI ke publik jika belum diberi autentikasi.
- Backup folder `data/` secara rutin karena berisi model, prediksi, akurasi, dan histori workflow.

## Catatan Data

Dashboard memakai data lokal di:

```text
data/raw
data/models
data/tracking
data/sentiment
```

Jika deploy ke server baru, pastikan folder `data/` ikut tersalin agar ranking, model, dan akurasi tetap tersedia.
