# PERINGATAN: API ini adalah jalur sinyal EKSPERIMENTAL yang terpisah dari
# dashboard Streamlit (streamlit_app.py) yang sebenarnya dipakai untuk keputusan
# trading. Pipeline di routes.py (isolation forest + conformal predictor +
# regime classifier) belum pernah divalidasi walk-forward vs baseline naif --
# TIDAK sama seperti DirectionClassifier/trust-audit di jalur produksi. Jangan
# perlakukan "confidence"/"p_value" dari API ini sebagai sinyal yang sudah
# terbukti punya edge. streamlit_app.py TIDAK memanggil API ini sama sekali.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from routes import router as api_router
from websocket import router as ws_router
from src.cockpit.api import router as cockpit_router
from src.database.database import engine, Base

# Buat tabel database secara otomatis (jika belum ada) saat aplikasi startup
Base.metadata.create_all(bind=engine)

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="AI Trading Decision Support API",
    description="REST API & WebSocket untuk deteksi anomali saham LQ45",
    version="1.0.0"
)

# Konfigurasi CORS agar bisa diakses oleh Frontend (Streamlit/React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Dalam produksi, ganti dengan domain frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Daftarkan routes (Endpoins)
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")
app.include_router(cockpit_router)

# Otomatis arahkan ke halaman dokumentasi saat membuka alamat utama
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/cockpit")
