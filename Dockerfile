# ==============================================================================
# TAHAP 1: BUILDER (Membangun dan Mengompilasi Dependensi)
# ==============================================================================
FROM python:3.11-slim as builder

# Install system dependencies untuk kompilasi (misal C/C++ compiler untuk TA-Lib)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Buat Virtual Environment di dalam container
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Salin dan install dependencies Python ke dalam Virtual Environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# TAHAP 2: RUNNER (Image Final Produksi)
# ==============================================================================
FROM python:3.11-slim

# Konfigurasi Environment (Jangan tulis file .pyc, arahkan stdout ke terminal)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# HANYA salin Virtual Environment yang sudah ter-install dari tahap builder
COPY --from=builder /opt/venv /opt/venv

# Salin seluruh kode sumber aplikasi
COPY . .

# Amankan aplikasi dengan menjalankan container sebagai user non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Ekspos port FastAPI (Opsional untuk dokumentasi, binding dilakukan docker-compose)
EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]