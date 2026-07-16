"""Pencarian kata kunci sentimen pasar saham Indonesia via Gemini API resmi
(bukan chat UI gemini.google.com -- itu sesi interaktif pribadi yang butuh
login, tidak bisa dipanggil programatik dari pipeline otomatis).

Dipakai SEBELUM mengisi field "Ticker untuk pencarian berita" / "Query
pencarian" di tab Sentimen Pasar -- AI mencari (via Google Search grounding
di Gemini API) ticker mana dari universe config/stocks.yaml yang paling
relevan untuk analisis sentimen pasar Indonesia SAAT INI, lalu menyarankan
query pencarian berita untuk tiap ticker itu. Hasilnya dipakai untuk
mengisi otomatis field Ticker/Query yang sudah ada -- bukan pengganti alur
pengambilan berita yang sudah ada, cuma langkah persiapan sebelumnya.

Butuh GEMINI_API_KEY (dari https://aistudio.google.com/apikey) di file
`.env`. Kalau tidak diset, fitur ini nonaktif dan seluruh fitur lain
dashboard tetap berjalan normal (pola sama seperti src/utils/mongo_store.py
untuk MongoDB -- integrasi eksternal opsional, degradasi anggun).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "gemini-flash-latest"


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str = DEFAULT_MODEL
    use_grounding: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())


def get_gemini_config() -> GeminiConfig:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    placeholder_tokens = ["ganti_dengan_api_key", "YOUR_API_KEY", "<api_key>"]
    if any(token.lower() in api_key.lower() for token in placeholder_tokens):
        api_key = ""
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    use_grounding_raw = os.getenv("AI_TRADING_GEMINI_USE_GROUNDING", "true").strip().lower()
    use_grounding = use_grounding_raw not in {"0", "false", "no", "off", "tidak"}
    return GeminiConfig(api_key=api_key, model=model, use_grounding=use_grounding)


def check_gemini_status(config: GeminiConfig | None = None) -> dict[str, Any]:
    config = config or get_gemini_config()
    if not config.enabled:
        return {
            "enabled": False,
            "ok": False,
            "model": config.model,
            "message": "GEMINI_API_KEY belum diset di file .env.",
        }
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return {
            "enabled": True,
            "ok": False,
            "model": config.model,
            "message": "Package google-genai belum terinstal. Jalankan: pip install google-genai",
        }
    return {
        "enabled": True,
        "ok": True,
        "model": config.model,
        "message": (
            f"Gemini API aktif (model: {config.model}; "
            f"Google Search grounding: {'aktif' if config.use_grounding else 'nonaktif'})."
        ),
    }


# Pola parsing baris respons Gemini, format yang diminta di _build_prompt():
# "TICKER: XXX | QUERY: ... | ALASAN: ..." -- regex sengaja longgar (spasi
# opsional, case-insensitive) karena output model bahasa tidak 100% konsisten
# formatnya meski sudah diberi instruksi ketat.
_SUGGESTION_LINE_RE = re.compile(
    r"TICKER\s*:\s*([A-Z0-9]{2,6})\s*\|\s*QUERY\s*:\s*(.+?)\s*\|\s*ALASAN\s*:\s*(.+)",
    re.IGNORECASE,
)


def _build_prompt(tickers: list[str], top_n: int) -> str:
    ticker_list = ", ".join(tickers)
    return (
        "Kamu membantu riset analisis sentimen pasar saham Indonesia (IDX). "
        f"Berikut universe ticker yang tersedia untuk dianalisis ({len(tickers)} ticker): {ticker_list}.\n\n"
        "Gunakan pencarian web untuk mengecek berita/perbincangan pasar saham Indonesia TERKINI "
        f"(hari ini), lalu pilih {top_n} ticker dari daftar di atas SAJA (jangan sarankan ticker "
        "di luar daftar) yang paling relevan untuk dianalisis sentimennya sekarang -- misalnya "
        "karena ada berita signifikan, pergerakan harga tidak biasa, atau sedang banyak dibicarakan.\n\n"
        "Jawab HANYA dalam format berikut, satu baris per ticker, TANPA teks lain di luar format ini:\n"
        "TICKER: <kode_ticker> | QUERY: <query pencarian berita yang bagus untuk ticker ini, "
        "dalam Bahasa Indonesia, mis. 'BBRI saham kredit kuartal'> | ALASAN: <alasan singkat 1 kalimat>"
    )


def _parse_suggestions(text: str, valid_tickers: set[str]) -> list[dict[str, str]]:
    suggestions = []
    seen_tickers = set()
    for line in text.splitlines():
        match = _SUGGESTION_LINE_RE.search(line)
        if not match:
            continue
        ticker = match.group(1).strip().upper()
        if ticker not in valid_tickers or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        suggestions.append({
            "ticker": ticker,
            "query": match.group(2).strip(),
            "reason": match.group(3).strip(),
        })
    return suggestions


def _is_quota_or_grounding_error(exc: Exception) -> bool:
    """Deteksi error kuota (429 RESOURCE_EXHAUSTED) -- di tier gratis, kuota
    untuk Google Search grounding SERING JAUH lebih ketat daripada kuota
    generate_content biasa (dikonfirmasi langsung 2026-07-14: generate_content
    polos berhasil, tapi dengan grounding tool kena 429 dalam beberapa kali
    panggilan). Dicek via atribut `.code` (SDK google-genai) dulu, fallback ke
    pencarian substring di pesan error kalau atributnya tidak ada -- SDK bisa
    berubah struktur exception-nya di versi mendatang."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    message = str(exc)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def suggest_sentiment_keywords(tickers: list[str], top_n: int = 5, config: GeminiConfig | None = None) -> dict[str, Any]:
    """Minta Gemini menyarankan ticker + kata kunci pencarian berita paling
    relevan untuk analisis sentimen hari ini, dibatasi HANYA ke `tickers`
    yang diberikan (tidak menyarankan ticker di luar itu).

    Coba dulu DENGAN Google Search grounding (supaya benar-benar "mencari",
    bukan menebak dari data training yang bisa basi). Kalau kuota grounding
    habis (429 -- umum terjadi di tier gratis, dikonfirmasi langsung saat
    membangun fitur ini), otomatis MUNDUR ke generate_content polos (tanpa
    pencarian live) -- hasilnya tetap dikembalikan, tapi field "grounded"
    akan False dan `warning` diisi peringatan supaya user tahu ini BUKAN
    hasil pencarian langsung, cuma pengetahuan model yang mungkin sudah tidak
    terkini. TIDAK BOLEH gagal total kalau kuota habis -- lebih baik kasih
    hasil dengan peringatan jujur daripada dashboard mati fungsi.

    Return dict SELALU punya key: "error" (str kalau GAGAL TOTAL/tidak ada
    saran sama sekali, None kalau ada hasil), "warning" (str caveat non-fatal,
    mis. fallback tanpa grounding, None kalau tidak ada), "grounded" (bool),
    "suggestions" (list, bisa kosong), dan "raw_response" (teks mentah dari
    model, untuk debug manual kalau parsing gagal/tidak lengkap).
    """
    config = config or get_gemini_config()
    empty = {"error": None, "warning": None, "suggestions": [], "raw_response": "", "grounded": False}
    if not config.enabled:
        return {**empty, "error": "GEMINI_API_KEY belum diset di file .env."}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {**empty, "error": "Package google-genai belum terinstal. Jalankan: pip install google-genai"}

    clean_tickers = [str(t).replace(".JK", "").upper().strip() for t in tickers if str(t).strip()]
    if not clean_tickers:
        return {**empty, "error": "Tidak ada ticker untuk dicari."}

    client = genai.Client(api_key=config.api_key)
    prompt = _build_prompt(clean_tickers, top_n)
    grounded = bool(config.use_grounding)
    warning = None
    try:
        if config.use_grounding:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        else:
            response = client.models.generate_content(model=config.model, contents=prompt)
        raw_text = response.text or ""
    except Exception as e:
        if not config.use_grounding or not _is_quota_or_grounding_error(e):
            return {**empty, "error": f"Panggilan Gemini API gagal: {e}"}
        # Kuota Google Search grounding habis -- mundur ke generate_content
        # polos supaya fitur tetap berfungsi, dengan peringatan jujur (bukan
        # error fatal -- hasil di bawah tetap dikembalikan kalau berhasil).
        grounded = False
        warning = (
            "Kuota Google Search grounding habis -- hasil di bawah berdasarkan "
            "pengetahuan model (BUKAN pencarian berita langsung), jadi bisa jadi "
            "tidak mencerminkan situasi pasar paling terkini. Coba lagi nanti "
            "atau cek kuota di https://ai.dev/rate-limit."
        )
        try:
            response = client.models.generate_content(model=config.model, contents=prompt)
            raw_text = response.text or ""
        except Exception as e2:
            return {**empty, "error": f"Panggilan Gemini API gagal (termasuk fallback tanpa grounding): {e2}"}

    suggestions = _parse_suggestions(raw_text, set(clean_tickers))
    if not suggestions:
        if not raw_text.strip():
            # Respons kosong (finish_reason bisa STOP tanpa Part berisi teks)
            # ditemukan sesekali terjadi di tier gratis saat kuota sedang
            # tertekan -- beda dari "format salah", ini transient, coba lagi
            # biasanya berhasil. Dibedakan pesannya supaya user tahu ini
            # bukan bug format prompt.
            empty_error = "Gemini mengembalikan respons kosong (kemungkinan kuota API sedang tertekan) -- coba klik lagi beberapa saat lagi."
            return {
                "error": empty_error,
                "warning": warning,
                "suggestions": [],
                "raw_response": raw_text,
                "grounded": grounded,
            }
        return {
            "error": "Respons Gemini tidak bisa diparsing jadi saran ticker (lihat raw_response untuk detail).",
            "warning": warning,
            "suggestions": [],
            "raw_response": raw_text,
            "grounded": grounded,
        }

    return {
        "error": None,
        "warning": warning,
        "suggestions": suggestions[:top_n],
        "raw_response": raw_text,
        "grounded": grounded,
    }
