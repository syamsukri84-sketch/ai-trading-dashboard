"""Test untuk src/nlp/gemini_keyword_search.py -- SEMUA test di sini pakai
mock, TIDAK pernah memanggil Gemini API sungguhan (butuh API key + biaya
kuota nyata, tidak cocok untuk test suite otomatis)."""

from unittest.mock import MagicMock, patch

import pytest

from src.nlp.gemini_keyword_search import (
    GeminiConfig,
    _is_quota_or_grounding_error,
    _parse_suggestions,
    check_gemini_status,
    get_gemini_config,
    suggest_sentiment_keywords,
)


# ---- get_gemini_config / check_gemini_status ----

def test_get_gemini_config_reads_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "real-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    config = get_gemini_config()
    assert config.api_key == "real-key-123"
    assert config.model == "gemini-test-model"
    assert config.enabled is True


def test_get_gemini_config_treats_placeholder_as_empty(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ganti_dengan_api_key_anda")
    config = get_gemini_config()
    assert config.api_key == ""
    assert config.enabled is False


def test_get_gemini_config_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = get_gemini_config()
    assert config.enabled is False


def test_check_gemini_status_disabled_without_key():
    status = check_gemini_status(GeminiConfig(api_key=""))
    assert status["enabled"] is False
    assert status["ok"] is False


def test_check_gemini_status_enabled_with_key():
    status = check_gemini_status(GeminiConfig(api_key="fake-key"))
    assert status["enabled"] is True
    assert status["ok"] is True
    assert "gemini" in status["model"].lower()


# ---- _parse_suggestions ----

def test_parse_suggestions_extracts_well_formed_lines():
    text = (
        "TICKER: BBCA | QUERY: BBCA saham kredit | ALASAN: Pertumbuhan kredit kuat.\n"
        "TICKER: BBRI | QUERY: BBRI saham NPL | ALASAN: Kekhawatiran NPL naik.\n"
    )
    result = _parse_suggestions(text, {"BBCA", "BBRI", "TLKM"})
    assert len(result) == 2
    assert result[0] == {"ticker": "BBCA", "query": "BBCA saham kredit", "reason": "Pertumbuhan kredit kuat."}


def test_parse_suggestions_ignores_ticker_outside_universe():
    text = "TICKER: AAPL | QUERY: apple saham | ALASAN: Bukan saham Indonesia.\n"
    result = _parse_suggestions(text, {"BBCA", "BBRI"})
    assert result == []


def test_parse_suggestions_dedups_repeated_ticker():
    text = (
        "TICKER: BBCA | QUERY: query pertama | ALASAN: alasan pertama.\n"
        "TICKER: BBCA | QUERY: query kedua | ALASAN: alasan kedua.\n"
    )
    result = _parse_suggestions(text, {"BBCA"})
    assert len(result) == 1
    assert result[0]["query"] == "query pertama"


def test_parse_suggestions_ignores_malformed_lines():
    text = "Ini bukan format yang diminta sama sekali.\nTICKER tanpa separator yang benar"
    result = _parse_suggestions(text, {"BBCA"})
    assert result == []


def test_parse_suggestions_empty_text():
    assert _parse_suggestions("", {"BBCA"}) == []


# ---- _is_quota_or_grounding_error ----

def test_is_quota_error_detects_code_429():
    exc = Exception("some error")
    exc.code = 429
    assert _is_quota_or_grounding_error(exc) is True


def test_is_quota_error_detects_resource_exhausted_message():
    assert _is_quota_or_grounding_error(Exception("429 RESOURCE_EXHAUSTED. quota exceeded")) is True


def test_is_quota_error_false_for_unrelated_error():
    exc = Exception("500 Internal Server Error")
    exc.code = 500
    assert _is_quota_or_grounding_error(exc) is False


# ---- suggest_sentiment_keywords (mocked genai.Client) ----

def _mock_response(text: str):
    response = MagicMock()
    response.text = text
    return response


def test_suggest_sentiment_keywords_disabled_without_key():
    result = suggest_sentiment_keywords(["BBCA"], config=GeminiConfig(api_key=""))
    assert result["error"] == "GEMINI_API_KEY belum diset di file .env."
    assert result["suggestions"] == []
    assert result["grounded"] is False


def test_suggest_sentiment_keywords_no_tickers():
    result = suggest_sentiment_keywords([], config=GeminiConfig(api_key="fake-key"))
    assert result["error"] == "Tidak ada ticker untuk dicari."


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_grounded_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(
        "TICKER: BBCA | QUERY: BBCA saham kredit kuat | ALASAN: Laba meningkat signifikan."
    )
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA", "BBRI"], top_n=5, config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is None
    assert result["warning"] is None
    assert result["grounded"] is True
    assert len(result["suggestions"]) == 1
    assert result["suggestions"][0]["ticker"] == "BBCA"
    # Grounding tool harus benar-benar diminta (google_search) di panggilan pertama.
    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["config"].tools is not None


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_falls_back_when_grounding_quota_exhausted(mock_client_cls):
    quota_error = Exception("429 RESOURCE_EXHAUSTED. quota exceeded")
    quota_error.code = 429

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        quota_error,  # panggilan pertama (dengan grounding) gagal karena kuota
        _mock_response("TICKER: BBRI | QUERY: BBRI saham NPL | ALASAN: NPL naik."),  # fallback tanpa grounding
    ]
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA", "BBRI"], top_n=5, config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is None
    assert result["grounded"] is False
    assert result["warning"] is not None and "grounding habis" in result["warning"]
    assert len(result["suggestions"]) == 1
    assert result["suggestions"][0]["ticker"] == "BBRI"
    assert mock_client.models.generate_content.call_count == 2


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_non_quota_error_does_not_fallback(mock_client_cls):
    other_error = Exception("500 Internal Server Error")
    other_error.code = 500
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = other_error
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA"], config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is not None
    assert "Panggilan Gemini API gagal" in result["error"]
    assert result["suggestions"] == []
    # Tidak ada percobaan kedua -- error non-kuota harus langsung gagal.
    assert mock_client.models.generate_content.call_count == 1


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_both_calls_fail_after_quota_fallback(mock_client_cls):
    quota_error = Exception("429 RESOURCE_EXHAUSTED.")
    quota_error.code = 429
    fallback_error = Exception("503 Service Unavailable")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [quota_error, fallback_error]
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA"], config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is not None
    assert "fallback tanpa grounding" in result["error"]
    assert result["suggestions"] == []


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_empty_response_gives_clear_message(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response("")
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA"], config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is not None
    assert "respons kosong" in result["error"]
    assert result["suggestions"] == []


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_unparseable_response(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response("Teks bebas yang tidak sesuai format sama sekali.")
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA"], config=GeminiConfig(api_key="fake-key"))

    assert result["error"] is not None
    assert "tidak bisa diparsing" in result["error"]
    assert result["raw_response"]


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_respects_top_n(mock_client_cls):
    text = "\n".join(
        f"TICKER: {t} | QUERY: query {t} | ALASAN: alasan {t}."
        for t in ["BBCA", "BBRI", "TLKM", "ASII"]
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(text)
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(
        ["BBCA", "BBRI", "TLKM", "ASII"], top_n=2, config=GeminiConfig(api_key="fake-key")
    )

    assert len(result["suggestions"]) == 2


@patch("google.genai.Client")
def test_suggest_sentiment_keywords_strips_jk_suffix_from_tickers(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_response(
        "TICKER: BBCA | QUERY: q | ALASAN: a."
    )
    mock_client_cls.return_value = mock_client

    result = suggest_sentiment_keywords(["BBCA.JK"], config=GeminiConfig(api_key="fake-key"))
    assert result["suggestions"][0]["ticker"] == "BBCA"
