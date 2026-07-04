from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
import re
from typing import Dict, List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 AI-Trading-Dashboard/1.0"}


class ArticleTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._capture_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "iframe"}:
            self._skip_depth += 1
        if tag in {"p", "article", "h1", "h2", "li"}:
            self._capture_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg", "iframe"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "article", "h1", "h2", "li"} and self._capture_depth:
            self._capture_depth -= 1
            self._chunks.append(" ")

    def handle_data(self, data):
        if self._skip_depth or not self._capture_depth:
            return
        clean = data.strip()
        if clean:
            self._chunks.append(clean)

    def get_text(self) -> str:
        return _normalize_text(" ".join(self._chunks))


def _parse_pub_date(value: str) -> str:
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(str(value))).strip()


def extract_article_text(html: str, max_chars: int = 4000) -> str:
    parser = ArticleTextParser()
    parser.feed(html)
    text = parser.get_text()
    return text[:max_chars].strip()


def fetch_article_text(url: str, timeout: int = 15, max_chars: int = 4000) -> str:
    response = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    return extract_article_text(response.text, max_chars=max_chars)


def fetch_google_news_sentiment_items(
    ticker: str,
    query: str | None = None,
    limit: int = 10,
    timeout: int = 15,
    include_article_body: bool = True,
    max_article_chars: int = 4000,
) -> List[Dict[str, str]]:
    """
    Fetches recent news from Google News RSS for sentiment analysis.
    When possible, it follows the article link and analyzes title plus article text.
    Returns rows compatible with data/sentiment/market_issues.csv.
    """
    clean_ticker = ticker.upper().strip()
    search_query = query.strip() if query else f"{clean_ticker} saham"
    url = (
        f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(search_query)}"
        "&hl=id&gl=ID&ceid=ID:id"
    )

    response = requests.get(
        url,
        timeout=timeout,
        headers=DEFAULT_HEADERS,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows: List[Dict[str, str]] = []

    for item in root.findall("./channel/item")[:limit]:
        title = item.findtext("title", default="").strip()
        source = item.findtext("source", default="Google News").strip() or "Google News"
        pub_date = _parse_pub_date(item.findtext("pubDate", default=""))
        link = item.findtext("link", default="").strip()

        if not title:
            continue

        article_text = ""
        if include_article_body and link:
            try:
                article_text = fetch_article_text(link, timeout=timeout, max_chars=max_article_chars)
            except Exception:
                article_text = ""

        text = _normalize_text(f"{title}. {article_text}") if article_text else _normalize_text(title)

        rows.append({
            "date": pub_date,
            "ticker": clean_ticker,
            "source": source,
            "text": text,
        })

    return rows
