from src.nlp.news_fetcher import extract_article_text, fetch_google_news_sentiment_items


class FakeRSSResponse:
    content = b"""
    <rss>
      <channel>
        <item>
          <title>BBRI mencatat laba meningkat signifikan</title>
          <source>Contoh Berita</source>
          <pubDate>Mon, 22 Jun 2026 09:00:00 GMT</pubDate>
          <link>https://example.com/bbri</link>
        </item>
        <item>
          <title>BBRI menghadapi tekanan pasar</title>
          <source>Contoh Market</source>
          <pubDate>Mon, 22 Jun 2026 10:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    def raise_for_status(self):
        return None


class FakeArticleResponse:
    text = """
    <html>
      <head><script>ignore me</script></head>
      <body>
        <article>
          <h1>BBRI tumbuh</h1>
          <p>Manajemen menyatakan pertumbuhan kredit tetap kuat.</p>
          <p>Laba naik dan risiko kredit stabil.</p>
        </article>
      </body>
    </html>
    """

    def raise_for_status(self):
        return None


def test_fetch_google_news_sentiment_items_parses_rss(monkeypatch):
    calls = {}

    def fake_get(url, timeout, headers):
        calls["url"] = url
        calls["timeout"] = timeout
        calls["headers"] = headers
        return FakeRSSResponse()

    monkeypatch.setattr("src.nlp.news_fetcher.requests.get", fake_get)

    rows = fetch_google_news_sentiment_items("bbri", query="BBRI saham", limit=1, include_article_body=False)

    assert len(rows) == 1
    assert rows[0] == {
        "date": "2026-06-22",
        "ticker": "BBRI",
        "source": "Contoh Berita",
        "text": "BBRI mencatat laba meningkat signifikan",
    }
    assert "BBRI+saham" in calls["url"]
    assert calls["timeout"] == 15
    assert "User-Agent" in calls["headers"]


def test_fetch_google_news_sentiment_items_includes_article_body(monkeypatch):
    requested_urls = []

    def fake_get(url, timeout, headers):
        requested_urls.append(url)
        if url == "https://example.com/bbri":
            return FakeArticleResponse()
        return FakeRSSResponse()

    monkeypatch.setattr("src.nlp.news_fetcher.requests.get", fake_get)

    rows = fetch_google_news_sentiment_items("BBRI", query="BBRI saham", limit=1)

    assert len(rows) == 1
    assert "BBRI mencatat laba meningkat signifikan" in rows[0]["text"]
    assert "pertumbuhan kredit tetap kuat" in rows[0]["text"]
    assert "risiko kredit stabil" in rows[0]["text"]
    assert requested_urls == [
        "https://news.google.com/rss/search?q=BBRI+saham&hl=id&gl=ID&ceid=ID:id",
        "https://example.com/bbri",
    ]


def test_extract_article_text_ignores_script_and_style():
    text = extract_article_text("""
    <html>
      <style>.hidden { color: red; }</style>
      <script>alert('x')</script>
      <p>Harga saham naik kuat.</p>
    </html>
    """)

    assert text == "Harga saham naik kuat."
