from src.utils.mongo_store import check_mongo_status, get_mongo_config


def test_mongo_config_disabled_without_uri(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)

    config = get_mongo_config()
    status = check_mongo_status(config)

    assert not config.enabled
    assert status["enabled"] is False
    assert status["ok"] is False
    assert status["database"] == "ai_trading"
