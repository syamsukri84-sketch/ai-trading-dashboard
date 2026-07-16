from fastapi.testclient import TestClient

from fastapi_app import app


def test_cockpit_page_loads():
    client = TestClient(app)
    response = client.get("/cockpit")

    assert response.status_code == 200
    assert "AI Trading Cockpit" in response.text


def test_cockpit_summary_shape():
    client = TestClient(app)
    response = client.get("/api/cockpit/summary?limit=3")

    assert response.status_code == 200
    payload = response.json()
    assert "generated_at" in payload
    assert "status" in payload
    assert "recommendations" in payload
    assert isinstance(payload["recommendations"], list)
    assert len(payload["recommendations"]) <= 3

