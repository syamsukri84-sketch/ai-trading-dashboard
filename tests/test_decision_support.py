from src.trading.decision_support import (
    build_decision_support,
    calculate_ai_confidence_score,
    calculate_position_sizing,
)


def test_confidence_score_returns_label_and_components():
    result = calculate_ai_confidence_score(
        projected_return_pct=5.0,
        anomaly_score=80.0,
        p_value=0.01,
        regime="CALM",
        rsi=35.0,
        volatility_pct=2.0,
        var_95_pct=3.0,
    )

    assert result["ai_confidence_score"] > 55
    assert result["confidence_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert "risk_penalty" in result["components"]


def test_position_sizing_uses_lot_size():
    sizing = calculate_position_sizing(
        capital=100_000_000,
        entry_price=3000,
        stop_loss=2900,
        risk_pct=1.0,
    )

    assert sizing["risk_amount"] == 1_000_000
    assert sizing["lots"] == 100
    assert sizing["shares"] == 10_000


def test_build_decision_support_ready_when_trade_and_sized():
    signal = {
        "action": "TRADE",
        "setup": {
            "entry": 3000,
            "stop_loss": 2900,
        },
    }

    result = build_decision_support(
        ticker="BBRI",
        current_price=3000,
        projected_return_pct=5.0,
        anomaly_score=85,
        p_value=0.01,
        regime="CALM",
        rsi=35,
        volatility_pct=2.0,
        var_95_pct=3.0,
        signal_result=signal,
        capital=100_000_000,
        risk_pct=1.0,
    )

    assert result["readiness"] == "READY"
    assert result["position_sizing"]["lots"] > 0
