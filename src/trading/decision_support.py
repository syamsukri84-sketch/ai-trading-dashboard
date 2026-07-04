from typing import Any, Dict


def calculate_ai_confidence_score(
    projected_return_pct: float,
    anomaly_score: float,
    p_value: float,
    regime: str,
    rsi: float,
    volatility_pct: float,
    var_95_pct: float,
) -> Dict[str, Any]:
    """
    Menggabungkan sinyal model menjadi skor keputusan 0-100.
    Skor tinggi berarti peluang lebih menarik dengan risiko yang masih masuk akal.
    """
    projected_return_component = min(abs(projected_return_pct) / 8.0, 1.0) * 25.0
    anomaly_component = min(max(anomaly_score, 0.0), 100.0) * 0.25
    confidence_component = min(max(1.0 - p_value, 0.0), 1.0) * 20.0

    regime_upper = regime.upper()
    regime_component = {
        "CALM": 15.0,
        "VOLATILE": 10.0,
        "CRASH": 4.0,
    }.get(regime_upper, 8.0)

    momentum_component = 0.0
    if projected_return_pct > 0 and rsi < 45:
        momentum_component = 10.0
    elif projected_return_pct < 0 and rsi > 55:
        momentum_component = 10.0
    elif 45 <= rsi <= 55:
        momentum_component = 4.0

    risk_penalty = min(max(volatility_pct, 0.0) / 8.0, 1.0) * 8.0
    risk_penalty += min(max(var_95_pct, 0.0) / 10.0, 1.0) * 7.0

    score = projected_return_component + anomaly_component + confidence_component + regime_component + momentum_component - risk_penalty
    score = max(0.0, min(100.0, score))

    if score >= 75:
        label = "HIGH"
    elif score >= 55:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "ai_confidence_score": float(score),
        "confidence_label": label,
        "components": {
            "projected_return": float(projected_return_component),
            "anomaly": float(anomaly_component),
            "statistical_confidence": float(confidence_component),
            "regime": float(regime_component),
            "momentum": float(momentum_component),
            "risk_penalty": float(risk_penalty),
        },
    }


def calculate_position_sizing(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float = 1.0,
    lot_size: int = 100,
) -> Dict[str, Any]:
    """Menghitung jumlah lot berdasarkan risiko maksimum per trade."""
    risk_amount = max(capital, 0.0) * max(risk_pct, 0.0) / 100.0
    risk_per_share = abs(entry_price - stop_loss)

    if entry_price <= 0 or risk_per_share <= 0 or lot_size <= 0:
        return {
            "risk_amount": float(risk_amount),
            "risk_per_share": float(risk_per_share),
            "shares": 0,
            "lots": 0,
            "position_value": 0.0,
            "capital_used_pct": 0.0,
        }

    raw_shares = int(risk_amount // risk_per_share)
    lots = raw_shares // lot_size
    shares = lots * lot_size
    position_value = shares * entry_price
    capital_used_pct = (position_value / capital) * 100.0 if capital > 0 else 0.0

    return {
        "risk_amount": float(risk_amount),
        "risk_per_share": float(risk_per_share),
        "shares": int(shares),
        "lots": int(lots),
        "position_value": float(position_value),
        "capital_used_pct": float(capital_used_pct),
    }


def build_decision_support(
    ticker: str,
    current_price: float,
    projected_return_pct: float,
    anomaly_score: float,
    p_value: float,
    regime: str,
    rsi: float,
    volatility_pct: float,
    var_95_pct: float,
    signal_result: Dict[str, Any],
    capital: float,
    risk_pct: float,
) -> Dict[str, Any]:
    confidence = calculate_ai_confidence_score(
        projected_return_pct=projected_return_pct,
        anomaly_score=anomaly_score,
        p_value=p_value,
        regime=regime,
        rsi=rsi,
        volatility_pct=volatility_pct,
        var_95_pct=var_95_pct,
    )

    sizing = {}
    readiness = "SKIP"
    reasons = []

    if signal_result.get("action") == "TRADE":
        setup = signal_result["setup"]
        sizing = calculate_position_sizing(
            capital=capital,
            entry_price=setup["entry"],
            stop_loss=setup["stop_loss"],
            risk_pct=risk_pct,
        )
        if confidence["ai_confidence_score"] >= 55 and sizing["lots"] > 0:
            readiness = "READY"
        else:
            readiness = "WATCHLIST"
            if confidence["ai_confidence_score"] < 55:
                reasons.append("AI confidence belum cukup kuat.")
            if sizing["lots"] <= 0:
                reasons.append("Modal/risk per trade belum cukup untuk minimal 1 lot.")
    else:
        reasons.append(signal_result.get("reason", "Tidak ada sinyal trade."))

    return {
        "ticker": ticker,
        "current_price": float(current_price),
        "readiness": readiness,
        "reasons": reasons,
        **confidence,
        "position_sizing": sizing,
    }


def build_trade_gate(
    projected_return_pct: float,
    confidence_pct: float,
    volatility_pct: float,
    min_confidence_pct: float = 60.0,
    min_projected_return_pct: float = 1.0,
    max_volatility_pct: float = 8.0,
) -> Dict[str, Any]:
    """Gate sederhana agar tidak semua prediksi otomatis menjadi kandidat trading."""
    checks = {
        "confidence_ok": confidence_pct >= min_confidence_pct,
        "return_buffer_ok": abs(projected_return_pct) >= min_projected_return_pct,
        "volatility_ok": volatility_pct <= max_volatility_pct,
    }
    passed = all(checks.values())
    reasons = []
    if not checks["confidence_ok"]:
        reasons.append(f"Confidence {confidence_pct:.1f}% < batas {min_confidence_pct:.1f}%.")
    if not checks["return_buffer_ok"]:
        reasons.append(f"Projected return {projected_return_pct:+.2f}% belum melewati buffer {min_projected_return_pct:.2f}%.")
    if not checks["volatility_ok"]:
        reasons.append(f"Volatilitas {volatility_pct:.2f}% > batas {max_volatility_pct:.2f}%.")

    return {
        "passed": passed,
        "status": "LAYAK DIPERTIMBANGKAN" if passed else "TUNGGU / SKIP",
        "checks": checks,
        "reasons": reasons,
    }
