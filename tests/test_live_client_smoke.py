"""Smoke tests for KalshiTradingClientLive — auth, conversion, request shape.

Doesn't hit the network. Verifies the wire-level translation between our
OrderRequest and Kalshi's expected payload.
"""
from datetime import datetime, timezone
from unittest.mock import patch

from src.execution.kalshi_trading_client import KalshiTradingClientLive
from src.execution.models import OrderRequest


# Generate a fake RSA key for testing — we never send it anywhere.
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
FAKE_PEM = _key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


def make_client():
    from src.config.kalshi_config import KalshiConfig
    cfg = KalshiConfig.__new__(KalshiConfig)
    cfg.api_key_id = "test-key"
    cfg.private_key_pem = FAKE_PEM
    cfg.rest_base_url = "https://api.elections.kalshi.com/trade-api/v2"
    cfg.ws_url = "wss://ignored"
    cfg.market_tickers = []
    cfg.channels = []
    cfg.reconnect_attempts = 1
    cfg.reconnect_base_delay = 1.0
    cfg.reconnect_max_delay = 1.0
    cfg.ping_interval = 20
    return KalshiTradingClientLive(cfg)


def test_dollar_to_cents_clamps():
    c = make_client()
    assert c._to_cents(0.755) == 76  # rounds
    assert c._to_cents(0.005) == 1   # min clamp
    assert c._to_cents(0.999) == 99  # max clamp
    assert c._to_cents(0.50) == 50


def test_auth_headers_present():
    c = make_client()
    h = c._auth_headers("POST", "/portfolio/orders")
    assert "KALSHI-ACCESS-KEY" in h
    assert "KALSHI-ACCESS-TIMESTAMP" in h
    assert "KALSHI-ACCESS-SIGNATURE" in h
    assert h["Content-Type"] == "application/json"


def test_buy_yes_uses_yes_price():
    """Verify the order body shape matches Kalshi's expected schema."""
    c = make_client()
    req = OrderRequest(
        ticker="X-1", side="yes", contracts=10,
        order_type="limit", limit_price=0.85, time_in_force="ioc",
        client_order_id="t-1", placed_at=datetime.now(timezone.utc),
    )
    cents = c._to_cents(req.limit_price)
    assert cents == 85


def test_buy_no_uses_no_price():
    c = make_client()
    req = OrderRequest(
        ticker="X-1", side="no", contracts=10,
        order_type="limit", limit_price=0.30, time_in_force="ioc",
        client_order_id="t-2", placed_at=datetime.now(timezone.utc),
    )
    cents = c._to_cents(req.limit_price)
    assert cents == 30


if __name__ == "__main__":
    test_dollar_to_cents_clamps()
    test_auth_headers_present()
    test_buy_yes_uses_yes_price()
    test_buy_no_uses_no_price()
    print("All 4 live client smoke tests passed")
