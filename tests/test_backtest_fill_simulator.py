from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.fill_simulator import FillSimulator
from src.backtest.models import Candle
from src.signals.models import CalibratedEdge


def _edge(side: str = "yes", ticker: str = "T") -> CalibratedEdge:
    return CalibratedEdge(
        ticker=ticker, title="t", side=side,
        estimated_fair_prob=0.4, current_implied_prob=0.3, edge_pp=0.10,
        confidence=0.7, kelly_fraction=0.05,
        thesis="", source_signal_model="test",
    )


def _snapshot(yes_bid: int = 30, yes_ask: int = 32):
    from src.insight.models import MarketSnapshot
    return MarketSnapshot(
        event="x", market="T", outcome="YES",
        quoted_price=(yes_bid + yes_ask) // 2,
        implied_probability=(yes_bid + yes_ask) / 200,
        yes_bid=yes_bid, yes_ask=yes_ask, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )


def test_unsupported_model_raises():
    with pytest.raises(ValueError):
        FillSimulator(model="blackmagic")


def test_midpoint_fill_yes_side():
    sim = FillSimulator(model="midpoint", slippage_per_contract=0.01, fee_per_contract=0.05)
    fill = sim.simulate(_edge("yes"), _snapshot(30, 32), contracts=10, t=datetime.now(timezone.utc))
    assert fill is not None
    # midpoint = 0.31 + slippage = 0.32
    assert abs(fill.price - 0.32) < 1e-6
    assert fill.contracts == 10
    assert fill.fee == 0.5  # 0.05 * 10
    assert fill.signal_model == "test"


def test_midpoint_fill_no_side_uses_complement():
    sim = FillSimulator(model="midpoint", slippage_per_contract=0.01, fee_per_contract=0.0)
    # yes mid = 0.31; no mid = 0.69
    fill = sim.simulate(_edge("no"), _snapshot(30, 32), contracts=10, t=datetime.now(timezone.utc))
    assert fill is not None
    assert abs(fill.price - 0.70) < 1e-6  # 0.69 + 0.01 slippage


def test_midpoint_fill_zero_contracts_returns_none():
    sim = FillSimulator(model="midpoint")
    assert sim.simulate(_edge(), _snapshot(), contracts=0, t=datetime.now(timezone.utc)) is None


def test_midpoint_fill_clamps_extreme_prices():
    sim = FillSimulator(model="midpoint", slippage_per_contract=0.0)
    # yes_bid=99, yes_ask=99 → mid 0.99 → fill clamped to 0.99
    fill = sim.simulate(_edge(), _snapshot(99, 99), contracts=1, t=datetime.now(timezone.utc))
    assert fill is not None
    assert fill.price <= 0.99
    assert fill.price >= 0.01


def test_trade_match_no_fill_when_no_recent_candles():
    sim = FillSimulator(model="trade_match")
    # No candles within ±5min → no fill
    fill = sim.simulate(_edge(), _snapshot(), contracts=10, t=datetime.now(timezone.utc), recent_candles=[])
    assert fill is None


def test_trade_match_fills_when_low_crosses_limit():
    sim = FillSimulator(model="trade_match", slippage_per_contract=0.0)
    t = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    candles = [Candle(ticker="T", timestamp=t, open=32, high=33, low=29, close=30, volume=100)]
    # yes_ask=32; low=29 crosses → fill at 29
    fill = sim.simulate(_edge("yes"), _snapshot(30, 32), contracts=5, t=t, recent_candles=candles)
    assert fill is not None
    assert abs(fill.price - 0.29) < 1e-6


def test_trade_match_no_fill_when_low_does_not_cross():
    sim = FillSimulator(model="trade_match", slippage_per_contract=0.0)
    t = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    candles = [Candle(ticker="T", timestamp=t, open=33, high=34, low=33, close=33, volume=100)]
    # yes_ask=32; low=33 does NOT cross → no fill
    fill = sim.simulate(_edge("yes"), _snapshot(30, 32), contracts=5, t=t, recent_candles=candles)
    assert fill is None
