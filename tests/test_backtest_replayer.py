from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.models import Candle
from src.backtest.replayer import (
    build_snapshot_at,
    find_candle_at,
    synthetic_candles,
    walk_timesteps,
    _granularity_to_timedelta,
)


def _candle(ticker: str, t: datetime, close: float) -> Candle:
    return Candle(ticker=ticker, timestamp=t, open=close, high=close, low=close, close=close, volume=10)


def test_find_candle_at_returns_most_recent_at_or_before():
    base = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    candles = [
        _candle("A", base, 50),
        _candle("A", base + timedelta(hours=1), 55),
        _candle("A", base + timedelta(hours=2), 60),
    ]
    found = find_candle_at(candles, base + timedelta(hours=1, minutes=30))
    assert found is not None
    assert found.close == 55


def test_find_candle_at_returns_none_before_first():
    base = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    candles = [_candle("A", base, 50)]
    assert find_candle_at(candles, base - timedelta(hours=1)) is None


def test_build_snapshot_uses_close_with_spread():
    base = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    candles = [_candle("KXIRAN", base, 25)]
    snap = build_snapshot_at("KXIRAN", "Iran market", base, candles, half_spread_cents=2)
    assert snap is not None
    assert snap.market == "KXIRAN"
    assert snap.yes_bid == 23
    assert snap.yes_ask == 27
    assert snap.quoted_price == 25
    assert abs(snap.implied_probability - 0.25) < 1e-9


def test_build_snapshot_clamps_extreme_prices():
    base = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
    # close at 99 → ask would round to 100, but we clamp to 99
    candles = [_candle("X", base, 99)]
    snap = build_snapshot_at("X", "x", base, candles, half_spread_cents=1)
    assert snap is not None
    assert snap.yes_ask <= 99
    assert snap.yes_bid >= 1


def test_build_snapshot_returns_none_when_no_candle():
    snap = build_snapshot_at("Z", "z", datetime(2026, 1, 1, tzinfo=timezone.utc), [], 1)
    assert snap is None


def test_walk_timesteps_hourly():
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, 4, 0, tzinfo=timezone.utc)
    steps = list(walk_timesteps(start, end, "1h"))
    assert len(steps) == 4
    assert steps[0] == start
    assert steps[-1] == datetime(2026, 4, 1, 3, 0, tzinfo=timezone.utc)


def test_walk_timesteps_daily():
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 4, tzinfo=timezone.utc)
    assert len(list(walk_timesteps(start, end, "1d"))) == 3


def test_granularity_parsing():
    assert _granularity_to_timedelta("1h") == timedelta(hours=1)
    assert _granularity_to_timedelta("4h") == timedelta(hours=4)
    assert _granularity_to_timedelta("1d") == timedelta(days=1)
    assert _granularity_to_timedelta("30m") == timedelta(minutes=30)
    with pytest.raises(ValueError):
        _granularity_to_timedelta("1w")


def test_synthetic_candles_produces_linear_drift():
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, 10, tzinfo=timezone.utc)
    candles = synthetic_candles("X", start, end, open_price=20.0, close_price=80.0, granularity="1h")
    assert len(candles) >= 10
    assert candles[0].close == 20.0
    # Last candle should be at or near close_price
    assert abs(candles[-1].close - 80.0) < 1e-6
