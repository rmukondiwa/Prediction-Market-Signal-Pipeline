"""
Replayer — reconstructs the state of a market at a historical timestep.

Given a list of `Candle`s for a ticker, this module produces a `MarketSnapshot`
suitable for feeding into the existing inference engine. The bid/ask spread is
approximated as `close ± half_spread` since we don't have order book history
prior to Stage 6 archive accumulation. This is documented and accepted bias —
once 60+ days of archive exist, replace with order-book-derived snapshots.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from src.backtest.models import Candle
from src.insight.models import MarketSnapshot
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_candle_at(candles: list[Candle], t: datetime) -> Candle | None:
    """Return the most recent candle at or before t. None if no such candle."""
    eligible = [c for c in candles if c.timestamp <= t]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.timestamp)


def build_snapshot_at(
    ticker: str,
    title: str,
    t: datetime,
    candles: list[Candle],
    half_spread_cents: int = 1,
    source: str = "candle_replay",
) -> MarketSnapshot | None:
    """
    Build a MarketSnapshot for `ticker` as of time `t` using candle data.
    Returns None if no candle covers `t` (market not yet trading at that time).
    """
    candle = find_candle_at(candles, t)
    if candle is None:
        return None

    close_cents = int(round(candle.close))
    yes_bid = max(1, close_cents - half_spread_cents)
    yes_ask = min(99, close_cents + half_spread_cents)
    quoted = (yes_bid + yes_ask) // 2

    return MarketSnapshot(
        event=title,
        market=ticker,
        outcome="YES",
        quoted_price=quoted,
        implied_probability=round(quoted / 100, 4),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        volume=candle.volume,
        open_interest=0,  # not available from candles
        source=source,
        timestamp=t if t.tzinfo else t.replace(tzinfo=timezone.utc),
    )


def walk_timesteps(start: datetime, end: datetime, granularity: str) -> Iterable[datetime]:
    """Yield evenly spaced timestamps from start (inclusive) to end (exclusive)."""
    delta = _granularity_to_timedelta(granularity)
    t = start
    while t < end:
        yield t
        t = t + delta


def _granularity_to_timedelta(g: str) -> timedelta:
    g = g.strip().lower()
    if g.endswith("h"):
        return timedelta(hours=int(g[:-1]))
    if g.endswith("d"):
        return timedelta(days=int(g[:-1]))
    if g.endswith("m"):
        return timedelta(minutes=int(g[:-1]))
    raise ValueError(f"Unsupported granularity: {g!r}")


def synthetic_candles(
    ticker: str,
    start: datetime,
    end: datetime,
    open_price: float,
    close_price: float,
    granularity: str = "1h",
) -> list[Candle]:
    """
    Generate a synthetic linear-interpolation candle series. Useful for tests
    and for filling gaps when Kalshi candle history is missing.
    """
    delta = _granularity_to_timedelta(granularity)
    out: list[Candle] = []
    total = max(1, int((end - start).total_seconds() / delta.total_seconds()))
    for i in range(total + 1):
        t = start + i * delta
        if t > end:
            break
        frac = i / max(1, total)
        price = open_price + (close_price - open_price) * frac
        out.append(
            Candle(
                ticker=ticker,
                timestamp=t,
                open=price, high=price, low=price, close=price,
                volume=100,
            )
        )
    return out
