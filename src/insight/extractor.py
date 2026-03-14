"""
Extractor: reads the latest market event from Redis and builds a MarketSnapshot.

Responsibilities:
- Pull the most recent entry from the market_events stream
- Fall back to orderbook_events (snapshot) if market_events is empty
- Compute derived fields (quoted_price, implied_probability)
- Return an exchange-agnostic MarketSnapshot ready for insight generation

This module is deterministic — no LLM involvement.
"""

import json
from datetime import datetime

import redis.asyncio as aioredis

from src.config.redis_config import RedisConfig
from src.insight.models import MarketSnapshot
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_ticker_to_event(ticker: str) -> str:
    """
    Derive a human-readable event name from a Kalshi ticker.
    e.g. 'KXINXMAXY-01JAN2027' -> 'KXINXMAXY (01JAN2027)'
    """
    parts = ticker.split("-", 1)
    if len(parts) == 2:
        return f"{parts[0]} ({parts[1]})"
    return ticker


def _snapshot_from_orderbook(ticker: str, event_data: dict) -> MarketSnapshot:
    """
    Derive a MarketSnapshot from an orderbook snapshot entry.
    yes_bid = best (highest) price in yes_levels
    yes_ask = 100 - best (highest) price in no_levels
    """
    yes_levels = event_data.get("yes_levels", [])
    no_levels = event_data.get("no_levels", [])

    yes_bid = max((lvl["price"] for lvl in yes_levels), default=0)
    best_no_bid = max((lvl["price"] for lvl in no_levels), default=0)
    yes_ask = 100 - best_no_bid if best_no_bid else yes_bid + 1

    quoted_price = (yes_bid + yes_ask) // 2

    return MarketSnapshot(
        event=_parse_ticker_to_event(ticker),
        market=ticker,
        outcome="YES",
        quoted_price=quoted_price,
        implied_probability=round(quoted_price / 100, 4),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        volume=0,
        open_interest=0,
        source=event_data.get("source", "kalshi"),
        timestamp=datetime.fromisoformat(event_data["timestamp"]),
    )


async def extract_latest_snapshot(
    ticker: str,
    redis_config: RedisConfig,
) -> MarketSnapshot:
    """
    Fetch the most recent market event for a given ticker from Redis Streams
    and return a structured MarketSnapshot.

    Tries market_events first; falls back to orderbook_events (snapshot entries)
    if no ticker data is found there.

    Raises:
        ValueError: if no data is found for the ticker in either stream.
    """
    redis = await aioredis.from_url(redis_config.url, decode_responses=True)

    try:
        market_entries = await redis.xrevrange(
            redis_config.market_events_stream, count=50
        )
        orderbook_entries = await redis.xrevrange(
            redis_config.orderbook_events_stream, count=50
        )
    finally:
        await redis.aclose()

    # Primary: market_events stream
    for _msg_id, fields in market_entries:
        raw_data = fields.get("data")
        if not raw_data:
            continue
        event_data = json.loads(raw_data)
        if event_data.get("ticker") != ticker:
            continue

        yes_bid = int(event_data["yes_bid"])
        yes_ask = int(event_data["yes_ask"])
        quoted_price = (yes_bid + yes_ask) // 2

        logger.info("Snapshot sourced from market_events", extra={"ticker": ticker})
        return MarketSnapshot(
            event=_parse_ticker_to_event(ticker),
            market=ticker,
            outcome="YES",
            quoted_price=quoted_price,
            implied_probability=round(quoted_price / 100, 4),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=int(event_data.get("volume", 0)),
            open_interest=int(event_data.get("open_interest", 0)),
            source=event_data.get("source", "kalshi"),
            timestamp=datetime.fromisoformat(event_data["timestamp"]),
        )

    # Fallback: orderbook_events stream (snapshot entries only)
    for _msg_id, fields in orderbook_entries:
        raw_data = fields.get("data")
        if not raw_data:
            continue
        event_data = json.loads(raw_data)
        if event_data.get("ticker") != ticker:
            continue
        if event_data.get("event_type") != "snapshot":
            continue

        logger.info("Snapshot sourced from orderbook_events (fallback)", extra={"ticker": ticker})
        return _snapshot_from_orderbook(ticker, event_data)

    raise ValueError(
        f"No market data found for ticker '{ticker}' in market_events or orderbook_events. "
        "Make sure the ingestion pipeline (main.py) has run first."
    )
