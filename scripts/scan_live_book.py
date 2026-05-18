"""
Live Kalshi order book aggregator + monotonicity scanner.

Subscribes to all markets in a strike-ladder series (default: KXBTCD)
via the Kalshi WebSocket, maintains a live in-memory order book for
each market, and checks the ladder for monotonicity violations every
100ms.

A monotonicity violation: best_bid[lower_strike] > best_ask[higher_strike]
P(BTC > $89k) must always be >= P(BTC > $90k). When the book contradicts
this, there is a potential arb: buy YES at the higher strike, sell YES at
the lower strike, lock in the spread.

Note: our REST order latency (~200-500ms) means we likely can't race to
fill these before other bots. The primary goal is to answer: "Do violations
exist, and how long do they last?" That data informs whether faster
execution is worth investing in.

Violations are logged to logs/live_book_violations.jsonl.
A heartbeat summary is logged every 60 seconds.

Usage:
    python -m scripts.scan_live_book
    python -m scripts.scan_live_book --series KXETHD
    python -m scripts.scan_live_book --check-interval 0.05
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.catalog.models import CatalogMarket
from src.catalog.store import load_catalog
from src.config.kalshi_config import KalshiConfig
from src.ingestion.kalshi.websocket_client import KalshiWebSocketClient
from src.models.orderbook_event import OrderBookEvent
from src.portfolio.order_book import OrderBookAggregator
from src.utils.logging import get_logger

logger = get_logger(__name__)

CATALOG_PATH = Path("data/catalog.json")
VIOLATIONS_LOG = Path("logs/live_book_violations.jsonl")
HEARTBEAT_INTERVAL = 600   # checks (60s at default 0.1s interval)

_STRIKE_RE = re.compile(r"-T([\d.]+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_strike(ticker: str) -> float | None:
    """Extract numeric strike from e.g. KXBTCD-26MAY1312-T89799.99 → 89799.99."""
    m = _STRIKE_RE.search(ticker)
    return float(m.group(1)) if m else None


def _load_series(series: str) -> list[CatalogMarket]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            "data/catalog.json not found. Run: python3 -m scripts.build_index"
        )
    catalog = load_catalog(CATALOG_PATH)
    markets = [
        m for m in catalog
        if m.ticker.startswith(series) and _parse_strike(m.ticker) is not None
    ]
    if not markets:
        raise ValueError(f"No markets found for series '{series}' in catalog.")
    logger.info("Series loaded", extra={"series": series, "count": len(markets)})
    return markets


def _group_by_event(markets: list[CatalogMarket]) -> dict[str, list[CatalogMarket]]:
    """Group markets by their parent event_ticker."""
    groups: dict[str, list[CatalogMarket]] = {}
    for m in markets:
        groups.setdefault(m.event_ticker, []).append(m)
    return groups


def _append_log(record: dict) -> None:
    VIOLATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with VIOLATIONS_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Monotonicity check
# ---------------------------------------------------------------------------

def check_monotonicity(
    event_ticker: str,
    markets: list[CatalogMarket],
    aggregator: OrderBookAggregator,
) -> list[dict]:
    """
    Sort markets by strike ascending, then walk adjacent pairs.
    A violation is: best_bid[lower] > best_ask[higher].

    Returns a list of violation dicts (empty if none found).
    """
    ordered = sorted(markets, key=lambda m: _parse_strike(m.ticker))  # type: ignore[arg-type]
    violations = []

    for i in range(len(ordered) - 1):
        low = ordered[i]
        high = ordered[i + 1]

        book_low = aggregator.get_book(low.ticker)
        book_high = aggregator.get_book(high.ticker)

        if book_low is None or book_high is None:
            continue

        bid = book_low.best_bid
        ask = book_high.best_ask

        if bid is None or ask is None:
            continue

        if bid > ask:
            violations.append({
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "event_ticker": event_ticker,
                "low_ticker": low.ticker,
                "high_ticker": high.ticker,
                "low_strike": _parse_strike(low.ticker),
                "high_strike": _parse_strike(high.ticker),
                "bid_low_cents": bid,
                "ask_high_cents": ask,
                "spread_cents": bid - ask,
                "action": (
                    f"buy YES {high.ticker} @ {ask}¢, "
                    f"sell YES {low.ticker} @ {bid}¢"
                ),
            })

    return violations


# ---------------------------------------------------------------------------
# Async tasks
# ---------------------------------------------------------------------------

async def check_loop(
    groups: dict[str, list[CatalogMarket]],
    aggregator: OrderBookAggregator,
    interval: float,
) -> None:
    """Runs monotonicity checks on every event group at a fixed interval."""
    checks = 0
    total_violations = 0

    while True:
        await asyncio.sleep(interval)
        checks += 1

        for event_ticker, markets in groups.items():
            for v in check_monotonicity(event_ticker, markets, aggregator):
                total_violations += 1
                logger.warning("Monotonicity violation", extra=v)
                _append_log(v)

        if checks % HEARTBEAT_INTERVAL == 0:
            logger.info(
                "Scanner heartbeat",
                extra={
                    "checks": checks,
                    "total_violations": total_violations,
                    "books_initialised": aggregator.initialised_count(),
                    "books_total": aggregator.book_count(),
                },
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Kalshi order book monotonicity scanner")
    p.add_argument(
        "--series", default="KXBTCD",
        help="Ticker prefix to monitor (default: KXBTCD)",
    )
    p.add_argument(
        "--check-interval", type=float, default=0.1,
        help="Seconds between monotonicity checks (default: 0.1)",
    )
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    markets = _load_series(args.series)
    groups = _group_by_event(markets)
    tickers = [m.ticker for m in markets]
    aggregator = OrderBookAggregator()

    logger.info(
        "Starting live book scanner",
        extra={
            "series": args.series,
            "markets": len(tickers),
            "event_groups": len(groups),
            "check_interval_s": args.check_interval,
            "log": str(VIOLATIONS_LOG),
        },
    )
    _append_log({
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "event": "scanner_started",
        "series": args.series,
        "markets": len(tickers),
        "event_groups": len(groups),
    })

    async def on_event(event) -> None:
        if isinstance(event, OrderBookEvent):
            aggregator.on_event(event)

    config = KalshiConfig(
        market_tickers=tickers,
        channels=["orderbook_delta"],
    )
    ws_client = KalshiWebSocketClient(config=config, on_event=on_event)

    await asyncio.gather(
        ws_client.run(),
        check_loop(groups, aggregator, args.check_interval),
    )


if __name__ == "__main__":
    asyncio.run(main())
