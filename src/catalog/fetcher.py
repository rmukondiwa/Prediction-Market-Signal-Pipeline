"""
Market catalog fetcher.

Cold-path stage 1: paginates all open Kalshi markets, fetches parent event
metadata for human-readable titles, and assembles CatalogMarket records.

Both endpoints are public — no authentication required.
"""

import asyncio

import aiohttp

from src.catalog.models import CatalogMarket
from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)

# Kalshi parlay/combo product prefixes — no meaningful titles, useless for causal inference
_SKIP_EVENT_PREFIXES = ("KXMVECROSSCATEGORY", "KXMVESPORTSMULTIGAMEEXTENDED")


async def fetch_all_markets(base_url: str, max_markets: int | None = None) -> list[dict]:
    """
    Paginate GET /markets?status=open&limit=1000 until cursor is exhausted.
    Returns the raw list of market dicts from the API.

    Args:
        max_markets: If set, stop fetching after this many markets. Useful for
                     dev/testing to avoid pulling the full 30k+ catalog.
    """
    markets: list[dict] = []
    cursor: str | None = None

    async with aiohttp.ClientSession() as session:
        while True:
            params: dict = {"status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor

            async def _fetch(session=session, params=params):
                async with session.get(f"{base_url}/markets", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()

            data = await retry_with_backoff(_fetch, label="fetch_all_markets page")
            batch = data.get("markets", [])
            markets.extend(batch)

            cursor = data.get("cursor", "")
            logger.info(
                "Fetched market page",
                extra={"count": len(batch), "total_so_far": len(markets)},
            )

            if not cursor or (max_markets and len(markets) >= max_markets):
                break

    if max_markets:
        markets = markets[:max_markets]
    logger.info("Finished fetching all markets", extra={"total": len(markets)})
    return markets


async def fetch_settled_markets(base_url: str, max_markets: int = 5000) -> list[dict]:
    """
    Paginate GET /markets?status=settled&limit=1000.
    Stops after max_markets (default 5000) — Kalshi has 400k+ settled markets,
    fetching all is unnecessary for backtesting.
    """
    markets: list[dict] = []
    cursor: str | None = None

    async with aiohttp.ClientSession() as session:
        while True:
            params: dict = {"status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor

            async def _fetch(session=session, params=params):
                async with session.get(f"{base_url}/markets", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()

            data = await retry_with_backoff(_fetch, label="fetch_settled_markets page")
            batch = data.get("markets", [])
            markets.extend(batch)

            cursor = data.get("cursor", "")
            logger.info(
                "Fetched settled market page",
                extra={"count": len(batch), "total_so_far": len(markets)},
            )

            if not cursor or len(markets) >= max_markets:
                break

    markets = markets[:max_markets]
    logger.info("Finished fetching settled markets", extra={"total": len(markets)})
    return markets


async def fetch_event(
    session: aiohttp.ClientSession,
    base_url: str,
    event_ticker: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """
    Fetch a single event by ticker. Returns the event dict or None on failure.
    429 responses trigger a longer cooldown before retrying.
    """
    async def _fetch():
        async with semaphore:
            await asyncio.sleep(0.1)  # 100ms stagger between concurrent requests
            async with session.get(f"{base_url}/events/{event_ticker}") as resp:
                if resp.status == 429:
                    # Take a longer break when rate limited — don't hammer immediately
                    logger.warning(
                        "Rate limited on event fetch, cooling down",
                        extra={"event_ticker": event_ticker},
                    )
                    await asyncio.sleep(5.0)
                    resp.raise_for_status()
                resp.raise_for_status()
                return await resp.json()

    try:
        data = await retry_with_backoff(
            _fetch,
            max_attempts=5,
            base_delay=2.0,  # start with 2s backoff, doubles on each attempt
            label=f"fetch_event:{event_ticker}",
        )
        return data.get("event")
    except Exception as exc:
        logger.warning(
            "Event fetch failed after retries, using ticker as fallback title",
            extra={"event_ticker": event_ticker, "error": str(exc)},
        )
        return None


async def build_catalog(base_url: str, max_markets: int | None = None) -> list[CatalogMarket]:
    """
    Orchestrator for the cold-path catalog build:
      1. Fetch all open markets (paginated)
      2. De-duplicate event_tickers (~3000 markets → ~300-500 unique events)
      3. Fetch each event concurrently, bounded by Semaphore(18) for rate limiting
      4. Merge market + event data into CatalogMarket records

    Args:
        max_markets: Cap on markets to fetch. None = full catalog. Set to e.g. 500
                     for fast dev/test runs.
    """
    markets = await fetch_all_markets(base_url, max_markets=max_markets)

    # Filter out parlay/combo markets before fetching events
    markets = [
        m for m in markets
        if not any(m.get("event_ticker", "").startswith(p) for p in _SKIP_EVENT_PREFIXES)
    ]
    logger.info("Markets after filtering combos/parlays", extra={"count": len(markets)})

    # Apply max_markets cap after filtering so we get real markets, not just parlays
    if max_markets and len(markets) > max_markets:
        markets = markets[:max_markets]
        logger.info("Applied post-filter cap", extra={"max_markets": max_markets})

    unique_event_tickers: set[str] = {
        m["event_ticker"] for m in markets if m.get("event_ticker")
    }
    logger.info("Unique events to fetch", extra={"count": len(unique_event_tickers)})

    semaphore = asyncio.Semaphore(5)
    event_map: dict[str, dict] = {}

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_event(session, base_url, ticker, semaphore)
            for ticker in unique_event_tickers
        ]
        results = await asyncio.gather(*tasks)

    for ticker, event in zip(unique_event_tickers, results):
        if event:
            event_map[ticker] = event

    catalog: list[CatalogMarket] = []
    for m in markets:
        event_ticker = m.get("event_ticker", "")
        event = event_map.get(event_ticker)

        title = event.get("title", event_ticker) if event else event_ticker
        subtitle = m.get("subtitle") or m.get("yes_sub_title") or ""
        category = (event.get("category") if event else None) or m.get("category") or ""

        # API returns prices as dollar strings e.g. "0.4500" (0-1 scale)
        # Convert to cents (0-100) for our model
        yes_bid = round(float(m.get("yes_bid_dollars") or 0) * 100)
        yes_ask = round(float(m.get("yes_ask_dollars") or 0) * 100)
        implied_probability = (yes_bid + yes_ask) / 2 / 100

        try:
            catalog.append(
                CatalogMarket(
                    ticker=m["ticker"],
                    event_ticker=event_ticker,
                    title=title,
                    subtitle=subtitle,
                    category=category,
                    status=m.get("status", "open"),
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    implied_probability=implied_probability,
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping malformed market",
                extra={"ticker": m.get("ticker"), "error": str(exc)},
            )

    logger.info("Catalog built", extra={"total_markets": len(catalog)})
    return catalog
