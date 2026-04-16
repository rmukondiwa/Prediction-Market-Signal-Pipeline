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


async def fetch_all_markets(base_url: str) -> list[dict]:
    """
    Paginate GET /markets?status=open&limit=1000 until cursor is exhausted.
    Returns the raw list of market dicts from the API.
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
            if not cursor:
                break

    logger.info("Finished fetching all markets", extra={"total": len(markets)})
    return markets


async def fetch_event(
    session: aiohttp.ClientSession,
    base_url: str,
    event_ticker: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """
    Fetch a single event by ticker. Returns the event dict or None on failure.
    Wrapped with retry_with_backoff; on exhaustion logs a warning and returns None.
    """
    async def _fetch():
        async with semaphore:
            async with session.get(f"{base_url}/events/{event_ticker}") as resp:
                resp.raise_for_status()
                return await resp.json()

    try:
        data = await retry_with_backoff(
            _fetch,
            max_attempts=3,
            base_delay=0.5,
            label=f"fetch_event:{event_ticker}",
        )
        return data.get("event")
    except Exception as exc:
        logger.warning(
            "Event fetch failed after retries, using ticker as fallback title",
            extra={"event_ticker": event_ticker, "error": str(exc)},
        )
        return None


async def build_catalog(base_url: str) -> list[CatalogMarket]:
    """
    Orchestrator for the cold-path catalog build:
      1. Fetch all open markets (paginated)
      2. De-duplicate event_tickers (~3000 markets → ~300-500 unique events)
      3. Fetch each event concurrently, bounded by Semaphore(18) for rate limiting
      4. Merge market + event data into CatalogMarket records
    """
    markets = await fetch_all_markets(base_url)

    unique_event_tickers: set[str] = {
        m["event_ticker"] for m in markets if m.get("event_ticker")
    }
    logger.info("Unique events to fetch", extra={"count": len(unique_event_tickers)})

    semaphore = asyncio.Semaphore(18)
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

        yes_bid = m.get("yes_bid") or 0
        yes_ask = m.get("yes_ask") or 0
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
