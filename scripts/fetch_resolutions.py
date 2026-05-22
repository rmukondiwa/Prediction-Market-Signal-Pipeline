"""
Backfill helper for Stage 7 backtests.

Pulls settlement data for a universe of tickers without depending on Stage 6
having been running. Uses the resolution_tracker module under the hood.

Usage:
    # Fastest — fetch only settled markets directly from Kalshi (~20 API calls)
    python -m scripts.fetch_resolutions --settled

    # Targeted — specific tickers
    python -m scripts.fetch_resolutions --tickers KXIRANUS-26JUN30,KXOIL100-26JUL31

    # Catalog-based with throttle control
    python -m scripts.fetch_resolutions --from-catalog data/catalog.json --limit 200 --concurrency 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.catalog.fetcher import fetch_settled_markets
from src.config.kalshi_config import KalshiConfig
from src.storage.resolution_tracker import track_resolutions
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill market resolutions for backtests")
    p.add_argument("--tickers", type=str, help="Comma-separated tickers")
    p.add_argument("--from-catalog", type=Path, help="Read tickers from catalog.json")
    p.add_argument("--settled", action="store_true",
                   help="Fetch settled markets directly from Kalshi API (fastest, no rate limit issues)")
    p.add_argument("--max-settled", type=int, default=5000,
                   help="Max settled markets to fetch with --settled (default: 5000)")
    p.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    p.add_argument("--concurrency", type=int, default=4,
                   help="Max concurrent API requests (default: 4, ~16 req/s)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of tickers checked — useful for quick tests")
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    cfg = KalshiConfig()

    tickers: list[str] = []
    if args.settled:
        logger.info("Fetching settled markets directly from Kalshi API")
        markets = await fetch_settled_markets(cfg.rest_base_url, max_markets=args.max_settled)
        tickers = [m["ticker"] for m in markets]
        logger.info("Settled markets fetched", extra={"count": len(tickers)})
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.from_catalog:
        catalog = json.loads(args.from_catalog.read_text())
        tickers = [m["ticker"] for m in catalog]
    else:
        print("Provide one of: --settled, --tickers, or --from-catalog")
        return

    if args.limit:
        tickers = tickers[: args.limit]
        logger.info("Limit applied", extra={"limit": args.limit})

    records = await track_resolutions(
        cfg.rest_base_url, tickers, args.archive_root, concurrency=args.concurrency
    )
    settled = [r for r in records if r.settlement_value is not None]
    print(f"Polled {len(records)} tickers; {len(settled)} are settled.")


if __name__ == "__main__":
    asyncio.run(main())
