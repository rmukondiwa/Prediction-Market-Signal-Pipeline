"""
Backfill helper for Stage 7 backtests.

Pulls settlement data for a universe of tickers without depending on Stage 6
having been running. Uses the resolution_tracker module under the hood.

Usage:
    python -m scripts.fetch_resolutions --tickers KXIRANUS-26JUN30,KXOIL100-26JUL31
    python -m scripts.fetch_resolutions --from-catalog data/catalog.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.config.kalshi_config import KalshiConfig
from src.storage.resolution_tracker import track_resolutions
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill market resolutions for backtests")
    p.add_argument("--tickers", type=str, help="Comma-separated tickers")
    p.add_argument("--from-catalog", type=Path, help="Read tickers from catalog.json")
    p.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    tickers: list[str] = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.from_catalog:
        catalog = json.loads(args.from_catalog.read_text())
        tickers = [m["ticker"] for m in catalog]
    else:
        print("Provide either --tickers or --from-catalog")
        return

    cfg = KalshiConfig()
    records = await track_resolutions(cfg.rest_base_url, tickers, args.archive_root)
    settled = [r for r in records if r.settlement_value is not None]
    print(f"Polled {len(records)} tickers; {len(settled)} are settled.")


if __name__ == "__main__":
    asyncio.run(main())
