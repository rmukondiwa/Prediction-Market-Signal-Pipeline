"""
Daily archive job.

Usage:
    python -m scripts.archive_daily
    python -m scripts.archive_daily --skip-streams      # only catalog + resolutions
    python -m scripts.archive_daily --skip-resolutions  # only streams + catalog

Schedule via cron, e.g. at 02:00 UTC:
    0 2 * * *  cd /path/to/project && .venv/bin/python -m scripts.archive_daily
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from src.config.kalshi_config import KalshiConfig
from src.config.redis_config import RedisConfig
from src.storage.catalog_archive import snapshot_catalog
from src.storage.resolution_tracker import load_resolutions, track_resolutions
from src.storage.snapshotter import run_snapshot
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_ARCHIVE_ROOT = Path("data/archive")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Phase 5 archive job")
    parser.add_argument("--archive-root", type=Path, default=_DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--skip-streams", action="store_true")
    parser.add_argument("--skip-catalog", action="store_true")
    parser.add_argument("--skip-resolutions", action="store_true")
    parser.add_argument("--max-markets", type=int, default=25000,
                        help="Cap markets fetched for catalog snapshot (default: 25000)")
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    if not args.skip_streams:
        try:
            redis_cfg = RedisConfig()
            await run_snapshot(redis_cfg, args.archive_root)
        except Exception as exc:
            logger.error("Stream snapshot failed", extra={"error": str(exc)})

    catalog_tickers: list[str] = []
    if not args.skip_catalog:
        try:
            kalshi_cfg = KalshiConfig()
            path = await snapshot_catalog(kalshi_cfg.rest_base_url, args.archive_root,
                                          max_markets=args.max_markets)
            from src.storage.catalog_archive import load_catalog_parquet
            catalog_tickers = [m.ticker for m in load_catalog_parquet(path)]
        except Exception as exc:
            logger.error("Catalog snapshot failed", extra={"error": str(exc)})

    if not args.skip_resolutions:
        try:
            kalshi_cfg = KalshiConfig()
            tickers = catalog_tickers
            if not tickers:
                # Fall back to whatever is already in resolutions
                tickers = list(load_resolutions(args.archive_root).keys())
            if tickers:
                await track_resolutions(kalshi_cfg.rest_base_url, tickers, args.archive_root)
            else:
                logger.warning("No tickers known yet — skipping resolutions pass")
        except Exception as exc:
            logger.error("Resolution tracking failed", extra={"error": str(exc)})


if __name__ == "__main__":
    asyncio.run(main())
