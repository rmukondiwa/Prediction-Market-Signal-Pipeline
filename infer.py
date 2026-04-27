"""
Hot-path inference CLI.

Usage:
    python3 infer.py TICKER
    python3 infer.py TICKER --dry-run
    python3 infer.py TICKER --refresh-catalog
    python3 infer.py TICKER --no-redis

Flags:
    ticker            Market ticker to analyse. Falls back to KALSHI_INSIGHT_TICKER env var.
    --dry-run         Run retrieval + reranking only. Print context markets. Skip inference.
    --refresh-catalog Rebuild catalog and embedding index before inference.
    --no-redis        Skip Redis snapshot; use catalog prices instead (dev mode).
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.catalog.store import load_catalog
from src.catalog.vector_store import VectorStore
from src.context.reranker import rerank
from src.context.retriever import retrieve_candidates
from src.inference.engine import run_inference
from src.insight.models import MarketSnapshot
from src.utils.logging import get_logger

logger = get_logger(__name__)

_CATALOG_PATH = Path("data/catalog.json")
_VECTORS_PATH = "data/vectors"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalshi cross-market inference engine")
    parser.add_argument("ticker", nargs="?", help="Market ticker to analyse")
    parser.add_argument("--dry-run", action="store_true", help="Retrieve + rerank only, skip inference")
    parser.add_argument("--refresh-catalog", action="store_true", help="Rebuild catalog and index first")
    parser.add_argument("--no-redis", action="store_true", help="Use catalog prices instead of live Redis snapshot")
    return parser.parse_args()


def _check_artifacts() -> bool:
    missing = []
    if not _CATALOG_PATH.exists():
        missing.append(str(_CATALOG_PATH))
    if not Path(f"{_VECTORS_PATH}.faiss").exists():
        missing.append(f"{_VECTORS_PATH}.faiss")
    if missing:
        print("Missing index artifacts. Run: python3 -m scripts.build_index")
        return False
    return True


def _snapshot_from_catalog(ticker: str, catalog) -> MarketSnapshot | None:
    market = next((m for m in catalog if m.ticker == ticker), None)
    if market is None:
        return None
    return MarketSnapshot(
        event=market.title,
        market=market.ticker,
        outcome=market.subtitle or "Yes",
        quoted_price=int((market.yes_bid + market.yes_ask) / 2),
        implied_probability=market.implied_probability,
        yes_bid=market.yes_bid,
        yes_ask=market.yes_ask,
        volume=0,
        open_interest=0,
        source="catalog",
        timestamp=datetime.now(timezone.utc),
    )


def _snapshot_from_redis(ticker: str) -> MarketSnapshot | None:
    try:
        from src.config.redis_config import RedisConfig
        from src.insight.extractor import extract_snapshot
        config = RedisConfig()
        return extract_snapshot(ticker, config)
    except Exception as exc:
        logger.warning("Redis snapshot failed", extra={"ticker": ticker, "error": str(exc)})
        return None


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    ticker = args.ticker or os.getenv("KALSHI_INSIGHT_TICKER", "").strip()
    if not ticker:
        print("Error: provide a ticker as argument or set KALSHI_INSIGHT_TICKER in .env")
        sys.exit(1)

    # Optionally rebuild the index first
    if args.refresh_catalog:
        logger.info("Refreshing catalog and index")
        from scripts.build_index import main as build_main
        await build_main()

    if not _check_artifacts():
        sys.exit(1)

    # Load catalog and vector store
    catalog = load_catalog(_CATALOG_PATH)
    vs = VectorStore()
    vs.load(_VECTORS_PATH)

    # Retrieve candidates via vector search
    candidates = retrieve_candidates(ticker, catalog, vs, k=30)

    # Rerank via Groq LLM
    focus_market = next((m for m in catalog if m.ticker == ticker), None)
    if focus_market is None:
        logger.warning("Ticker not in catalog, proceeding with fallback", extra={"ticker": ticker})
        from src.catalog.models import CatalogMarket
        focus_market = CatalogMarket(
            ticker=ticker, event_ticker=ticker, title=ticker,
            subtitle="", category="", status="open",
            yes_bid=50, yes_ask=50, implied_probability=0.5,
        )

    context = rerank(focus_market, candidates)

    if args.dry_run:
        print(f"\n--- DRY RUN: context markets for {ticker} ---\n")
        for m in context:
            print(f"  [{m.relevance_score:.1f}/10] {m.ticker} | {m.title}")
            print(f"    implied: {m.implied_probability:.1%} | {m.relationship}\n")
        if not context:
            print("  No context markets passed the relevance threshold.")
        return

    if context == [] :
        logger.warning(
            "Reranker returned 0 context markets, running single-market mode",
            extra={"ticker": ticker},
        )

    # Get market snapshot
    snapshot: MarketSnapshot | None = None

    if not args.no_redis:
        snapshot = _snapshot_from_redis(ticker)
        if snapshot is None:
            print("Redis unreachable or no data for ticker. Use --no-redis to use catalog prices.")
            sys.exit(1)

    if snapshot is None:
        snapshot = _snapshot_from_catalog(ticker, catalog)

    if snapshot is None:
        print(f"No snapshot available for {ticker}. Run python3 -m scripts.build_index to refresh catalog.")
        sys.exit(1)

    # Run inference
    try:
        report = run_inference(snapshot, context)
    except Exception as exc:
        # Retry once
        logger.warning("Inference failed, retrying once", extra={"error": str(exc)})
        try:
            report = run_inference(snapshot, context)
        except Exception as exc2:
            print(f"Inference error: {exc2}")
            sys.exit(1)

    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
