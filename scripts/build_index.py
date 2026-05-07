"""
Cold-path index builder.

Usage:
    python3 -m scripts.build_index
    python3 -m scripts.build_index --max-markets 500   # fast dev/test run

Steps:
    1. Fetch all open Kalshi markets via REST API (paginated, no auth)
    2. Save catalog to data/catalog.json
    3. Build embedding inputs from market titles/subtitles
    4. Embed via OpenAI text-embedding-3-small (batches of 100)
    5. Build FAISS IndexFlatIP vector store
    6. Save index to data/vectors.faiss + data/vectors_meta.json
"""

import argparse
import asyncio
import time

from dotenv import load_dotenv

from src.catalog.embedder import build_embedding_inputs, embed_texts
from src.catalog.fetcher import build_catalog
from src.catalog.store import save_catalog
from src.catalog.vector_store import VectorStore
from src.config.kalshi_config import KalshiConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


_USEFUL_CATEGORIES = {
    "Elections", "Politics", "Economics", "Crypto", "Financials",
    "Commodities", "Companies", "Science and Technology", "Climate and Weather",
    "World", "Health",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kalshi market catalog and FAISS index")
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Cap on useful markets after filtering (default: all).",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip category filtering — index all non-parlay markets.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip API fetch — reuse existing data/catalog.json. Useful for re-filtering/re-embedding.",
    )
    return parser.parse_args()


async def main(max_markets: int | None = None, no_filter: bool = False, skip_fetch: bool = False) -> None:
    load_dotenv()
    config = KalshiConfig()
    total_start = time.perf_counter()

    # Stage 1: fetch catalog (or reuse existing)
    t = time.perf_counter()
    if skip_fetch:
        from src.catalog.store import load_catalog
        logger.info("Skipping fetch — loading existing catalog from disk")
        markets = load_catalog()
    else:
        logger.info("Fetching market catalog from Kalshi API")
        markets = await build_catalog(config.rest_base_url)
    logger.info("Stage 1 complete", extra={"step": "fetch", "elapsed_s": round(time.perf_counter() - t, 2)})

    if not markets:
        logger.error("No markets returned after filtering")
        return

    # Filter to useful categories unless --no-filter is passed
    if not no_filter:
        before = len(markets)
        markets = [m for m in markets if m.category in _USEFUL_CATEGORIES]
        logger.info(
            "Category filter applied",
            extra={"before": before, "after": len(markets), "categories": sorted(_USEFUL_CATEGORIES)},
        )

    if not markets:
        logger.error("No markets remain after category filter — run with --no-filter to skip")
        return

    if max_markets:
        markets = markets[:max_markets]
        logger.info("Applied max_markets cap", extra={"max_markets": max_markets})

    save_catalog(markets)
    logger.info("Catalog saved", extra={"count": len(markets)})

    # Stage 2: embed
    t = time.perf_counter()
    logger.info("Building embedding inputs")
    texts = build_embedding_inputs(markets)

    logger.info("Embedding markets via OpenAI", extra={"total": len(texts)})
    embeddings = embed_texts(texts)
    logger.info("Stage 2 complete", extra={"step": "embed", "elapsed_s": round(time.perf_counter() - t, 2)})

    # Stage 3: build and save FAISS index
    t = time.perf_counter()
    logger.info("Building FAISS index")
    metadata = [m.model_dump(mode="json") for m in markets]
    vs = VectorStore()
    vs.add(embeddings, metadata)
    vs.save()
    logger.info("Stage 3 complete", extra={"step": "faiss", "elapsed_s": round(time.perf_counter() - t, 2)})

    logger.info(
        "Index build complete",
        extra={"markets_indexed": len(markets), "total_elapsed_s": round(time.perf_counter() - total_start, 2)},
    )


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(max_markets=args.max_markets, no_filter=args.no_filter, skip_fetch=args.skip_fetch))
