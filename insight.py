"""
Insight Layer entry point.

Reads the latest market event for a configured ticker from Redis,
extracts a structured snapshot, and generates a market insight via Claude.

Usage:
    python insight.py

Requires:
    - The ingestion pipeline (main.py) to have run at least briefly so that
      market events exist in Redis.
    - ANTHROPIC_API_KEY set in your environment or .env file.
"""

import asyncio
import json

from dotenv import load_dotenv

from src.config.redis_config import RedisConfig
from src.insight.extractor import extract_latest_snapshot
from src.insight.generator import generate_insight
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

# Ticker to generate insight for — override via env or edit directly
import os
TICKER = os.getenv("KALSHI_INSIGHT_TICKER") or os.getenv("KALSHI_MARKET_TICKERS", "").split(",")[0].strip()


async def main() -> None:
    if not TICKER:
        print("ERROR: No ticker configured. Set KALSHI_MARKET_TICKERS in your .env file.")
        return

    redis_cfg = RedisConfig()

    print(f"Extracting latest market snapshot for: {TICKER}")
    snapshot = await extract_latest_snapshot(TICKER, redis_cfg)

    print("Generating insight via Claude...\n")
    report = generate_insight(snapshot)

    output = json.loads(report.model_dump_json(indent=2))
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
