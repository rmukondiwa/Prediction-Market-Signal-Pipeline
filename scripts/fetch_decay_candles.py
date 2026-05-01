"""
Fetch hourly candlesticks for the settlement-decay backtest universe.

Reads data/decay_universe_meta.json (built by the survey step), pulls
hourly OHLC for each market's lifetime via the public Kalshi API, and
writes the result to data/decay_candles.json. Idempotent — skips
markets already in the cache.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import aiohttp

from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
META_PATH = Path("data/decay_universe_meta.json")
CANDLES_PATH = Path("data/decay_candles.json")


def _to_ts(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


async def fetch_candles(session: aiohttp.ClientSession, ticker: str,
                        event_ticker: str, start_ts: int, end_ts: int,
                        period: int = 60) -> list[dict] | None:
    """OHLC for one market's lifetime. Tries period=60 (hourly) first, then
    1440 (daily) if the hourly query returns empty. None on hard error."""
    series = event_ticker.split("-", 1)[0]

    async def _query(p: int) -> list[dict]:
        url = (f"{BASE}/series/{series}/markets/{ticker}/candlesticks"
               f"?period_interval={p}&start_ts={start_ts}&end_ts={end_ts}")

        async def _do() -> list[dict]:
            async with session.get(url) as r:
                if r.status >= 500:
                    r.raise_for_status()
                if r.status >= 400:
                    return []
                data = await r.json()
                return data.get("candlesticks", [])

        return await retry_with_backoff(_do, max_attempts=3, base_delay=1.0,
                                        label=f"candles:{ticker}:{p}m")

    try:
        candles = await _query(period)
        if candles:
            return candles
        # Fall back to daily for markets where hourly is empty
        return await _query(1440)
    except Exception as e:
        logger.warning("Candle fetch failed", extra={"ticker": ticker, "error": str(e)})
        return None


async def main(limit: int | None = None) -> None:
    if not META_PATH.exists():
        raise SystemExit(f"Run the survey step first; missing {META_PATH}")

    meta = json.loads(META_PATH.read_text())
    if limit:
        meta = meta[:limit]
    print(f"Fetching candles for {len(meta)} markets...")

    cache: dict[str, Any] = {}
    if CANDLES_PATH.exists():
        cache = json.loads(CANDLES_PATH.read_text())
        print(f"  Cache hit: {len(cache)} markets already fetched")

    sem = asyncio.Semaphore(2)
    counts = {"new": 0, "skip": 0, "fail": 0, "empty": 0, "retried": 0}

    async def one(m: dict) -> None:
        ticker = m["ticker"]
        if ticker in cache and cache[ticker]:
            # only skip when we already have NON-empty data
            counts["skip"] += 1
            return
        if ticker in cache and not cache[ticker]:
            counts["retried"] += 1
        async with sem:
            await asyncio.sleep(0.4)  # 400ms — gentler on the rate limit
            try:
                start_ts = _to_ts(m["open_time"])
                end_ts = _to_ts(m["close_time"])
            except Exception:
                counts["fail"] += 1
                return
            async with aiohttp.ClientSession() as session:
                candles = await fetch_candles(session, ticker, m["event_ticker"],
                                              start_ts, end_ts)
            if candles is None:
                counts["fail"] += 1
                return
            if not candles:
                counts["empty"] += 1
                cache[ticker] = []
                return
            cache[ticker] = candles
            counts["new"] += 1
            if counts["new"] % 25 == 0:
                # snapshot incrementally so a crash doesn't lose progress
                CANDLES_PATH.write_text(json.dumps(cache))
                print(f"    progress: new={counts['new']} fail={counts['fail']} empty={counts['empty']}")

    await asyncio.gather(*(one(m) for m in meta))

    CANDLES_PATH.write_text(json.dumps(cache))
    print(f"\nDone. {counts}")
    print(f"Candles saved to {CANDLES_PATH}")
    sizes = [len(c) for c in cache.values() if c]
    if sizes:
        sizes.sort()
        print(f"Candles per market: median={sizes[len(sizes)//2]} max={max(sizes)} min={min(sizes)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    asyncio.run(main(args.limit))
