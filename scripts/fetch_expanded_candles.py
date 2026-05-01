"""Fetch candles for the expanded universe with smart period selection."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
from pathlib import Path

import aiohttp

from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
META_PATH = Path("data/expanded_universe_meta.json")
CANDLES_PATH = Path("data/expanded_candles.json")


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _pick_period(lifetime_seconds: int) -> int:
    """Pick the smallest period that returns ~50-200 candles for the lifetime."""
    h = lifetime_seconds / 3600
    if h <= 1.5:
        return 1     # ≤90 min → 1-min candles
    if h <= 30:
        return 60    # 1.5h-30h → hourly
    return 1440      # multi-day → daily


async def fetch(session, ticker, event_ticker, start_ts, end_ts, period):
    series = event_ticker.split("-", 1)[0]
    url = (f"{BASE}/series/{series}/markets/{ticker}/candlesticks"
           f"?period_interval={period}&start_ts={start_ts}&end_ts={end_ts}")

    async def _do():
        async with session.get(url) as r:
            if r.status >= 500:
                r.raise_for_status()
            if r.status >= 400:
                return []
            data = await r.json()
            return data.get("candlesticks", [])

    try:
        return await retry_with_backoff(_do, max_attempts=3, base_delay=1.0,
                                        label=f"exp:{ticker}:{period}")
    except Exception as e:
        return None


async def main(limit: int | None = None) -> None:
    meta = json.loads(META_PATH.read_text())
    if limit:
        meta = meta[:limit]
    print(f"Fetching candles for {len(meta)} expanded-universe markets...")

    cache = {}
    if CANDLES_PATH.exists():
        cache = json.loads(CANDLES_PATH.read_text())
        print(f"  Cache hit: {len(cache)} markets already fetched")

    sem = asyncio.Semaphore(3)
    counts = {"new": 0, "skip": 0, "fail": 0, "empty": 0}

    async def one(m: dict) -> None:
        ticker = m["ticker"]
        if ticker in cache and cache[ticker]:
            counts["skip"] += 1
            return
        async with sem:
            await asyncio.sleep(0.3)
            try:
                start_ts = _ts(m["open_time"])
                end_ts = _ts(m["close_time"])
            except Exception:
                counts["fail"] += 1
                return
            lifetime = end_ts - start_ts
            preferred_period = _pick_period(lifetime)
            async with aiohttp.ClientSession() as session:
                # Try preferred, then alternatives
                for p in [preferred_period, 60, 1440, 1]:
                    candles = await fetch(session, ticker, m["event_ticker"],
                                          start_ts, end_ts, p)
                    if candles:
                        cache[ticker] = candles
                        counts["new"] += 1
                        if counts["new"] % 50 == 0:
                            CANDLES_PATH.write_text(json.dumps(cache))
                            print(f"    progress: new={counts['new']} fail={counts['fail']} empty={counts['empty']}")
                        return
                cache[ticker] = []
                counts["empty"] += 1

    await asyncio.gather(*(one(m) for m in meta))
    CANDLES_PATH.write_text(json.dumps(cache))
    print(f"\nDone. {counts}")
    sizes = [len(c) for c in cache.values() if c]
    if sizes:
        sizes.sort()
        print(f"Candles per market: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    asyncio.run(main(args.limit))
