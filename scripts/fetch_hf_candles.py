"""Fetch 1-minute candles for the high-frequency 15-min-crypto universe."""
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
META_PATH = Path("data/hf_universe_meta.json")
CANDLES_PATH = Path("data/hf_candles.json")


def _to_ts(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


async def fetch_candles(session: aiohttp.ClientSession, ticker: str,
                        event_ticker: str, start_ts: int, end_ts: int) -> list[dict] | None:
    series = event_ticker.split("-", 1)[0]
    url = (f"{BASE}/series/{series}/markets/{ticker}/candlesticks"
           f"?period_interval=1&start_ts={start_ts}&end_ts={end_ts}")

    async def _do() -> list[dict]:
        async with session.get(url) as r:
            if r.status >= 500:
                r.raise_for_status()
            if r.status >= 400:
                return []
            data = await r.json()
            return data.get("candlesticks", [])

    try:
        return await retry_with_backoff(_do, max_attempts=3, base_delay=1.5,
                                        label=f"hfcandles:{ticker}")
    except Exception as e:
        logger.warning("HF candle fetch failed", extra={"ticker": ticker, "error": str(e)})
        return None


async def main(limit: int | None = None) -> None:
    meta = json.loads(META_PATH.read_text())
    if limit:
        meta = meta[:limit]
    print(f"Fetching 1-min candles for {len(meta)} HF markets...")

    cache: dict = {}
    if CANDLES_PATH.exists():
        cache = json.loads(CANDLES_PATH.read_text())
        print(f"  Cache hit: {len(cache)} markets already fetched")

    sem = asyncio.Semaphore(4)
    counts = {"new": 0, "skip": 0, "fail": 0, "empty": 0}

    async def one(m: dict) -> None:
        ticker = m["ticker"]
        if ticker in cache and cache[ticker]:
            counts["skip"] += 1
            return
        async with sem:
            await asyncio.sleep(0.25)
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
            if counts["new"] % 50 == 0:
                CANDLES_PATH.write_text(json.dumps(cache))
                print(f"    progress: new={counts['new']} fail={counts['fail']} empty={counts['empty']}")

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
