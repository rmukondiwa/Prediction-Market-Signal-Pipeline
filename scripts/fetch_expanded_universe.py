"""
Fetch settled-market metadata for the EXPANDED universe (daily crypto,
hourly crypto, hourly weather) over the last 90 days.

Targets the highest-volume series identified in the survey:
  - KXBTCD, KXETHD, KXSOLD, KXXRPD, KXDOGED  (daily crypto)
  - KXBTC, KXETH, KXDOGE                      (hourly crypto)
  - KXHIGHT* / KXLOWT* across major US cities (hourly weather)

Writes data/expanded_universe_meta.json. Skips parlay markets and applies
a minimum-volume filter to keep the universe tradeable.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from pathlib import Path

import aiohttp

from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
META_PATH = Path("data/expanded_universe_meta.json")

DAILY_CRYPTO = ["KXBTCD", "KXETHD", "KXSOLD", "KXXRPD", "KXDOGED"]
HOURLY_CRYPTO = ["KXBTC", "KXETH", "KXDOGE"]  # exclude 15M variants
WEATHER_HIGH = [f"KXHIGHT{c}" for c in ["LAX", "SFO", "SEA", "PHX", "LV", "DEN",
                                        "CHI", "AUS", "SATX", "OKC", "NOLA", "MIN",
                                        "HOU", "DAL", "BOS", "ATL", "DC"]]
WEATHER_LOW = [f"KXLOWT{c}" for c in ["LAX", "SFO", "SEA", "PHX", "LV", "DEN",
                                      "CHI", "AUS", "NYC", "MIA", "PHIL"]]
TARGETS = set(DAILY_CRYPTO + HOURLY_CRYPTO + WEATHER_HIGH + WEATHER_LOW)


async def fetch_window(session: aiohttp.ClientSession, start_ts: int, end_ts: int) -> list[dict]:
    """Pull all settled markets in a time window, paginating."""
    out = []
    cursor = None
    for _ in range(20):
        url = f"{BASE}/markets?status=settled&limit=1000&min_close_ts={start_ts}&max_close_ts={end_ts}"
        if cursor:
            url += f"&cursor={cursor}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status >= 400:
                    return out
                data = await r.json()
        except Exception as e:
            logger.warning("Window fetch failed", extra={"error": str(e)})
            return out
        out.extend(data.get("markets", []))
        cursor = data.get("cursor", "")
        if not cursor:
            break
    return out


async def main(days: int = 90, min_vol: float = 30.0) -> None:
    now = int(time.time())
    print(f"Pulling {days} days of settled markets, filtering to {len(TARGETS)} target series...")

    all_markets: list[dict] = []
    async with aiohttp.ClientSession() as session:
        # Run windows in parallel (but with concurrency cap)
        sem = asyncio.Semaphore(3)

        async def one(start_ts: int, end_ts: int):
            async with sem:
                await asyncio.sleep(0.1)
                ms = await fetch_window(session, start_ts, end_ts)
                return ms

        tasks = []
        for window_start in range(now - days*86400, now, 7*86400):
            window_end = window_start + 7*86400
            tasks.append(one(window_start, window_end))

        results = await asyncio.gather(*tasks)
        for r in results:
            all_markets.extend(r)

    print(f"  Raw fetched: {len(all_markets)}")

    # Filter to target series with sufficient volume
    target_markets = []
    for m in all_markets:
        ev = m.get('event_ticker', '')
        prefix = ev.split('-')[0]
        if prefix not in TARGETS:
            continue
        try:
            vol = float(m.get('volume_fp', 0))
        except (ValueError, TypeError):
            vol = 0
        if vol < min_vol:
            continue
        target_markets.append(m)

    # Deduplicate by ticker
    by_ticker = {m['ticker']: m for m in target_markets}
    target_markets = list(by_ticker.values())

    print(f"  After filter (target series, vol≥{min_vol}): {len(target_markets)}")

    # Per-series stats
    from collections import Counter
    by_series = Counter(m['event_ticker'].split('-')[0] for m in target_markets)
    print(f"\nPer series:")
    for s, c in by_series.most_common():
        print(f"  {s:25} {c}")

    META_PATH.write_text(json.dumps(target_markets))
    print(f"\nSaved {len(target_markets)} markets to {META_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--min-vol", type=float, default=30.0)
    args = p.parse_args()
    asyncio.run(main(args.days, args.min_vol))
