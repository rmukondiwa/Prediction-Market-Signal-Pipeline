"""
Forward-test the SettlementDecaySignal on live Kalshi markets — paper only.

Polling loop:
    1. Pull all open markets from configured target series
    2. For each, compute time-to-close and check signal threshold
    3. If signal fires, compute depth-aware size and place a synthetic fill
    4. Track positions to settlement and record realized P&L
    5. Append every event to a daily JSONL log + maintain a Redis-backed
       (or in-memory) portfolio state

Sizing strategy:
    - Per-trade USD = min(bankroll × max_fraction,
                         depth_cap × current_ask)
    - depth_cap = floor(yes_ask_size_fp × 0.25)   # take ≤25% of inside depth
    - Caps: max_fraction=0.10 (10% of bankroll), max_per_market_usd cap from RiskLimits

Public Kalshi endpoints only — no auth needed for read. No real orders are
placed. KalshiTradingClientStub records every "would-place" intent for
post-run audit.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import time
import uuid
from pathlib import Path

import aiohttp

from src.execution.kalshi_trading_client import KalshiTradingClientStub
from src.execution.models import OrderRequest
from src.portfolio.models import Fill, RiskLimits, WorkingOrder
from src.portfolio.state import InMemoryBackend, PortfolioState
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXDOGE15M",
                  "KXBNB15M", "KXXRP15M", "KXHYPE15M"]


def parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


async def fetch_open_markets(session: aiohttp.ClientSession,
                             target_series: set[str],
                             pages: int = 30) -> list[dict]:
    """Find open markets in target series. Uses per-series event lookup
    (much faster than paginating the entire 15k+ open universe)."""
    out: list[dict] = []
    for series in target_series:
        # /events?series_ticker=X returns events in this series
        url = f"{BASE}/events?series_ticker={series}&status=open&limit=200"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status >= 400:
                    continue
                data = await r.json()
        except Exception as e:
            logger.warning("Series event fetch failed",
                           extra={"series": series, "error": str(e)})
            continue
        events = data.get("events", [])
        for e in events:
            # Each event embeds its markets in the response
            for m in e.get("markets", []) or []:
                if m.get("status") in {"open", "active"}:
                    out.append(m)
            # /events list doesn't include markets — fetch each event's detail
            if not e.get("markets"):
                ev_ticker = e.get("event_ticker", "")
                if not ev_ticker:
                    continue
                try:
                    url2 = f"{BASE}/events/{ev_ticker}"
                    async with session.get(url2, timeout=aiohttp.ClientTimeout(total=10)) as r2:
                        if r2.status >= 400:
                            continue
                        ed = await r2.json()
                    # Markets live at the TOP LEVEL of the response, not nested
                    # under "event" — this took a debug session to figure out
                    for m in ed.get("markets", []) or []:
                        if m.get("status") in {"open", "active"}:
                            out.append(m)
                    # Belt-and-braces: also check the embedded version in case
                    # the API ever changes
                    for m in ed.get("event", {}).get("markets", []) or []:
                        if m.get("status") in {"open", "active"}:
                            out.append(m)
                except Exception:
                    pass
    return out


def market_quote(m: dict) -> tuple[float, float, float] | None:
    """Return (yes_bid, yes_ask, ask_size_contracts) — None on bad data."""
    try:
        bid = float(m.get("yes_bid_dollars", 0) or 0)
        ask = float(m.get("yes_ask_dollars", 0) or 0)
        ask_sz = float(m.get("yes_ask_size_fp", 0) or 0)
    except (ValueError, TypeError):
        return None
    if bid <= 0 or ask <= 0 or bid > ask:
        return None
    return bid, ask, ask_sz


def compute_size(bankroll: float, max_fraction: float,
                 ask_price: float, ask_size: float,
                 depth_take_fraction: float = 0.25,
                 max_contracts_per_order: int = 5_000,
                 max_per_market_usd: float = 500.0) -> int:
    """Depth-aware sizing: smaller of (bankroll fraction, depth cap, hard cap)."""
    if ask_price <= 0:
        return 0
    by_bankroll = (bankroll * max_fraction) / ask_price
    by_depth = ask_size * depth_take_fraction
    by_hard = max_per_market_usd / ask_price
    contracts = int(min(by_bankroll, by_depth, by_hard, max_contracts_per_order))
    return max(0, contracts)


class DecayPaperTrader:
    def __init__(self, target_series: set[str], state: PortfolioState,
                 client: KalshiTradingClientStub,
                 min_implied: float, max_minutes: float,
                 max_fraction: float, log_path: Path,
                 starting_bankroll: float):
        self.targets = target_series
        self.state = state
        self.client = client
        self.min_implied = min_implied
        self.max_minutes = max_minutes
        self.max_fraction = max_fraction
        self.log = log_path
        self.starting_bankroll = starting_bankroll
        self.session = None
        self.last_seen_signals: dict[str, dt.datetime] = {}
        self._log_lock = asyncio.Lock()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def _emit(self, event: dict) -> None:
        event = {"ts": dt.datetime.utcnow().isoformat(), **event}
        async with self._log_lock:
            with self.log.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    async def tick(self) -> dict:
        """One scan of all open markets in target series."""
        open_markets = await fetch_open_markets(self.session, self.targets)
        now = dt.datetime.now(dt.timezone.utc)
        cash = await self.state.get_cash()

        decisions = 0
        fills = 0
        for m in open_markets:
            ticker = m["ticker"]
            try:
                close_t = parse_iso(m["close_time"])
            except Exception:
                continue
            mins_to_close = (close_t - now).total_seconds() / 60.0
            if mins_to_close <= 0 or mins_to_close > self.max_minutes:
                continue
            quote = market_quote(m)
            if quote is None:
                continue
            bid, ask, ask_sz = quote
            mid = (bid + ask) / 2
            if mid < self.min_implied:
                continue
            # Don't double-fill the same market within the same window
            last = self.last_seen_signals.get(ticker)
            if last and (now - last).total_seconds() < 60:
                continue

            decisions += 1
            contracts = compute_size(
                bankroll=cash, max_fraction=self.max_fraction,
                ask_price=ask, ask_size=ask_sz,
            )
            if contracts <= 0:
                await self._emit({"event": "signal_skipped", "ticker": ticker,
                                  "reason": "zero_size", "ask": ask, "ask_size": ask_sz})
                self.last_seen_signals[ticker] = now
                continue

            # Place a paper order via the stub (records intent)
            req = OrderRequest(
                ticker=ticker, side="yes", contracts=contracts,
                order_type="limit", limit_price=ask,
                time_in_force="ioc",
                client_order_id=f"decay-{uuid.uuid4().hex[:10]}",
                placed_at=now,
            )
            result = await self.client.place_order(req)
            if not result.accepted:
                continue

            # Synthetic fill at the ask
            fill = Fill(
                fill_id=f"paper-{uuid.uuid4().hex[:10]}",
                order_id=result.order_id,
                ticker=ticker, side="yes",
                contracts=contracts, price=ask,
                fee=round(0.07 * contracts * ask * (1 - ask), 4),
                timestamp=now, signal_model="settlement_decay",
            )
            await self.state.apply_fill(fill)
            fills += 1
            self.last_seen_signals[ticker] = now
            await self._emit({"event": "fill", "ticker": ticker,
                              "contracts": contracts, "price": ask,
                              "implied_at_entry": mid,
                              "minutes_to_close": mins_to_close,
                              "ask_size": ask_sz, "fee": fill.fee,
                              "title": m.get("title", "")[:60]})

        return {"open_markets": len(open_markets), "decisions": decisions,
                "fills": fills, "cash": cash}

    async def settle_finished(self) -> int:
        """Check positions whose markets have closed; settle them."""
        positions = await self.state.list_positions()
        if not positions:
            return 0
        now = dt.datetime.now(dt.timezone.utc)
        settled = 0
        async with aiohttp.ClientSession() as session:
            for pos in positions:
                # GET /markets/{ticker}
                try:
                    async with session.get(f"{BASE}/markets/{pos.ticker}",
                                           timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status >= 400:
                            continue
                        data = await r.json()
                except Exception:
                    continue
                m = data.get("market", {})
                status = m.get("status", "")
                if status not in {"settled", "finalized"}:
                    continue
                result = (m.get("result") or "").lower()
                settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
                if settlement_value is None:
                    continue
                pnl = await self.state.settle(pos.ticker, settlement_value, now)
                settled += 1
                await self._emit({"event": "settled", "ticker": pos.ticker,
                                  "outcome": result, "pnl": round(pnl, 4),
                                  "side": pos.side, "contracts": pos.contracts,
                                  "avg_cost": pos.avg_cost})
        return settled

    async def run(self, tick_seconds: int = 30) -> None:
        await self._emit({"event": "start",
                          "starting_bankroll": self.starting_bankroll,
                          "targets": sorted(self.targets),
                          "min_implied": self.min_implied,
                          "max_minutes": self.max_minutes,
                          "max_fraction": self.max_fraction,
                          "tick_seconds": tick_seconds})
        while True:
            t0 = time.perf_counter()
            try:
                stats = await self.tick()
                settled = await self.settle_finished()
                snap = await self.state.snapshot()
                await self._emit({
                    "event": "tick_summary",
                    **stats, "settled": settled,
                    "positions_open": len(snap.positions),
                    "realized_pnl": round(snap.realized_pnl, 4),
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                })
            except Exception as e:
                await self._emit({"event": "tick_error", "error": str(e)})
                logger.exception("tick failed")
            await asyncio.sleep(tick_seconds)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--series", type=str, default=",".join(DEFAULT_SERIES),
                   help="Comma-separated event series prefixes")
    p.add_argument("--bankroll", type=float, default=5_000.0)
    p.add_argument("--min-implied", type=float, default=0.80)
    p.add_argument("--max-minutes", type=float, default=15.0)
    p.add_argument("--max-fraction", type=float, default=0.10,
                   help="Max bankroll fraction per trade")
    p.add_argument("--tick-seconds", type=int, default=30)
    p.add_argument("--log", type=Path, default=Path("logs/paper_decay.jsonl"))
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    target_series = set(s.strip() for s in args.series.split(",") if s.strip())

    backend = InMemoryBackend()
    state = PortfolioState(backend, env=f"paper_decay")
    await state.initialize(args.bankroll)
    client = KalshiTradingClientStub()

    async with DecayPaperTrader(
        target_series=target_series, state=state, client=client,
        min_implied=args.min_implied, max_minutes=args.max_minutes,
        max_fraction=args.max_fraction, log_path=args.log,
        starting_bankroll=args.bankroll,
    ) as trader:
        await trader.run(tick_seconds=args.tick_seconds)


if __name__ == "__main__":
    asyncio.run(main())
