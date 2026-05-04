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
                 max_per_market_usd: float = 500.0,
                 existing_market_notional: float = 0.0) -> int:
    """Depth-aware sizing: smallest of (bankroll fraction, depth cap, hard cap,
    remaining per-market headroom).

    `existing_market_notional` is the dollar value of all open fills on this
    ticker (sum of contracts × avg_cost). The per-market cap subtracts this
    so trail_up across multiple thresholds can't aggregate past the cap.
    Yesterday's HYPE 117ct fill was caused by this missing aggregation —
    each individual fill respected the 5% cap but they all hit on the same
    market in one tick.
    """
    if ask_price <= 0:
        return 0
    by_bankroll = (bankroll * max_fraction) / ask_price
    by_depth = ask_size * depth_take_fraction
    remaining_market_usd = max(0.0, max_per_market_usd - existing_market_notional)
    by_market = remaining_market_usd / ask_price
    contracts = int(min(by_bankroll, by_depth, by_market, max_contracts_per_order))
    return max(0, contracts)


class DecayPaperTrader:
    def __init__(self, target_series: set[str], state: PortfolioState,
                 client: KalshiTradingClientStub,
                 yes_thresholds: list[float], no_thresholds: list[float],
                 max_minutes: float, max_fraction: float,
                 log_path: Path, starting_bankroll: float,
                 daily_loss_limit_usd: float | None = None):
        self.targets = target_series
        self.state = state
        self.client = client
        # trail_up: enter at each threshold step, once per side per market
        self.yes_thresholds = sorted(yes_thresholds)
        self.no_thresholds = sorted(no_thresholds)
        self.max_minutes = max_minutes
        self.max_fraction = max_fraction
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.kill_switch_tripped = False
        self.log = log_path
        self.starting_bankroll = starting_bankroll
        self.session = None
        # per-market threshold-hit state: {ticker: {"yes": set, "no": set}}
        self.hit_thresholds: dict[str, dict[str, set]] = {}
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

    async def _maybe_fill(self, ticker: str, m: dict, side: str, mid: float,
                          fill_price: float, ask_sz: float, mins_to_close: float,
                          now: dt.datetime, threshold: float, cash: float,
                          existing_market_notional: float = 0.0) -> bool:
        """Place a paper order if depth-aware sizing is positive. Returns True
        on a successful fill."""
        contracts = compute_size(
            bankroll=cash, max_fraction=self.max_fraction,
            ask_price=fill_price, ask_size=ask_sz,
            existing_market_notional=existing_market_notional,
        )
        if contracts <= 0:
            await self._emit({"event": "signal_skipped", "ticker": ticker,
                              "reason": "zero_size", "side": side,
                              "fill_price": fill_price, "ask_size": ask_sz,
                              "existing_notional": round(existing_market_notional, 2),
                              "threshold": threshold})
            return False

        req = OrderRequest(
            ticker=ticker, side=side, contracts=contracts,
            order_type="limit", limit_price=fill_price,
            time_in_force="ioc",
            client_order_id=f"decay-{uuid.uuid4().hex[:10]}",
            placed_at=now,
        )
        result = await self.client.place_order(req)
        if not result.accepted:
            return False
        fill = Fill(
            fill_id=f"paper-{uuid.uuid4().hex[:10]}",
            order_id=result.order_id,
            ticker=ticker, side=side,
            contracts=contracts, price=fill_price,
            fee=round(0.07 * contracts * fill_price * (1 - fill_price), 4),
            timestamp=now, signal_model=f"settlement_decay_trail_{side}@{threshold:.2f}",
        )
        await self.state.apply_fill(fill)
        await self._emit({"event": "fill", "ticker": ticker, "side": side,
                          "contracts": contracts, "price": fill_price,
                          "implied_at_entry": mid, "threshold": threshold,
                          "minutes_to_close": mins_to_close,
                          "ask_size": ask_sz, "fee": fill.fee,
                          "title": m.get("title", "")[:60]})
        return True

    async def tick(self) -> dict:
        """One scan: trail_up entry on YES (each yes threshold) and NO (each no threshold).

        BUG FIX: previously we'd enter NO on a market we already had YES on (and
        vice versa) when implied whipsawed. That hedges against ourselves and
        guarantees a loss. Now we check existing positions per market and skip
        opposite-side entries entirely. The original side keeps trailing as
        more thresholds clear; the opposite side is gated by lock-out."""
        open_markets = await fetch_open_markets(self.session, self.targets)
        now = dt.datetime.now(dt.timezone.utc)

        # Daily loss kill switch — check BEFORE doing anything else
        realized_pnl = await self.state.get_realized_pnl()
        if self.daily_loss_limit_usd is not None and realized_pnl <= -abs(self.daily_loss_limit_usd):
            if not self.kill_switch_tripped:
                await self._emit({
                    "event": "kill_switch_tripped",
                    "reason": "daily_loss_limit_exceeded",
                    "realized_pnl": round(realized_pnl, 4),
                    "limit": self.daily_loss_limit_usd,
                })
                self.kill_switch_tripped = True
            return {"open_markets": len(open_markets), "decisions": 0,
                    "fills": 0, "cash": await self.state.get_cash(),
                    "realized_pnl": round(realized_pnl, 4),
                    "kill_switch": "tripped"}

        cash = await self.state.get_cash()
        # Snapshot positions once per tick (avoids hammering Redis)
        positions = await self.state.list_positions()
        existing_positions = {p.ticker: p.side for p in positions}
        # Per-market notional for exposure-cap aggregation across trail_up steps
        market_notional = {p.ticker: p.contracts * p.avg_cost for p in positions}

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

            hit = self.hit_thresholds.setdefault(ticker, {"yes": set(), "no": set()})
            existing_side = existing_positions.get(ticker)  # None | "yes" | "no"
            current_notional = market_notional.get(ticker, 0.0)

            # YES side: only if we don't already have a NO position on this market
            if existing_side != "no":
                for thr in self.yes_thresholds:
                    if thr in hit["yes"]: continue
                    if mid >= thr:
                        decisions += 1
                        fill_price = min(0.99, ask + 0.005)
                        if await self._maybe_fill(ticker, m, "yes", mid, fill_price,
                                                   ask_sz, mins_to_close, now, thr, cash,
                                                   existing_market_notional=current_notional):
                            fills += 1
                            # Tick-local accumulator so subsequent threshold steps
                            # this same tick see updated per-market notional
                            current_notional += compute_size(
                                bankroll=cash, max_fraction=self.max_fraction,
                                ask_price=fill_price, ask_size=ask_sz,
                                existing_market_notional=current_notional - 0.001,
                            ) * fill_price
                        hit["yes"].add(thr)
            else:
                await self._emit({"event": "signal_skipped", "ticker": ticker,
                                  "reason": "opposite_position_exists",
                                  "side": "yes", "existing_side": "no", "mid": mid})

            # NO side: only if we don't already have a YES position on this market
            if existing_side != "yes":
                for thr in self.no_thresholds:
                    if thr in hit["no"]: continue
                    if mid <= thr:
                        decisions += 1
                        no_ask = 1.0 - bid
                        fill_price = min(0.99, no_ask + 0.005)
                        no_sz = float(m.get("yes_bid_size_fp", 0) or 0)
                        if await self._maybe_fill(ticker, m, "no", mid, fill_price,
                                                   no_sz, mins_to_close, now, thr, cash,
                                                   existing_market_notional=current_notional):
                            fills += 1
                            current_notional += compute_size(
                                bankroll=cash, max_fraction=self.max_fraction,
                                ask_price=fill_price, ask_size=no_sz,
                                existing_market_notional=current_notional - 0.001,
                            ) * fill_price
                        hit["no"].add(thr)
            else:
                await self._emit({"event": "signal_skipped", "ticker": ticker,
                                  "reason": "opposite_position_exists",
                                  "side": "no", "existing_side": "yes", "mid": mid})

        # Pull realized PnL from state — the SOURCE OF TRUTH for session P&L,
        # including any intermediate opposite-side closures that don't show up
        # as `settled` events. Yesterday's audit hallucination was caused by
        # summing settled events instead of reading this back.
        realized_pnl = await self.state.get_realized_pnl()
        return {"open_markets": len(open_markets), "decisions": decisions,
                "fills": fills, "cash": cash, "realized_pnl": round(realized_pnl, 4)}

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
                          "yes_thresholds": self.yes_thresholds,
                          "no_thresholds": self.no_thresholds,
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
    p.add_argument("--series", type=str, default=",".join(DEFAULT_SERIES))
    p.add_argument("--bankroll", type=float, default=5_000.0)
    p.add_argument("--yes-thresholds", type=str, default="0.75,0.85,0.95",
                   help="Trail-up YES entry thresholds (comma-separated)")
    p.add_argument("--no-thresholds", type=str, default="0.25,0.15,0.05",
                   help="Trail-up NO entry thresholds (comma-separated)")
    p.add_argument("--max-minutes", type=float, default=15.0)
    p.add_argument("--max-fraction", type=float, default=0.05,
                   help="Max bankroll fraction per single fill (smaller because trail_up multiplies entries per market)")
    p.add_argument("--daily-loss-limit", type=float, default=300.0,
                   help="Halt new orders if realized PnL drops below -this (USD)")
    p.add_argument("--tick-seconds", type=int, default=30)
    p.add_argument("--log", type=Path, default=Path("logs/paper_decay.jsonl"))
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    target_series = set(s.strip() for s in args.series.split(",") if s.strip())
    yes_thr = sorted(float(x) for x in args.yes_thresholds.split(",") if x.strip())
    no_thr = sorted((float(x) for x in args.no_thresholds.split(",") if x.strip()), reverse=True)

    backend = InMemoryBackend()
    state = PortfolioState(backend, env=f"paper_decay")
    await state.initialize(args.bankroll)
    client = KalshiTradingClientStub()

    async with DecayPaperTrader(
        target_series=target_series, state=state, client=client,
        yes_thresholds=yes_thr, no_thresholds=no_thr,
        max_minutes=args.max_minutes, max_fraction=args.max_fraction,
        log_path=args.log, starting_bankroll=args.bankroll,
        daily_loss_limit_usd=args.daily_loss_limit,
    ) as trader:
        await trader.run(tick_seconds=args.tick_seconds)


if __name__ == "__main__":
    asyncio.run(main())
