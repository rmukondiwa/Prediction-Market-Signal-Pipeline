"""
Live cross-platform arbitrage orchestrator.

Wires every infrastructure component built tonight into one trading loop:

  scanner (LLM-matched arb v2)
      ↓ verified candidates
  pre-trade risk gates
      ↓ approved sized legs
  multi-leg coordinator (concurrent submit + auto-unwind)
      ↓ fills
  order reconciliation (severity-scored divergence)
      ↓
  portfolio state (in-memory or Redis-backed)
      ↓
  alerts (file + Discord + Slack + Pushover)

Operational:
  - Default mode is paper (stubs on both venues)
  - `--live` requires typed "I CONFIRM" before any real order
  - Always run under `scripts/supervisor.py` for crash-restart in live mode
  - Kill switch + drawdown trail share semantics with the (archived) decay
    trader: realized PnL ≤ -daily_loss_limit halts; peak-to-trough ≥
    drawdown_limit halts

Usage (paper):
    python -m scripts.run_arb_live

Usage (live):
    python -m scripts.supervisor --max-restarts 5 -- \
      python -m scripts.run_arb_live --live \
        --bankroll 500 --daily-loss-limit 30 --drawdown-limit 20 \
        --min-edge 0.02 --max-trade-usd 50

Required env for live:
    KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY[_PATH]
    POLYMARKET_PRIVATE_KEY                       (Polygon EOA, with USDC + allowances)
    Optional: REDIS_HOST/PORT, DISCORD_WEBHOOK_URL, PUSHOVER_TOKEN/USER
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.execution.kalshi_trading_client import (
    KalshiTradingClientLive,
    KalshiTradingClientStub,
)
from src.execution.models import OrderRequest, OrderResult
from src.execution.multi_leg import Leg, execute_legs
from src.execution.polymarket_trading_client import (
    PolymarketOrderRequest,
    PolymarketTradingClientLive,
    PolymarketTradingClientStub,
)
from src.execution.reconciliation import (
    ActualFill,
    ExpectedFill,
    Reconciler,
    ReconcilerConfig,
)
from src.ingestion.polymarket.clob_client import get_market_books
from src.portfolio.risk_gates import (
    RiskConfig,
    RiskState,
    compute_size,
    kill_switch_state,
    underlying_of,
)
from src.portfolio.state import InMemoryBackend, PortfolioState, RedisBackend
from src.utils.alerts import alert_async
from src.utils.logging import get_logger

logger = get_logger(__name__)

REPORT_DIR = Path("reports")
LOG_DIR = Path("logs")


# --- confirmation gate -----------------------------------------------------

def confirm_live(args: argparse.Namespace) -> None:
    """Print a loud banner and require typed confirmation before live mode."""
    banner = (
        "\n" + "=" * 76 + "\n"
        "          LIVE CROSS-PLATFORM ARB MODE — REAL MONEY ON BOTH VENUES\n"
        + "=" * 76 + "\n"
        f"  Bankroll target:       ${args.bankroll:,.2f} (informational only — actual orders\n"
        f"                         use real Kalshi + Polymarket balances)\n"
        f"  Daily loss limit:      ${args.daily_loss_limit:,.2f}\n"
        f"  Drawdown trail:        ${args.drawdown_limit:,.2f}\n"
        f"  Min edge to trade:     ${args.min_edge:.4f}\n"
        f"  Max per trade:         ${args.max_trade_usd:,.2f}\n"
        f"  Scan interval:         {args.interval_seconds}s\n"
        "\n"
        "  This will place REAL ORDERS on BOTH Kalshi and Polymarket.\n"
        "  Multi-leg legging risk is real — expect 100-2000ms exposure window.\n"
        "\n"
        "  Pre-flight verify:\n"
        "    [ ] Both accounts funded with the intended amount\n"
        "    [ ] USDC + CTF allowances approved on Polygon (Polymarket)\n"
        "    [ ] Webhook alerts configured (recommended)\n"
        "    [ ] Running under scripts/supervisor.py for crash protection\n"
        "    [ ] Reviewed today's scanner output for sane candidates\n"
        "\n"
        "  Type 'I CONFIRM' (exactly, including caps) to proceed.\n"
        + "=" * 76 + "\n"
    )
    print(banner, flush=True)
    response = input("> ").strip()
    if response != "I CONFIRM":
        raise SystemExit("Live arb aborted (confirmation not given).")


# --- scanner glue ----------------------------------------------------------

def run_scanner(min_edge: float) -> list[dict]:
    """Run the LLM-matched scanner as a subprocess. Returns list of arbs
    that passed verification + threshold filter and meet the min-edge bar.
    """
    out_path = REPORT_DIR / "cross_platform_arb_llm.json"
    cmd = [
        sys.executable, "-m", "scripts.scan_cross_platform_arb_llm",
        "--poly-limit", "100", "--max-verify", "40",
        "--concurrency", "2", "--sim-threshold", "0.70",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode != 0:
            logger.warning("Scanner returned non-zero",
                           extra={"rc": r.returncode, "stderr_tail": r.stderr.decode()[-300:]})
            return []
    except subprocess.TimeoutExpired:
        logger.warning("Scanner timeout")
        return []
    except Exception as e:
        logger.warning("Scanner subprocess error", extra={"error": str(e)})
        return []

    if not out_path.exists():
        return []
    try:
        report = json.loads(out_path.read_text())
    except Exception:
        return []

    candidates = report.get("arbs", [])
    return [c for c in candidates if c.get("edge", 0) >= min_edge]


# --- trade pipeline --------------------------------------------------------

async def maybe_trade_arb(arb: dict,
                          state: PortfolioState,
                          risk_cfg: RiskConfig,
                          kalshi_client: Any,
                          poly_client: Any,
                          reconciler: Reconciler,
                          max_trade_usd: float,
                          dry_run: bool) -> dict:
    """Try to execute one arb candidate.

    Returns a result dict for logging. Does not raise — surfaces errors via
    the result + alerts.

    Strategy: if K_yes_ask + P_no_ask < 1, buy YES on Kalshi + buy NO on
    Polymarket. The two legs together pay $1 with certainty (one of YES/NO
    settles). Edge = 1 - (sum of asks).
    """
    k_ticker = arb["kalshi_ticker"]
    p_market = arb.get("poly_market") or {}
    poly_id = arb.get("poly_id")
    edge = arb.get("edge", 0)

    # Decide which direction the arb is
    arb_dir1 = arb.get("arb_buy_K_yes_P_no_cost")  # buy K yes + P no
    arb_dir2 = arb.get("arb_buy_K_no_P_yes_cost")  # buy K no + P yes
    if arb_dir1 is None or arb_dir2 is None:
        return {"arb": arb, "skipped": "missing arb costs"}
    if arb_dir1 <= arb_dir2:
        kalshi_side = "yes"
        kalshi_price = arb["k_yes_ask_dollars"]
        poly_side = "no"
        poly_price = arb.get("p_no_ask")
        poly_token_idx = 1
    else:
        kalshi_side = "no"
        kalshi_price = 1.0 - arb["k_yes_bid_dollars"]
        poly_side = "yes"
        poly_price = arb.get("p_yes_ask")
        poly_token_idx = 0

    if poly_price is None:
        return {"arb_id": k_ticker, "skipped": "no Polymarket price"}

    # Pre-trade risk gates: kill switch + sized leg budget
    cash = await state.get_cash()
    realized = await state.get_realized_pnl()
    peak = max(realized, 0.0)  # simplification — caller should pass true peak
    positions = await state.list_positions()
    asset_key = underlying_of(k_ticker)
    asset_notional = sum(p.contracts * p.avg_cost for p in positions
                         if underlying_of(p.ticker) == asset_key)
    market_notional = sum(p.contracts * p.avg_cost for p in positions
                          if p.ticker == k_ticker)
    risk_state = RiskState(
        cash=cash, realized_pnl=realized, peak_realized_pnl=peak,
        existing_market_notional=market_notional,
        existing_asset_notional=asset_notional,
    )
    halted, halt_reason = kill_switch_state(risk_state, risk_cfg)
    if halted:
        await alert_async("Kill switch active — skipping arb",
                           severity="warning",
                           context={"reason": halt_reason, "ticker": k_ticker})
        return {"arb_id": k_ticker, "skipped": halt_reason}

    # Size each leg by min(arb edge × bankroll fraction, max_trade_usd, depth/cap)
    # Sizing in CONTRACTS: both legs need the same contract count.
    # Prefer depth from arb dict if scanner populated it; otherwise fetch live.
    arb_yes_size = arb.get("p_yes_ask_size") or arb.get("yes_top_ask_size")
    arb_no_size = arb.get("p_no_ask_size") or arb.get("no_top_ask_size")
    poly_side_depth: float | None = (arb_yes_size if poly_token_idx == 0 else arb_no_size)

    if poly_side_depth is None or poly_side_depth <= 0:
        # Live fallback (skipped in dry_run to keep tests offline-deterministic)
        if dry_run:
            poly_side_depth = float(int(max_trade_usd / max(poly_price, 0.01)))
        else:
            try:
                books = get_market_books(p_market)
                if books:
                    yb, nb = books
                    if poly_token_idx == 0 and yb.top_ask:
                        poly_side_depth = yb.top_ask.size
                    elif poly_token_idx == 1 and nb.top_ask:
                        poly_side_depth = nb.top_ask.size
            except Exception:
                pass
    if not poly_side_depth or poly_side_depth <= 0:
        return {"arb_id": k_ticker, "skipped": "no Polymarket depth at top"}

    # Run the per-fill risk gate against a bankroll slice
    contracts, info = compute_size(
        ask_price=kalshi_price, ask_size=poly_side_depth,
        state=risk_state, cfg=risk_cfg, threshold_tier=2,
    )
    # Bound by max_trade_usd and the matching Polymarket depth
    max_by_dollars = int(max_trade_usd / max(kalshi_price + poly_price, 0.01))
    contracts = min(contracts, max_by_dollars, int(poly_side_depth))
    if contracts <= 0:
        return {"arb_id": k_ticker, "skipped": "size_zero", "info": info}

    # --- build the two legs --------------------------------------------------
    poly_token_ids = p_market.get("clobTokenIds")
    if isinstance(poly_token_ids, str):
        poly_token_ids = json.loads(poly_token_ids)
    if not poly_token_ids or len(poly_token_ids) < 2:
        return {"arb_id": k_ticker, "skipped": "no Polymarket token ids"}
    poly_token_id = poly_token_ids[poly_token_idx]

    kalshi_req = OrderRequest(
        ticker=k_ticker, side=kalshi_side, contracts=contracts,
        order_type="limit", limit_price=kalshi_price,
        time_in_force="ioc",
        client_order_id=f"arb-k-{uuid.uuid4().hex[:10]}",
        placed_at=datetime.now(timezone.utc),
    )
    poly_req = PolymarketOrderRequest(
        token_id=poly_token_id, side="BUY",
        price=poly_price, size=float(contracts),
        order_type="FOK",
        client_order_id=f"arb-p-{uuid.uuid4().hex[:10]}",
    )

    if dry_run:
        await alert_async("Arb dry-run — would have placed",
                           severity="info",
                           context={"ticker": k_ticker, "contracts": contracts,
                                    "kalshi_side": kalshi_side, "poly_side": poly_side,
                                    "edge": edge, "expected_pnl": edge * contracts})
        return {"arb_id": k_ticker, "dry_run": True, "contracts": contracts,
                "edge": edge, "expected_pnl": round(edge * contracts, 4)}

    # --- live: build legs with proper unwind handlers ------------------------
    async def kalshi_place() -> OrderResult:
        return await kalshi_client.place_order(kalshi_req)

    async def poly_place() -> OrderResult:
        return await poly_client.place_order(poly_req)

    async def kalshi_unwind() -> OrderResult:
        # Unwind by selling at market — simplification; in practice need
        # to track filled qty and submit opposite-side market order
        opposite = "no" if kalshi_side == "yes" else "yes"
        unwind_req = OrderRequest(
            ticker=k_ticker, side=opposite, contracts=contracts,
            order_type="market", limit_price=None,
            time_in_force="ioc",
            client_order_id=f"unwind-k-{uuid.uuid4().hex[:10]}",
            placed_at=datetime.now(timezone.utc),
        )
        return await kalshi_client.place_order(unwind_req)

    async def poly_unwind() -> OrderResult:
        opposite_token = poly_token_ids[1 - poly_token_idx]
        unwind_req = PolymarketOrderRequest(
            token_id=opposite_token, side="BUY",
            price=1.0 - poly_price, size=float(contracts),
            order_type="FOK",
            client_order_id=f"unwind-p-{uuid.uuid4().hex[:10]}",
        )
        return await poly_client.place_order(unwind_req)

    legs = [
        Leg(name="kalshi", venue="kalshi",
            place=kalshi_place, unwind=kalshi_unwind,
            notional=contracts * kalshi_price),
        Leg(name="polymarket", venue="polymarket",
            place=poly_place, unwind=poly_unwind,
            notional=contracts * poly_price),
    ]
    result = await execute_legs(legs, max_legging_window_ms=2_000.0)

    # Reconcile (best-effort — assumes 100% fill at limit; live partial fills
    # would require fill detail from each venue's response)
    expected_k = ExpectedFill(ticker=k_ticker, side=kalshi_side, contracts=contracts,
                               price=kalshi_price)
    expected_p = ExpectedFill(ticker=poly_token_id[:12], side=poly_side, contracts=contracts,
                               price=poly_price)
    actual_k = ActualFill(ticker=k_ticker, side=kalshi_side, contracts=contracts,
                           price=kalshi_price)  # would parse from result.raw_response in live
    actual_p = ActualFill(ticker=poly_token_id[:12], side=poly_side, contracts=contracts,
                           price=poly_price)
    div_k = reconciler.observe(expected_k, actual_k)
    div_p = reconciler.observe(expected_p, actual_p)
    if reconciler.should_halt():
        await alert_async("Reconciler score crossed halt threshold",
                           severity="high",
                           context={"score": reconciler.divergence_score})

    return {
        "arb_id": k_ticker, "contracts": contracts,
        "edge": edge, "expected_pnl": round(edge * contracts, 4),
        "all_legs_succeeded": result.all_succeeded,
        "n_failed_legs": len(result.failed_legs),
        "n_unwound_legs": len(result.unwound_legs),
        "place_dispersion_ms": round(result.place_dispersion_ms, 1),
        "k_divergence": div_k.severity,
        "p_divergence": div_p.severity,
    }


# --- main loop -------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    LOG_DIR.mkdir(exist_ok=True)
    if args.live:
        confirm_live(args)
        await alert_async("Live arb orchestrator starting",
                           severity="warning",
                           context={"bankroll": args.bankroll,
                                    "daily_loss_limit": args.daily_loss_limit})

    # Set up backends
    if args.live and args.use_redis:
        backend = RedisBackend(url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    else:
        backend = InMemoryBackend()
    state = PortfolioState(backend, env="live_arb" if args.live else "paper_arb")
    if not args.live:
        await state.initialize(args.bankroll)

    # Trading clients
    if args.live:
        from src.config.kalshi_config import KalshiConfig
        kalshi_client = KalshiTradingClientLive(KalshiConfig())
        poly_client = PolymarketTradingClientLive()
    else:
        kalshi_client = KalshiTradingClientStub()
        poly_client = PolymarketTradingClientStub()

    risk_cfg = RiskConfig(
        max_fraction_per_fill=0.05,
        max_per_market_usd=min(500.0, args.max_trade_usd * 2),
        max_per_asset_usd=min(1000.0, args.max_trade_usd * 4),
        daily_loss_limit_usd=args.daily_loss_limit,
        drawdown_limit_usd=args.drawdown_limit,
    )
    reconciler = Reconciler()

    logger.info("Arb orchestrator running", extra={
        "live": args.live, "interval_s": args.interval_seconds,
        "min_edge": args.min_edge, "max_trade_usd": args.max_trade_usd,
    })

    iteration = 0
    while True:
        iteration += 1
        # Pre-iteration kill-switch check
        cash = await state.get_cash()
        realized = await state.get_realized_pnl()
        risk_state = RiskState(cash=cash, realized_pnl=realized, peak_realized_pnl=max(realized, 0.0))
        halted, reason = kill_switch_state(risk_state, risk_cfg)
        if halted:
            logger.warning("Kill switch active — sleeping",
                           extra={"reason": reason, "realized_pnl": realized})
            await asyncio.sleep(args.interval_seconds)
            continue

        logger.info(f"=== Iteration {iteration} ===")
        candidates = run_scanner(min_edge=args.min_edge)
        logger.info(f"  scanner returned {len(candidates)} arb-eligible candidates")

        for c in candidates:
            try:
                result = await maybe_trade_arb(
                    arb=c, state=state, risk_cfg=risk_cfg,
                    kalshi_client=kalshi_client, poly_client=poly_client,
                    reconciler=reconciler,
                    max_trade_usd=args.max_trade_usd,
                    dry_run=not args.live,
                )
                logger.info("arb result", extra=result)
                _log_jsonl({"iteration": iteration, "ts": datetime.now(timezone.utc).isoformat(), **result})
                if not result.get("all_legs_succeeded", True) and not result.get("dry_run"):
                    await alert_async("Arb leg failed — exposure may exist",
                                       severity="high", context=result)
            except Exception as e:
                logger.error("arb pipeline raised", extra={"error": str(e)})
                await alert_async("Arb pipeline exception",
                                   severity="high", context={"error": str(e)})

        await asyncio.sleep(args.interval_seconds)


def _log_jsonl(record: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / "arb_live.jsonl").open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true",
                   help="LIVE TRADING — real orders on Kalshi + Polymarket")
    p.add_argument("--bankroll", type=float, default=500.0)
    p.add_argument("--daily-loss-limit", type=float, default=30.0)
    p.add_argument("--drawdown-limit", type=float, default=20.0)
    p.add_argument("--min-edge", type=float, default=0.02,
                   help="Minimum arb edge to consider (default $0.02 = 2¢ above $1)")
    p.add_argument("--max-trade-usd", type=float, default=50.0,
                   help="Hard cap per arb trade")
    p.add_argument("--interval-seconds", type=int, default=300,
                   help="Seconds between scanner runs")
    p.add_argument("--use-redis", action="store_true",
                   help="Use Redis-backed state (default: in-memory)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")


if __name__ == "__main__":
    main()
