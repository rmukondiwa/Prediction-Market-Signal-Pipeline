"""
Paper trading loop — entry point for Stage 9.

Usage:
    python -m scripts.paper_trade --signal-model raw_llm --interval 1h
    python -m scripts.paper_trade --signal-model consistency_arb --once

Loop sketch (one tick per --interval):
    1. For each ticker on the watchlist:
       a. Extract live market snapshot (Redis or catalog fallback)
       b. Run retrieve + rerank + run_inference (live, no cache)
       c. Apply chosen SignalModel → CalibratedEdge list
       d. Hand to PaperExecutor → risk → sizer → order → simulated fill
    2. Mark portfolio to market
    3. Log summary

Only runs against the stub trading client. Real Kalshi integration is gated
on Stage 10 — DO NOT add `LiveExecutor` to this script.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.execution.kalshi_trading_client import KalshiTradingClientStub
from src.execution.paper_executor import PaperExecutor
from src.portfolio.models import RiskLimits
from src.portfolio.state import InMemoryBackend, PortfolioState
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 paper-trading loop")
    p.add_argument("--signal-model", type=str, default="raw_llm")
    p.add_argument("--watchlist", type=str, required=False,
                   help="Comma-separated tickers; defaults to KALSHI_MARKET_TICKERS env var")
    p.add_argument("--interval", type=str, default="1h",
                   help="Inference cadence (e.g. 1h, 30m). --once overrides.")
    p.add_argument("--once", action="store_true", help="Run a single tick then exit")
    p.add_argument("--starting-capital", type=float, default=10_000.0)
    return p.parse_args()


def _interval_seconds(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("s"):
        return int(s[:-1])
    raise ValueError(f"Unsupported interval: {s!r}")


async def _build_signal_model(name: str):
    if name == "raw_llm":
        from src.signals.raw_llm import RawLLMSignal
        return RawLLMSignal()
    if name == "consistency_arb":
        from src.signals.consistency_arb import ConsistencyArbSignal
        return ConsistencyArbSignal()
    if name == "calibrated_llm":
        from src.signals.calibrated_llm import CalibratedLLMSignal
        return CalibratedLLMSignal(Path("data/calibration_map.pkl"))
    if name == "bayesian_base_rate":
        from src.signals.bayesian_base_rate import BayesianBaseRateSignal
        return BayesianBaseRateSignal()
    if name == "coherence_regression":
        from src.signals.coherence_regression import CoherenceRegressionSignal
        return CoherenceRegressionSignal()
    raise ValueError(f"Unknown signal model: {name!r}")


async def _run_once(args, executor: PaperExecutor, model) -> None:
    """One inference cycle. Reuses the existing infer.py path for retrieval +
    inference. Wraps in try/except so a single bad ticker doesn't kill the loop."""
    from src.catalog.store import load_catalog
    from src.catalog.vector_store import VectorStore
    from src.context.reranker import rerank
    from src.context.retriever import retrieve_candidates
    from src.inference.engine import run_inference
    from src.signals.models import HistoricalContext

    catalog_path = Path("data/catalog.json")
    vectors_path = "data/vectors"
    if not catalog_path.exists() or not Path(f"{vectors_path}.faiss").exists():
        logger.error("Missing index artifacts. Run: python -m scripts.build_index")
        return

    catalog = load_catalog(catalog_path)
    vs = VectorStore()
    vs.load(vectors_path)

    watchlist_str = args.watchlist or os.getenv("KALSHI_MARKET_TICKERS", "")
    tickers = [t.strip() for t in watchlist_str.split(",") if t.strip()]
    if not tickers:
        logger.error("No watchlist provided; set --watchlist or KALSHI_MARKET_TICKERS")
        return

    history = HistoricalContext()

    for ticker in tickers:
        try:
            focus = next((m for m in catalog if m.ticker == ticker), None)
            if focus is None:
                logger.warning("Ticker not in catalog, skipping", extra={"ticker": ticker})
                continue

            from src.insight.models import MarketSnapshot
            snapshot = MarketSnapshot(
                event=focus.title, market=focus.ticker, outcome="YES",
                quoted_price=int((focus.yes_bid + focus.yes_ask) / 2),
                implied_probability=focus.implied_probability,
                yes_bid=focus.yes_bid, yes_ask=focus.yes_ask,
                volume=0, open_interest=0, source="catalog",
                timestamp=datetime.now(timezone.utc),
            )

            candidates = retrieve_candidates(ticker, catalog, vs, k=20)
            context = rerank(focus, candidates)
            llm_report = run_inference(snapshot, context)

            edges = model.signals(snapshot, context, llm_report, history)
            if not edges:
                logger.info("No edges from signal model", extra={"ticker": ticker})
                continue
            await executor.process_edges(edges, snapshot, context)
        except Exception as exc:
            logger.warning("Inference cycle failed for ticker",
                           extra={"ticker": ticker, "error": str(exc)})

    snap = await executor.portfolio.snapshot()
    logger.info("Tick complete", extra={
        "cash": snap.cash, "positions": len(snap.positions),
        "realized_pnl": snap.realized_pnl,
        "signal_model": args.signal_model,
    })


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    backend = InMemoryBackend()
    portfolio = PortfolioState(backend, env=f"paper:{args.signal_model}")
    await portfolio.initialize(args.starting_capital)

    client = KalshiTradingClientStub()
    executor = PaperExecutor(portfolio, client, RiskLimits())
    model = await _build_signal_model(args.signal_model)

    if args.once:
        await _run_once(args, executor, model)
        return

    interval_s = _interval_seconds(args.interval)
    logger.info("Paper-trade loop starting", extra={
        "interval_s": interval_s, "signal_model": args.signal_model,
    })
    while True:
        await _run_once(args, executor, model)
        await asyncio.sleep(interval_s)


if __name__ == "__main__":
    asyncio.run(main())
