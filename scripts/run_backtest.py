"""
Run a backtest.

Usage:
    python -m scripts.run_backtest \\
        --start 2026-01-01 --end 2026-03-31 \\
        --signal-models raw_llm,consistency_arb \\
        --output reports/bt_$(date +%Y%m%d).json

Reads candle/resolution data from data/archive (Stage 6) when available,
falls back to synthetic data when explicitly requested via --synthetic.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.backtest.cache import InferenceCache
from src.backtest.models import BacktestConfig, RiskLimitsConfig
from src.backtest.runner import BacktestRunner
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Phase 5 backtest")
    p.add_argument("--start", type=str, required=True, help="ISO date e.g. 2026-01-01")
    p.add_argument("--end", type=str, required=True, help="ISO date e.g. 2026-03-31")
    p.add_argument("--granularity", type=str, default="1h")
    p.add_argument("--signal-models", type=str, default="raw_llm",
                   help="Comma-separated list of signal model names to run")
    p.add_argument("--starting-capital", type=float, default=10_000.0)
    p.add_argument("--cache-dir", type=Path, default=Path("data/backtest_cache"))
    p.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data; useful for smoke-testing the runner without real LLM/network")
    return p.parse_args()


def _build_signal_models(names: list[str]):
    from src.signals.raw_llm import RawLLMSignal
    registry = {"raw_llm": RawLLMSignal()}

    try:
        from src.signals.consistency_arb import ConsistencyArbSignal
        registry["consistency_arb"] = ConsistencyArbSignal()
    except Exception:
        pass

    try:
        from src.signals.bayesian_base_rate import BayesianBaseRateSignal
        registry["bayesian_base_rate"] = BayesianBaseRateSignal()
    except Exception:
        pass

    try:
        from src.signals.coherence_regression import CoherenceRegressionSignal
        registry["coherence_regression"] = CoherenceRegressionSignal()
    except Exception:
        pass

    try:
        from src.signals.calibrated_llm import CalibratedLLMSignal
        # only include if calibration map exists
        cm_path = Path("data/calibration_map.pkl")
        if cm_path.exists():
            registry["calibrated_llm"] = CalibratedLLMSignal(cm_path)
    except Exception:
        pass

    out = {}
    for n in names:
        n = n.strip()
        if n in registry:
            out[n] = registry[n]
        else:
            logger.warning("Unknown signal model, skipping", extra={"name": n})
    return out


def _print_summary(report) -> None:
    print(f"\n=== Backtest summary ===")
    print(f"  market_baseline_brier: {report.market_baseline_brier:.4f}")
    print(f"  cache: {report.cache_stats}\n")
    fmt = "  {:22} | {:>7} | {:>7} | {:>7} | {:>6} | {:>10}"
    print(fmt.format("model", "Brier", "Sharpe", "MaxDD", "Fills", "PnL"))
    print("  " + "-" * 70)
    for name, m in report.per_signal.items():
        sh = f"{m.sharpe:+.2f}" if m.sharpe is not None else "  n/a"
        print(fmt.format(
            name,
            f"{m.brier_score:.4f}",
            sh,
            f"{m.max_drawdown:.2%}",
            m.n_fills,
            f"${m.pnl_realized:+.2f}",
        ))


async def main() -> None:
    load_dotenv()
    args = _parse_args()

    config = BacktestConfig(
        start_date=datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc),
        end_date=datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc),
        granularity=args.granularity,
        starting_capital=args.starting_capital,
        risk_limits=RiskLimitsConfig(),
        signal_models=[s.strip() for s in args.signal_models.split(",") if s.strip()],
    )

    signal_models = _build_signal_models(config.signal_models)
    if not signal_models:
        print("No signal models loaded — aborting.")
        return

    cache = InferenceCache(args.cache_dir)

    if args.synthetic:
        # Smoke-test path: build a minimal universe inline
        report = _run_synthetic(config, signal_models, cache)
    else:
        # Real path requires Stage 6 archive accumulation + Kalshi candle backfill.
        # Print guidance rather than silently producing empty results.
        print(
            "\nReal-data backtest path requires:\n"
            "  - data/archive/resolutions.parquet (Stage 6, runs daily)\n"
            "  - data/archive/<dates>/catalog.parquet (Stage 6)\n"
            "  - candle backfill via scripts/fetch_resolutions.py (TODO)\n"
            "\nFor now, run with --synthetic to verify the runner works end-to-end.\n"
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
    _print_summary(report)
    print(f"\n  Report written to: {args.output}")


def _run_synthetic(config, signal_models, cache):
    """Build a tiny synthetic universe and run the backtest end-to-end."""
    from datetime import timedelta

    from src.backtest.replayer import synthetic_candles
    from src.context.models import ContextMarket
    from src.inference.models import (
        DerivedProbability, Edge, InferenceReport, Mispricing,
    )
    from src.signals.models import HistoricalContext

    candles_by_ticker = {}
    outcomes = {}
    settlement_times = {}
    titles = {}

    # Two synthetic markets that resolved YES, two that resolved NO
    base = config.start_date
    for i, (t, outcome, drift) in enumerate([
        ("SYN-A-YES", 1.0, +20),
        ("SYN-B-NO", 0.0, -15),
        ("SYN-C-YES", 1.0, +30),
        ("SYN-D-NO", 0.0, -25),
    ]):
        candles_by_ticker[t] = synthetic_candles(
            t, base, config.end_date,
            open_price=50.0, close_price=50.0 + drift,
            granularity=config.granularity,
        )
        outcomes[t] = outcome
        settlement_times[t] = config.end_date + timedelta(hours=1)
        titles[t] = f"Synthetic market {i}"

    def fake_retrieve(snapshot, history):
        return []

    def fake_infer(snapshot, context):
        # Synthetic LLM: claims a 5% edge in the direction of recent drift
        # (which is exactly what RawLLM will translate into a buy signal)
        implied = snapshot.implied_probability
        # Drift-aware fair guess: if implied is moving up, claim higher fair
        fair = min(0.99, max(0.01, implied + 0.05))
        side = "yes" if fair > implied else "no"
        return InferenceReport(
            focus_market=snapshot,
            context_markets=[],
            consistency_analysis="",
            derived_probabilities=[],
            detected_mispricings=[Mispricing(
                ticker=snapshot.market, title=snapshot.event,
                direction="underpriced" if side == "yes" else "overpriced",
                current_implied_prob=implied, estimated_fair_prob=fair,
                reasoning="synthetic",
            )],
            suggested_edges=[Edge(
                ticker=snapshot.market, title=snapshot.event,
                side=side, confidence="medium",
                thesis="synthetic", kelly_fraction=0.05,
            )],
        )

    runner = BacktestRunner(
        config=config,
        signal_models=signal_models,
        candles_by_ticker=candles_by_ticker,
        outcomes=outcomes,
        settlement_times=settlement_times,
        retrieve_fn=fake_retrieve,
        infer_fn=fake_infer,
        history=HistoricalContext(),
        cache=cache,
        title_lookup=titles,
    )
    return runner.run()


if __name__ == "__main__":
    asyncio.run(main())
