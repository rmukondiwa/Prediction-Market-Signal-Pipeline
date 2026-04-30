"""
Backtest runner.

Orchestrates the full replay-and-evaluate loop:
    for ticker in resolved universe:
        for t in walk_timesteps:
            reconstruct snapshot at t
            retrieve+rerank+infer (cached)
            for each signal model:
                emit edges → risk gate → size → simulate fill → record
        settle at ticker's resolution time

The runner is intentionally injectable — every external dependency
(retriever, reranker, inference fn, fill simulator, signal models) can be
swapped for tests via the constructor.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable

from src.backtest.cache import InferenceCache
from src.backtest.fill_simulator import FillSimulator
from src.backtest.metrics import (
    brier_score,
    daily_returns_from_curve,
    equity_curve,
    hit_rate_by_confidence,
    market_baseline_brier,
    max_drawdown,
    sharpe,
)
from src.backtest.models import (
    BacktestConfig,
    BacktestReport,
    Candle,
    SignalModelMetrics,
    SimulatedFill,
)
from src.backtest.replayer import build_snapshot_at, walk_timesteps
from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import SignalModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Function types for injectable dependencies
RetrieveFn = Callable[[MarketSnapshot, "HistoricalContext"], list[ContextMarket]]
InferFn = Callable[[MarketSnapshot, list[ContextMarket]], InferenceReport]


class BacktestRunner:
    """
    Runs a backtest. Constructed once per backtest invocation.
    Dependencies are injected at construction so tests can stub the LLM stack.
    """

    def __init__(
        self,
        config: BacktestConfig,
        signal_models: dict[str, SignalModel],
        candles_by_ticker: dict[str, list[Candle]],
        outcomes: dict[str, float],
        settlement_times: dict[str, datetime],
        retrieve_fn: RetrieveFn,
        infer_fn: InferFn,
        history: HistoricalContext | None = None,
        fill_simulator: FillSimulator | None = None,
        cache: InferenceCache | None = None,
        title_lookup: dict[str, str] | None = None,
    ):
        self.config = config
        self.signal_models = signal_models
        self.candles = candles_by_ticker
        self.outcomes = outcomes
        self.settlement_times = settlement_times
        self.retrieve = retrieve_fn
        self.infer = infer_fn
        self.history = history or HistoricalContext()
        self.simulator = fill_simulator or FillSimulator(
            model=config.fill_model,
            slippage_per_contract=config.slippage_per_contract,
            fee_per_contract=config.fee_per_contract,
        )
        self.cache = cache
        self.titles = title_lookup or {}

    def run(self) -> BacktestReport:
        per_signal_fills: dict[str, list[SimulatedFill]] = defaultdict(list)
        per_signal_decisions: dict[str, int] = defaultdict(int)
        # Track market-baseline predictions for Brier comparison
        baseline_implied: list[float] = []
        baseline_outcomes: list[float] = []

        universe = self.config.universe or sorted(self.outcomes.keys())
        logger.info("Starting backtest", extra={
            "tickers": len(universe),
            "models": list(self.signal_models.keys()),
        })

        for ticker in universe:
            outcome = self.outcomes.get(ticker)
            if outcome is None:
                continue
            ticker_candles = self.candles.get(ticker, [])
            if not ticker_candles:
                logger.warning("No candles for ticker, skipping", extra={"ticker": ticker})
                continue

            title = self.titles.get(ticker, ticker)

            for t in walk_timesteps(self.config.start_date, self.config.end_date, self.config.granularity):
                # Stop walking once we're past this market's settlement
                settle_t = self.settlement_times.get(ticker)
                if settle_t and t >= settle_t:
                    break

                snapshot = build_snapshot_at(ticker, title, t, ticker_candles)
                if snapshot is None:
                    continue

                # Track baseline at every timestep this market produces a snapshot
                baseline_implied.append(snapshot.implied_probability)
                baseline_outcomes.append(outcome)

                self.history.as_of = t
                context = self.retrieve(snapshot, self.history)
                llm_report = self.infer(snapshot, context)

                for model_name, model in self.signal_models.items():
                    edges = model.signals(snapshot, context, llm_report, self.history)
                    per_signal_decisions[model_name] += len(edges)
                    for ce in edges:
                        contracts = self._size(ce, snapshot)
                        if contracts <= 0:
                            continue
                        fill = self.simulator.simulate(ce, snapshot, contracts, t)
                        if fill is not None:
                            per_signal_fills[model_name].append(fill)

        baseline = market_baseline_brier(baseline_implied, baseline_outcomes) if baseline_implied else 0.0
        per_signal_metrics = {
            name: self._metrics_for(name, fills, per_signal_decisions[name])
            for name, fills in per_signal_fills.items()
        }
        # Ensure every requested model appears in the report even if it had no fills
        for name in self.signal_models:
            if name not in per_signal_metrics:
                per_signal_metrics[name] = self._metrics_for(name, [], per_signal_decisions[name])

        return BacktestReport(
            config=self.config,
            market_baseline_brier=baseline,
            per_signal=per_signal_metrics,
            cache_stats=self.cache.stats() if self.cache else {},
        )

    def _size(self, edge: CalibratedEdge, snapshot: MarketSnapshot) -> int:
        """Quarter-Kelly contracts based on remaining capital and current price."""
        kelly = max(0.0, min(self.config.risk_limits.max_kelly_fraction, edge.kelly_fraction))
        if kelly <= 0:
            return 0
        # Use price on the side we're buying
        price = (snapshot.yes_bid + snapshot.yes_ask) / 2 / 100.0
        if edge.side == "no":
            price = 1.0 - price
        if price <= 0:
            return 0
        cash_to_deploy = self.config.starting_capital * kelly
        contracts = int(cash_to_deploy // max(price, 0.01))
        # Per-market USD cap
        max_contracts = int(self.config.risk_limits.max_per_market_usd // max(price, 0.01))
        return max(0, min(contracts, max_contracts))

    def _metrics_for(
        self,
        name: str,
        fills: list[SimulatedFill],
        n_decisions: int,
    ) -> SignalModelMetrics:
        # Brier on this signal model's calibrated probabilities
        preds: list[float] = []
        outs: list[float] = []
        for f in fills:
            if f.ticker not in self.outcomes:
                continue
            preds.append(f.estimated_fair_prob)
            outs.append(self.outcomes[f.ticker])

        bs = brier_score(preds, outs) if preds else 0.0
        curve = equity_curve(fills, self.outcomes, self.config.starting_capital)
        dd = max_drawdown(curve)
        sh = sharpe(daily_returns_from_curve(curve))
        by_conf = hit_rate_by_confidence(fills, self.outcomes)

        # Realized P&L = final equity - starting capital, restricted to fills with outcomes
        realized = curve[-1] - self.config.starting_capital if curve else 0.0

        return SignalModelMetrics(
            name=name,
            n_decisions=n_decisions,
            n_fills=len(fills),
            pnl_realized=round(realized, 4),
            pnl_unrealized=0.0,  # backtest universe is resolved markets only
            by_confidence=by_conf,
            brier_score=round(bs, 6),
            sharpe=round(sh, 4) if sh is not None else None,
            max_drawdown=round(dd, 4),
            edge_decay_curve=[],
            fills=fills,
        )
