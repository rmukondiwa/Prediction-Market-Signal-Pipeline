"""End-to-end test of BacktestRunner with stubbed retrieve/infer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.backtest.cache import InferenceCache
from src.backtest.models import BacktestConfig, RiskLimitsConfig
from src.backtest.replayer import synthetic_candles
from src.backtest.runner import BacktestRunner
from src.context.models import ContextMarket
from src.inference.models import (
    DerivedProbability, Edge, InferenceReport, Mispricing,
)
from src.signals.models import HistoricalContext
from src.signals.raw_llm import RawLLMSignal


def _make_config(start, end):
    return BacktestConfig(
        start_date=start,
        end_date=end,
        granularity="1h",
        starting_capital=10_000.0,
        fill_model="midpoint",
        slippage_per_contract=0.01,
        fee_per_contract=0.0,
        risk_limits=RiskLimitsConfig(),
        signal_models=["raw_llm"],
    )


def _stub_infer_factory(target_ticker: str, fair_prob: float):
    """Build a fake infer fn that always claims `target_ticker` is mispriced
    in the direction of `fair_prob`."""
    def fake_infer(snapshot, context):
        implied = snapshot.implied_probability
        side = "yes" if fair_prob > implied else "no"
        return InferenceReport(
            focus_market=snapshot,
            context_markets=[],
            consistency_analysis="",
            derived_probabilities=[],
            detected_mispricings=[Mispricing(
                ticker=snapshot.market, title=snapshot.event,
                direction="underpriced" if side == "yes" else "overpriced",
                current_implied_prob=implied,
                estimated_fair_prob=fair_prob,
                reasoning="test",
            )],
            suggested_edges=[Edge(
                ticker=snapshot.market, title=snapshot.event,
                side=side, confidence="high",
                thesis="test", kelly_fraction=0.10,
            )],
        )
    return fake_infer


def test_runner_produces_report_with_baseline_brier(tmp_cache_dir):
    """Smoke test: runner produces a complete BacktestReport."""
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 2, tzinfo=timezone.utc)
    config = _make_config(start, end)

    # Universe: A resolved YES, B resolved NO, both with steady prices
    candles = {
        "A": synthetic_candles("A", start, end, 30, 30, "1h"),
        "B": synthetic_candles("B", start, end, 70, 70, "1h"),
    }
    outcomes = {"A": 1.0, "B": 0.0}
    settle = {"A": end + timedelta(hours=1), "B": end + timedelta(hours=1)}

    runner = BacktestRunner(
        config=config,
        signal_models={"raw_llm": RawLLMSignal()},
        candles_by_ticker=candles,
        outcomes=outcomes,
        settlement_times=settle,
        retrieve_fn=lambda s, h: [],
        infer_fn=_stub_infer_factory("A", 0.50),  # claim everything ~50%
        cache=InferenceCache(tmp_cache_dir),
    )
    report = runner.run()

    assert report.config == config
    assert report.market_baseline_brier > 0
    assert "raw_llm" in report.per_signal
    assert report.per_signal["raw_llm"].n_decisions > 0


def test_runner_skips_tickers_without_candles(tmp_cache_dir):
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, 4, tzinfo=timezone.utc)
    config = _make_config(start, end)

    candles = {"A": synthetic_candles("A", start, end, 50, 50, "1h")}
    outcomes = {"A": 1.0, "B": 0.0}  # B has no candles

    runner = BacktestRunner(
        config=config,
        signal_models={"raw_llm": RawLLMSignal()},
        candles_by_ticker=candles,
        outcomes=outcomes,
        settlement_times={"A": end + timedelta(hours=1), "B": end + timedelta(hours=1)},
        retrieve_fn=lambda s, h: [],
        infer_fn=_stub_infer_factory("A", 0.6),
    )
    report = runner.run()
    # Only A produced fills
    fills = report.per_signal["raw_llm"].fills
    assert all(f.ticker == "A" for f in fills)


def test_runner_market_baseline_brier_matches_settlement():
    """If the implied prob is always 0.30 and the market resolves YES,
    market_baseline_brier should be (0.30 - 1.0)**2 = 0.49."""
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, 4, tzinfo=timezone.utc)
    config = _make_config(start, end)

    # Candle close at 30 cents → implied 0.30 throughout
    candles = {"A": synthetic_candles("A", start, end, 30, 30, "1h")}
    outcomes = {"A": 1.0}  # YES won

    runner = BacktestRunner(
        config=config,
        signal_models={"raw_llm": RawLLMSignal()},
        candles_by_ticker=candles,
        outcomes=outcomes,
        settlement_times={"A": end + timedelta(hours=1)},
        retrieve_fn=lambda s, h: [],
        infer_fn=_stub_infer_factory("A", 0.5),  # not relevant for baseline
    )
    report = runner.run()
    # build_snapshot_at uses half-spread → implied = (29+31)/2 / 100 = 0.30
    expected = (0.30 - 1.0) ** 2
    assert abs(report.market_baseline_brier - expected) < 1e-3


def test_runner_kill_switch_zero_kelly_produces_no_fills(tmp_cache_dir):
    """If the LLM emits zero kelly_fraction edges, no fills should be produced."""
    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    end = datetime(2026, 4, 1, 4, tzinfo=timezone.utc)
    config = _make_config(start, end)

    def zero_kelly_infer(snapshot, context):
        return InferenceReport(
            focus_market=snapshot, context_markets=[],
            consistency_analysis="", derived_probabilities=[],
            detected_mispricings=[Mispricing(
                ticker=snapshot.market, title="x", direction="underpriced",
                current_implied_prob=snapshot.implied_probability,
                estimated_fair_prob=snapshot.implied_probability + 0.001,
                reasoning="",
            )],
            suggested_edges=[Edge(
                ticker=snapshot.market, title="x", side="yes",
                confidence="low", thesis="", kelly_fraction=0.0,
            )],
        )

    candles = {"A": synthetic_candles("A", start, end, 50, 50, "1h")}
    runner = BacktestRunner(
        config=config,
        signal_models={"raw_llm": RawLLMSignal()},
        candles_by_ticker=candles,
        outcomes={"A": 1.0},
        settlement_times={"A": end + timedelta(hours=1)},
        retrieve_fn=lambda s, h: [],
        infer_fn=zero_kelly_infer,
    )
    report = runner.run()
    assert report.per_signal["raw_llm"].n_fills == 0
