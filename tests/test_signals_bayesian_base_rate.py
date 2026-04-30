from __future__ import annotations

from datetime import datetime, timezone

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.bayesian_base_rate import BayesianBaseRateSignal, _bayesian_update
from src.signals.models import HistoricalContext


def test_bayesian_update_pulls_toward_market_when_kappa_high():
    posterior = _bayesian_update(prior=0.20, market_implied=0.50, kappa=10.0)
    # Heavy market weight → posterior near market
    assert 0.30 < posterior < 0.55


def test_bayesian_update_balanced_with_low_kappa():
    posterior = _bayesian_update(prior=0.20, market_implied=0.50, kappa=1.0)
    # Lower kappa → more weight on prior (relative to high-kappa case)
    assert posterior < 0.50


def _focus_snapshot(implied: float = 0.50) -> MarketSnapshot:
    return MarketSnapshot(
        event="x", market="X-1", outcome="YES",
        quoted_price=int(implied * 100), implied_probability=implied,
        yes_bid=int(implied * 100) - 1, yes_ask=int(implied * 100) + 1,
        volume=0, open_interest=0, source="test",
        timestamp=datetime.now(timezone.utc),
    )


def _ctx(ticker: str, prob: float = 0.30) -> ContextMarket:
    return ContextMarket(
        ticker=ticker, event_ticker=ticker, title=ticker, subtitle="",
        category="", status="open",
        yes_bid=int(prob * 100) - 1, yes_ask=int(prob * 100) + 1,
        implied_probability=prob,
        similarity_score=0.7, relevance_score=8.0,
        relationship="analogue",
    )


def _empty_report(focus: MarketSnapshot) -> InferenceReport:
    return InferenceReport(
        focus_market=focus, context_markets=[],
        consistency_analysis="", derived_probabilities=[],
        detected_mispricings=[], suggested_edges=[],
    )


def test_no_signal_below_min_analogues():
    """With <5 resolved analogues, return nothing."""
    focus = _focus_snapshot(0.50)
    context = [_ctx(f"A-{i}") for i in range(3)]
    history = HistoricalContext(resolutions={
        f"A-{i}": {"settlement_value": 1.0} for i in range(3)
    })
    edges = BayesianBaseRateSignal().signals(focus, context, _empty_report(focus), history)
    assert edges == []


def test_signal_emitted_when_prior_diverges_from_market():
    """5 analogous markets all resolved YES → prior 100% → strong yes-side edge if market is at 50%."""
    focus = _focus_snapshot(0.50)
    context = [_ctx(f"A-{i}") for i in range(5)]
    history = HistoricalContext(resolutions={
        f"A-{i}": {"settlement_value": 1.0} for i in range(5)
    })
    edges = BayesianBaseRateSignal(min_analogues=5).signals(focus, context, _empty_report(focus), history)
    assert len(edges) == 1
    e = edges[0]
    assert e.side == "yes"
    assert e.estimated_fair_prob > focus.implied_probability
    assert e.source_signal_model == "bayesian_base_rate"
    assert "Empirical base rate" in e.thesis


def test_signal_emitted_when_prior_below_market():
    """5 analogues all resolved NO → prior near 0% → no-side edge if market is at 50%."""
    focus = _focus_snapshot(0.50)
    context = [_ctx(f"A-{i}") for i in range(5)]
    history = HistoricalContext(resolutions={
        f"A-{i}": {"settlement_value": 0.0} for i in range(5)
    })
    edges = BayesianBaseRateSignal(min_analogues=5).signals(focus, context, _empty_report(focus), history)
    assert len(edges) == 1
    assert edges[0].side == "no"


def test_no_signal_when_edge_too_small():
    """If prior matches market implied closely, no actionable edge."""
    focus = _focus_snapshot(0.50)
    context = [_ctx(f"A-{i}") for i in range(10)]
    # 50% base rate
    history = HistoricalContext(resolutions={
        f"A-{i}": {"settlement_value": 1.0 if i < 5 else 0.0} for i in range(10)
    })
    edges = BayesianBaseRateSignal(min_analogues=5).signals(focus, context, _empty_report(focus), history)
    assert edges == []
