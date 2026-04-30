from __future__ import annotations

from datetime import datetime, timezone

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.coherence_regression import CoherenceRegressionSignal
from src.signals.models import HistoricalContext


def _focus(implied: float) -> MarketSnapshot:
    return MarketSnapshot(
        event="focus", market="F-1", outcome="YES",
        quoted_price=int(implied * 100), implied_probability=implied,
        yes_bid=int(implied * 100) - 1, yes_ask=int(implied * 100) + 1,
        volume=0, open_interest=0, source="test",
        timestamp=datetime.now(timezone.utc),
    )


def _ctx(ticker: str, prob: float, relevance: float = 8.0) -> ContextMarket:
    return ContextMarket(
        ticker=ticker, event_ticker=ticker, title=ticker, subtitle="",
        category="", status="open",
        yes_bid=int(prob * 100) - 1, yes_ask=int(prob * 100) + 1,
        implied_probability=prob,
        similarity_score=0.7, relevance_score=relevance,
        relationship="related",
    )


def _empty_report(focus):
    return InferenceReport(
        focus_market=focus, context_markets=[],
        consistency_analysis="", derived_probabilities=[],
        detected_mispricings=[], suggested_edges=[],
    )


def test_no_signal_with_too_little_context():
    focus = _focus(0.30)
    context = [_ctx("A", 0.50)]  # only 1 context market
    edges = CoherenceRegressionSignal().signals(focus, context, _empty_report(focus), HistoricalContext())
    assert edges == []


def test_signal_emitted_when_focus_lags_related_markets():
    """Focus at 30%, related markets all at 60% → focus is underpriced → buy yes."""
    focus = _focus(0.30)
    context = [_ctx(f"A-{i}", 0.60) for i in range(4)]
    edges = CoherenceRegressionSignal().signals(focus, context, _empty_report(focus), HistoricalContext())
    assert len(edges) == 1
    e = edges[0]
    assert e.side == "yes"
    assert e.estimated_fair_prob > 0.30
    assert e.source_signal_model == "coherence_regression"


def test_signal_emitted_when_focus_leads_related_markets():
    """Focus at 70%, related markets all at 30% → focus is overpriced → buy no."""
    focus = _focus(0.70)
    context = [_ctx(f"A-{i}", 0.30) for i in range(4)]
    edges = CoherenceRegressionSignal().signals(focus, context, _empty_report(focus), HistoricalContext())
    assert len(edges) == 1
    assert edges[0].side == "no"


def test_no_signal_when_residual_below_threshold():
    """Tiny mismatch → no edge."""
    focus = _focus(0.50)
    context = [_ctx(f"A-{i}", 0.51) for i in range(4)]
    edges = CoherenceRegressionSignal(min_edge_pp=0.05).signals(focus, context, _empty_report(focus), HistoricalContext())
    assert edges == []


def test_relevance_weighting_dominates():
    """A high-relevance context market with a divergent price should dominate the weighted mean."""
    focus = _focus(0.50)
    context = [
        _ctx("A", 0.80, relevance=10.0),  # very relevant, says 0.80
        _ctx("B", 0.20, relevance=1.0),   # less relevant, says 0.20
        _ctx("C", 0.80, relevance=9.0),   # also strongly says 0.80
        _ctx("D", 0.80, relevance=9.0),
    ]
    edges = CoherenceRegressionSignal().signals(focus, context, _empty_report(focus), HistoricalContext())
    assert len(edges) == 1
    assert edges[0].side == "yes"  # weighted mean ~0.78, focus 0.50 → buy yes
