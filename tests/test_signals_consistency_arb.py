from __future__ import annotations

from datetime import datetime, timezone

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.consistency_arb import (
    ConsistencyArbSignal,
    _extract_deadline,
    _extract_event_root,
)
from src.signals.models import HistoricalContext


def test_extract_deadline_kalshi_format():
    d = _extract_deadline("KXIRANUS-26JUN30-YES")
    assert d is not None
    assert (d.year, d.month, d.day) == (2026, 6, 30)


def test_extract_deadline_4digit_year():
    d = _extract_deadline("KXSPYY-01JAN2027")
    assert d is not None
    assert d.year == 2027


def test_extract_deadline_unparseable():
    assert _extract_deadline("FOO-BAR") is None


def test_extract_event_root_strips_date_suffix():
    assert _extract_event_root("KXIRAN-26JUN30-YES") == "KXIRAN"
    assert _extract_event_root("KXIRAN-26DEC31") == "KXIRAN"


def _empty_report(focus: MarketSnapshot) -> InferenceReport:
    return InferenceReport(
        focus_market=focus, context_markets=[],
        consistency_analysis="", derived_probabilities=[],
        detected_mispricings=[], suggested_edges=[],
    )


def _ctx(ticker: str, title: str, prob: float) -> ContextMarket:
    return ContextMarket(
        ticker=ticker, event_ticker=ticker, title=title, subtitle="",
        category="", status="open",
        yes_bid=int(prob * 100) - 1, yes_ask=int(prob * 100) + 1,
        implied_probability=prob,
        similarity_score=0.7, relevance_score=8.0,
        relationship="same event tree, different deadline",
    )


def test_monotonicity_violation_emits_two_edges():
    """P(by June)=40% > P(by December)=30% violates monotonicity → emit 2 edges."""
    focus = MarketSnapshot(
        event="Iran by June", market="KXIRAN-26JUN30-YES", outcome="YES",
        quoted_price=40, implied_probability=0.40,
        yes_bid=39, yes_ask=41, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )
    context = [_ctx("KXIRAN-26DEC31-YES", "Iran by December", 0.30)]
    report = _empty_report(focus)
    model = ConsistencyArbSignal()
    edges = model.signals(focus, context, report, HistoricalContext())

    # Should emit 2 edges: short the over (June) and long the under (December)
    sides = sorted((e.ticker, e.side) for e in edges)
    assert ("KXIRAN-26DEC31-YES", "yes") in sides  # underpriced → buy yes
    assert ("KXIRAN-26JUN30-YES", "no") in sides   # overpriced → buy no
    assert len(edges) == 2

    for e in edges:
        assert e.confidence >= 0.9
        assert e.source_signal_model == "consistency_arb"


def test_no_violation_no_edges():
    """When monotonicity holds, no edges."""
    focus = MarketSnapshot(
        event="Iran by June", market="KXIRAN-26JUN30-YES", outcome="YES",
        quoted_price=20, implied_probability=0.20,
        yes_bid=19, yes_ask=21, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )
    # Earlier deadline at 20%, later at 30% → no violation
    context = [_ctx("KXIRAN-26DEC31-YES", "Iran by December", 0.30)]
    report = _empty_report(focus)
    edges = ConsistencyArbSignal().signals(focus, context, report, HistoricalContext())
    assert edges == []


def test_violation_below_threshold_ignored():
    """A 1pp gap should be ignored — within bid/ask noise."""
    focus = MarketSnapshot(
        event="Iran by June", market="KXIRAN-26JUN30-YES", outcome="YES",
        quoted_price=31, implied_probability=0.31,
        yes_bid=30, yes_ask=32, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )
    context = [_ctx("KXIRAN-26DEC31-YES", "Iran by December", 0.30)]
    report = _empty_report(focus)
    edges = ConsistencyArbSignal(min_violation_pp=0.05).signals(focus, context, report, HistoricalContext())
    assert edges == []


def test_unrelated_event_roots_dont_get_compared():
    focus = MarketSnapshot(
        event="A", market="ALPHA-26JUN30-YES", outcome="YES",
        quoted_price=80, implied_probability=0.80,
        yes_bid=79, yes_ask=81, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )
    # Different event root, even though one has a higher implied prob
    context = [_ctx("BETA-26DEC31-YES", "B", 0.10)]
    report = _empty_report(focus)
    edges = ConsistencyArbSignal().signals(focus, context, report, HistoricalContext())
    assert edges == []
