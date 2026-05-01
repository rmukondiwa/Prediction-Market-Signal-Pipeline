from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import HistoricalContext
from src.signals.settlement_decay import SettlementDecaySignal


def _focus(implied: float, ticker: str = "X-1") -> MarketSnapshot:
    return MarketSnapshot(
        event=ticker, market=ticker, outcome="YES",
        quoted_price=int(implied * 100), implied_probability=implied,
        yes_bid=int(implied * 100) - 1, yes_ask=int(implied * 100) + 1,
        volume=0, open_interest=0, source="test",
        timestamp=datetime.now(timezone.utc),
    )


def _empty_report(focus: MarketSnapshot) -> InferenceReport:
    return InferenceReport(
        focus_market=focus, context_markets=[],
        consistency_analysis="", derived_probabilities=[],
        detected_mispricings=[], suggested_edges=[],
    )


def _ctx_with_settlement(focus_ticker: str, hours_to_close: float) -> HistoricalContext:
    """Build a HistoricalContext where the focus ticker's settlement is N hours away."""
    now = datetime(2026, 4, 30, tzinfo=timezone.utc)
    close_t = now + timedelta(hours=hours_to_close)
    h = HistoricalContext(as_of=now)
    # Stuff settlement_times via model_extra (Pydantic v2 escape hatch)
    h.__dict__["settlement_times"] = {focus_ticker: close_t}
    return h


def test_no_signal_when_implied_below_threshold():
    sig = SettlementDecaySignal(min_implied_prob=0.95, no_side_max_implied=0.05,
                                max_hours_to_close=72)
    focus = _focus(0.80)  # below threshold
    edges = sig.signals(focus, [], _empty_report(focus),
                        _ctx_with_settlement(focus.market, hours_to_close=24))
    assert edges == []


def test_yes_side_signal_fires_when_implied_high_and_close_to_settlement():
    sig = SettlementDecaySignal(min_implied_prob=0.95, max_hours_to_close=72,
                                fair_prob_when_yes=0.99)
    focus = _focus(0.96)
    edges = sig.signals(focus, [], _empty_report(focus),
                        _ctx_with_settlement(focus.market, hours_to_close=12))
    assert len(edges) == 1
    e = edges[0]
    assert e.side == "yes"
    assert e.estimated_fair_prob == 0.99
    assert e.current_implied_prob == 0.96
    assert e.kelly_fraction > 0
    assert e.source_signal_model == "settlement_decay"


def test_no_signal_after_market_already_closed():
    """Negative hours-to-close → no signal, even with high implied."""
    sig = SettlementDecaySignal(min_implied_prob=0.95, max_hours_to_close=72)
    focus = _focus(0.97)
    edges = sig.signals(focus, [], _empty_report(focus),
                        _ctx_with_settlement(focus.market, hours_to_close=-1))
    assert edges == []


def test_no_signal_when_too_far_from_settlement():
    sig = SettlementDecaySignal(min_implied_prob=0.95, max_hours_to_close=24)
    focus = _focus(0.97)
    # 200 hours out — too far for max_hours_to_close=24
    edges = sig.signals(focus, [], _empty_report(focus),
                        _ctx_with_settlement(focus.market, hours_to_close=200))
    assert edges == []


def test_no_side_signal_fires_for_low_implied():
    sig = SettlementDecaySignal(min_implied_prob=0.95, no_side_max_implied=0.05,
                                max_hours_to_close=72, fair_prob_when_no=0.01)
    focus = _focus(0.04)  # very unlikely YES → buy NO
    edges = sig.signals(focus, [], _empty_report(focus),
                        _ctx_with_settlement(focus.market, hours_to_close=24))
    assert len(edges) == 1
    e = edges[0]
    assert e.side == "no"
    assert e.kelly_fraction > 0


def test_no_signal_without_settlement_time():
    """No settlement_times info → no signal (we won't trade blind)."""
    sig = SettlementDecaySignal(min_implied_prob=0.95)
    focus = _focus(0.97)
    history = HistoricalContext(as_of=datetime.now(timezone.utc))
    # No settlement_times set
    edges = sig.signals(focus, [], _empty_report(focus), history)
    assert edges == []
