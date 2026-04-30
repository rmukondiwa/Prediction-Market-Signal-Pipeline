from __future__ import annotations

import pytest

from src.signals.models import HistoricalContext
from src.signals.protocol import SignalModel, quarter_kelly
from src.signals.raw_llm import RawLLMSignal


def test_raw_llm_satisfies_protocol():
    assert isinstance(RawLLMSignal(), SignalModel)


def test_quarter_kelly_no_edge_returns_zero():
    # implied higher than fair → no edge on the yes side
    assert quarter_kelly(0.30, 0.50, "yes") == 0.0


def test_quarter_kelly_yes_edge_is_positive_clamped():
    # 50% fair vs 30% implied → 20pp edge on yes
    k = quarter_kelly(0.50, 0.30, "yes")
    assert 0.0 < k <= 0.25


def test_quarter_kelly_no_side_uses_complement():
    # 30% fair → 70% on no; if implied=50% → 20pp edge on no side
    k = quarter_kelly(0.30, 0.50, "no")
    assert 0.0 < k <= 0.25


def test_quarter_kelly_clamps_to_25_pct():
    """Even with massive edge, quarter-Kelly never exceeds 0.25."""
    # Sweep extreme edges; max raw kelly ~= 1.0 → max quarter ~= 0.25
    for p, q in [(0.99, 0.01), (0.95, 0.05), (0.90, 0.10)]:
        k = quarter_kelly(p, q, "yes")
        assert 0.0 < k <= 0.25


def test_quarter_kelly_invalid_inputs():
    assert quarter_kelly(0.5, 0.0, "yes") == 0.0
    assert quarter_kelly(0.5, 1.0, "yes") == 0.0


def test_raw_llm_passes_through_edges(sample_inference_report, sample_market_snapshot):
    """RawLLMSignal should produce one CalibratedEdge per LLM Edge."""
    model = RawLLMSignal()
    edges = model.signals(
        focus=sample_market_snapshot,
        context=sample_inference_report.context_markets,
        llm_report=sample_inference_report,
        history=HistoricalContext(),
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.ticker == "KXIRANUS-26JUN30-YES"
    assert edge.side == "yes"
    assert edge.source_signal_model == "raw_llm"
    # estimated_fair_prob should equal the LLM mispricing's value
    assert edge.estimated_fair_prob == 0.30
    assert edge.current_implied_prob == 0.23
    # edge_pp is fair - implied
    assert abs(edge.edge_pp - 0.07) < 1e-9
    assert edge.kelly_fraction == 0.05  # passed through


def test_raw_llm_skips_edges_without_mispricing_data():
    """If the LLM emits an Edge without a corresponding Mispricing entry, skip."""
    from datetime import datetime, timezone

    from src.context.models import ContextMarket
    from src.inference.models import Edge, InferenceReport
    from src.insight.models import MarketSnapshot

    snapshot = MarketSnapshot(
        event="x", market="X-1", outcome="YES", quoted_price=50,
        implied_probability=0.50, yes_bid=49, yes_ask=51,
        volume=0, open_interest=0, source="test",
        timestamp=datetime.now(timezone.utc),
    )
    report = InferenceReport(
        focus_market=snapshot, context_markets=[],
        consistency_analysis="",
        derived_probabilities=[],
        detected_mispricings=[],  # empty — no fair price available
        suggested_edges=[Edge(
            ticker="X-1", title="x", side="yes", confidence="medium",
            thesis="", kelly_fraction=0.1,
        )],
    )
    model = RawLLMSignal()
    edges = model.signals(snapshot, [], report, HistoricalContext())
    assert edges == []
