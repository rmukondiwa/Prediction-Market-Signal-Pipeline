"""
RawLLMSignal — the baseline.

Pass-through of the LLM's `Edge.kelly_fraction` and `estimated_fair_prob`
without any empirical correction. This is the model that every other signal
model must beat to justify its existence.
"""
from __future__ import annotations

from src.context.models import ContextMarket
from src.inference.models import Edge, InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext

_CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 0.9}


def _confidence_to_float(s: str) -> float:
    return _CONFIDENCE_MAP.get(s.lower(), 0.5)


def _implied_for(ticker: str, llm_report: InferenceReport) -> float | None:
    """Find the current implied probability for a ticker, looking in
    detected_mispricings (most precise) then context_markets, then focus."""
    for m in llm_report.detected_mispricings:
        if m.ticker == ticker:
            return m.current_implied_prob
    for c in llm_report.context_markets:
        if c.ticker == ticker:
            return c.implied_probability
    if llm_report.focus_market.market == ticker:
        return llm_report.focus_market.implied_probability
    return None


def _fair_for(ticker: str, llm_report: InferenceReport) -> float | None:
    for m in llm_report.detected_mispricings:
        if m.ticker == ticker:
            return m.estimated_fair_prob
    return None


class RawLLMSignal:
    name = "raw_llm"

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        out: list[CalibratedEdge] = []
        for edge in llm_report.suggested_edges:
            implied = _implied_for(edge.ticker, llm_report)
            fair = _fair_for(edge.ticker, llm_report)
            if implied is None or fair is None:
                continue
            edge_pp = fair - implied
            out.append(
                CalibratedEdge(
                    ticker=edge.ticker,
                    title=edge.title,
                    side=edge.side,
                    estimated_fair_prob=fair,
                    current_implied_prob=implied,
                    edge_pp=edge_pp,
                    confidence=_confidence_to_float(edge.confidence),
                    kelly_fraction=edge.kelly_fraction,
                    thesis=edge.thesis,
                    source_signal_model=self.name,
                    metadata={"llm_confidence": edge.confidence},
                )
            )
        return out
