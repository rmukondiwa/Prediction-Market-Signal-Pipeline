"""
SignalModel protocol — the central abstraction of the trading layer.

Every signal model takes the LLM `InferenceReport` as input and produces a list
of `CalibratedEdge` outputs. The backtester runs each registered model in
isolation and compares them against a market-implied baseline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext


@runtime_checkable
class SignalModel(Protocol):
    """Contract every signal model must satisfy."""

    name: str

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        """Return zero or more calibrated edges. Empty list = no opinion."""
        ...


def quarter_kelly(estimated_fair_prob: float, current_implied_prob: float, side: str) -> float:
    """
    Compute quarter-Kelly fraction for a single edge.
    Returns 0.0 if there is no edge in the chosen direction.
    Clamped to [0.0, 0.25].
    """
    p_yes = estimated_fair_prob
    q_yes = current_implied_prob

    if q_yes <= 0 or q_yes >= 1:
        return 0.0

    if side == "yes":
        p, q = p_yes, q_yes
    else:
        p, q = 1 - p_yes, 1 - q_yes

    if q <= 0 or q >= 1:
        return 0.0

    edge = p - q
    payout = 1.0 / q - 1.0
    if payout <= 0 or edge <= 0:
        return 0.0

    kelly = (p * payout - (1 - p)) / payout
    quarter = kelly * 0.25
    return max(0.0, min(0.25, quarter))
