"""
SettlementDecaySignal — buy near-certain YES (or near-certain NO) close to
resolution and harvest the residual gap.

Theoretical basis: probability axiom — at settlement t = T, P(outcome) ∈ {0, 1}
exactly. If implied is 0.97 with hours to close, the residual 3¢ is paying
for tail risk that shrinks toward zero as t → T. Strategy: buy YES at the
ask when implied ≥ threshold AND time-to-close ≤ window. No LLM cost.

Stateless: requires HistoricalContext.as_of (current backtest timestep) plus
a settlement_times dict (per-ticker scheduled close timestamps).
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import quarter_kelly
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SettlementDecaySignal:
    name = "settlement_decay"

    def __init__(
        self,
        min_implied_prob: float = 0.95,
        max_hours_to_close: float = 72.0,
        no_side_max_implied: float = 0.05,  # mirror: buy NO at low implied
        fair_prob_when_yes: float = 0.99,   # what we estimate fair to be
        fair_prob_when_no: float = 0.01,
    ):
        self.min_implied = min_implied_prob
        self.max_hours = max_hours_to_close
        self.no_max_implied = no_side_max_implied
        self.fair_yes = fair_prob_when_yes
        self.fair_no = fair_prob_when_no

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        # Need to know when this market closes
        settlement_times: dict = (
            getattr(history, "settlement_times", None)
            or (history.model_extra or {}).get("settlement_times")
            or history.__dict__.get("settlement_times")
            or {}
        )
        close_t = settlement_times.get(focus.market) if settlement_times else None
        if close_t is None:
            return []

        as_of = history.as_of or datetime.now(timezone.utc)
        if close_t.tzinfo is None:
            close_t = close_t.replace(tzinfo=timezone.utc)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        hours_to_close = (close_t - as_of).total_seconds() / 3600.0
        # Don't trade after close
        if hours_to_close <= 0:
            return []
        if hours_to_close > self.max_hours:
            return []

        implied = focus.implied_probability
        edges: list[CalibratedEdge] = []

        # YES side: high implied
        if implied >= self.min_implied:
            kf = quarter_kelly(self.fair_yes, implied, "yes")
            if kf > 0:
                edges.append(CalibratedEdge(
                    ticker=focus.market, title=focus.event, side="yes",
                    estimated_fair_prob=self.fair_yes,
                    current_implied_prob=implied,
                    edge_pp=self.fair_yes - implied,
                    confidence=min(1.0, (self.fair_yes - implied) * 50),
                    kelly_fraction=kf,
                    thesis=(f"Implied {implied:.2%} with {hours_to_close:.1f}h to close. "
                            f"Residual {self.fair_yes - implied:.2%} gap to {self.fair_yes:.0%} "
                            f"is paying for tail risk that shrinks to zero at settlement."),
                    source_signal_model=self.name,
                    metadata={"hours_to_close": hours_to_close,
                              "side": "decay_yes"},
                ))

        # NO side: low implied (mirror)
        if implied <= self.no_max_implied:
            # On the NO side, we buy NO at price (1-implied) when implied is low
            kf = quarter_kelly(self.fair_no, implied, "no")
            if kf > 0:
                edges.append(CalibratedEdge(
                    ticker=focus.market, title=focus.event, side="no",
                    estimated_fair_prob=self.fair_no,
                    current_implied_prob=implied,
                    edge_pp=(1 - self.fair_no) - (1 - implied),
                    confidence=min(1.0, (implied - self.fair_no) * 50),
                    kelly_fraction=kf,
                    thesis=(f"Implied YES {implied:.2%} with {hours_to_close:.1f}h to close. "
                            f"NO side: residual gap to settlement is converging to zero."),
                    source_signal_model=self.name,
                    metadata={"hours_to_close": hours_to_close,
                              "side": "decay_no"},
                ))

        return edges
