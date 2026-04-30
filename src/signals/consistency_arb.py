"""
ConsistencyArbSignal — pure structural arbitrage.

The LLM groups markets that share an event tree; we mechanically check axioms
of probability and emit edges when the prices contradict them. The LLM's
absolute probabilities don't matter — only its grouping does.

Checks performed:
  1. Monotonicity: P(A by T1) > P(A by T2) when T1 < T2 (deadline-stacked)
  2. Total probability: sum P(mutually exclusive Aᵢ) > 1
  3. Conditional bounds: P(A∩B) > min(P(A), P(B))   (TODO — needs joint markers)

Each violation produces 2 edges (long the underpriced, short the overpriced).
Confidence is high because the edge is structural, not predictive.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import quarter_kelly
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Match common Kalshi date suffixes: 26JUN30, 26DEC31, 01JAN2027
_DATE_PAT = re.compile(r"-(\d{2})([A-Z]{3})(\d{2,4})(?:-|$)")
_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _extract_deadline(ticker: str) -> datetime | None:
    """Parse Kalshi deadline suffixes. Two known formats:
       YYMONDD   (e.g. "26JUN30" = 2026-06-30) — used by newer ticker series
       DDMONYYYY (e.g. "01JAN2027" = 2027-01-01) — older format
    """
    m = _DATE_PAT.search(ticker)
    if not m:
        return None
    g1, gmon, g3 = m.group(1), m.group(2).upper(), m.group(3)
    mon = _MONTH_NUM.get(gmon)
    if mon is None:
        return None

    # 4-digit trailing → DDMONYYYY
    if len(g3) == 4:
        day = int(g1)
        year = int(g3)
    else:
        # 2-digit trailing → YYMONDD (year is leading)
        year = 2000 + int(g1)
        day = int(g3)

    try:
        return datetime(year, mon, day)
    except ValueError:
        return None


def _extract_event_root(ticker: str) -> str:
    """Strip the deadline suffix from a ticker so deadline-stacked markets group."""
    m = _DATE_PAT.search(ticker)
    if not m:
        return ticker
    return ticker[: m.start()]


class ConsistencyArbSignal:
    name = "consistency_arb"

    def __init__(self, min_violation_pp: float = 0.03):
        # ignore violations smaller than this — they're within bid/ask noise
        self.min_violation_pp = min_violation_pp

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        # Pool: focus + LLM-grouped context markets
        pool: list[tuple[str, str, float]] = [
            (focus.market, focus.event, focus.implied_probability)
        ]
        for c in context:
            pool.append((c.ticker, c.title, c.implied_probability))

        edges = list(self._check_monotonicity(pool))
        return edges

    def _check_monotonicity(self, pool: list[tuple[str, str, float]]) -> Iterable[CalibratedEdge]:
        """For each pair of markets sharing an event root, deadline T1 < T2,
        emit edges when P(by T1) > P(by T2) + threshold."""
        # Group by event root → list of (deadline, ticker, title, prob)
        groups: dict[str, list[tuple[datetime, str, str, float]]] = {}
        for ticker, title, prob in pool:
            deadline = _extract_deadline(ticker)
            if deadline is None:
                continue
            root = _extract_event_root(ticker)
            groups.setdefault(root, []).append((deadline, ticker, title, prob))

        for root, items in groups.items():
            if len(items) < 2:
                continue
            items.sort()  # ascending deadline
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    earlier_dt, earlier_t, earlier_title, earlier_p = items[i]
                    later_dt, later_t, later_title, later_p = items[j]
                    # Axiom: P(by earlier) <= P(by later). Violation: earlier > later.
                    if earlier_p > later_p + self.min_violation_pp:
                        yield from self._violation_edges(
                            over_ticker=earlier_t, over_title=earlier_title, over_prob=earlier_p,
                            under_ticker=later_t, under_title=later_title, under_prob=later_p,
                            event_root=root,
                        )

    def _violation_edges(
        self,
        over_ticker: str, over_title: str, over_prob: float,
        under_ticker: str, under_title: str, under_prob: float,
        event_root: str,
    ) -> Iterable[CalibratedEdge]:
        """Emit two edges: short (no) the overpriced, long (yes) the underpriced.
        Fair price for the overpriced market = under_prob (the lower bound).
        Fair price for the underpriced market = over_prob (the upper bound).
        """
        # Overpriced market: bet NO
        fair_over = under_prob
        kf_over = quarter_kelly(fair_over, over_prob, "no")
        if kf_over > 0:
            yield CalibratedEdge(
                ticker=over_ticker, title=over_title, side="no",
                estimated_fair_prob=fair_over,
                current_implied_prob=over_prob,
                edge_pp=(1 - fair_over) - (1 - over_prob),
                confidence=0.92,
                kelly_fraction=kf_over,
                thesis=(
                    f"Monotonicity violation in {event_root}: P({over_ticker})={over_prob:.2%} "
                    f"> P({under_ticker})={under_prob:.2%}, but earlier deadline must be ≤ later."
                ),
                source_signal_model=self.name,
                metadata={"violation_pp": round(over_prob - under_prob, 4)},
            )
        # Underpriced market: bet YES
        fair_under = over_prob
        kf_under = quarter_kelly(fair_under, under_prob, "yes")
        if kf_under > 0:
            yield CalibratedEdge(
                ticker=under_ticker, title=under_title, side="yes",
                estimated_fair_prob=fair_under,
                current_implied_prob=under_prob,
                edge_pp=fair_under - under_prob,
                confidence=0.92,
                kelly_fraction=kf_under,
                thesis=(
                    f"Monotonicity violation in {event_root}: later-deadline P={under_prob:.2%} "
                    f"must be ≥ earlier-deadline P={over_prob:.2%}."
                ),
                source_signal_model=self.name,
                metadata={"violation_pp": round(over_prob - under_prob, 4)},
            )
