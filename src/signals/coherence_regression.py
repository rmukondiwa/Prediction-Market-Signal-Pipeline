"""
CoherenceRegressionSignal — residual-based signal from cross-market regression.

The LLM groups related markets; we regress the focus market's implied
probability on the related-markets' prices and use the residual as signal.
When everything else moved but the focus market didn't, the residual is large
and points in the direction of "this should move."

Uses a simple ridge regression over the current snapshot. Future improvement:
fit on a recent time window of prices rather than the single current snapshot.
"""
from __future__ import annotations

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import quarter_kelly
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CoherenceRegressionSignal:
    name = "coherence_regression"

    def __init__(self, min_context: int = 4, min_edge_pp: float = 0.04, alpha: float = 1.0):
        self.min_context = min_context
        self.min_edge_pp = min_edge_pp
        self.alpha = alpha  # ridge regularisation

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        if len(context) < self.min_context:
            return []

        # Build synthetic "history": treat each context market's price as one
        # data point, weighted by relevance_score. The "expected" price for the
        # focus market is the relevance-weighted mean of context implied probs.
        weights = [max(0.0, c.relevance_score) for c in context]
        if sum(weights) <= 0:
            return []
        weighted_mean = sum(w * c.implied_probability for w, c in zip(weights, context)) / sum(weights)

        implied = focus.implied_probability
        if implied <= 0 or implied >= 1:
            return []

        # Predicted price = weighted_mean (ridge with single feature collapses to this).
        # Residual = predicted - implied. Positive residual → focus underpriced → buy yes.
        residual = weighted_mean - implied
        if abs(residual) < self.min_edge_pp:
            return []

        # Shrink the residual by the regulariser to reflect uncertainty
        shrunk = residual / (1.0 + self.alpha)
        fair = max(0.01, min(0.99, implied + shrunk))

        side = "yes" if shrunk > 0 else "no"
        kf = quarter_kelly(fair, implied, side)
        if kf <= 0:
            return []

        return [CalibratedEdge(
            ticker=focus.market, title=focus.event, side=side,
            estimated_fair_prob=fair,
            current_implied_prob=implied,
            edge_pp=fair - implied if side == "yes" else (1 - fair) - (1 - implied),
            confidence=min(1.0, abs(shrunk) * 5),
            kelly_fraction=kf,
            thesis=(
                f"Coherence regression: related markets weighted-mean={weighted_mean:.2%}, "
                f"focus implied={implied:.2%}, residual={residual:+.2%}. "
                f"Inferred fair={fair:.2%}."
            ),
            source_signal_model=self.name,
            metadata={
                "n_context": len(context),
                "weighted_mean": weighted_mean,
                "raw_residual": residual,
            },
        )]
