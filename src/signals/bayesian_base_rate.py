"""
BayesianBaseRateSignal — empirical prior + Bayesian update.

The LLM finds analogous *resolved* markets (via the existing semantic-similarity
+ rerank pipeline). We compute the empirical base rate of YES outcomes among
those analogues. We treat that base rate as the prior, treat the current market
price as a noisy likelihood, and produce a posterior probability.

Key property: posterior is grounded in *actual outcomes*, not LLM speculation.
Best when the LLM identifies many analogous resolved markets (recurring
"will X be above Y by date" type patterns).
"""
from __future__ import annotations

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import quarter_kelly
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BayesianBaseRateSignal:
    name = "bayesian_base_rate"

    def __init__(
        self,
        min_analogues: int = 5,
        market_noise_kappa: float = 6.0,
        min_edge_pp: float = 0.03,
    ):
        # `kappa` controls how much weight the market price has vs. the prior.
        # Higher kappa → posterior pulls harder toward the market price.
        self.min_analogues = min_analogues
        self.kappa = market_noise_kappa
        self.min_edge_pp = min_edge_pp

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        analogues = self._find_analogues(focus, context, history)
        if len(analogues) < self.min_analogues:
            return []

        prior = sum(analogues) / len(analogues)
        prior = max(0.02, min(0.98, prior))  # keep away from 0/1 for numerical stability

        implied = focus.implied_probability
        if implied <= 0 or implied >= 1:
            return []

        posterior = _bayesian_update(prior=prior, market_implied=implied, kappa=self.kappa)
        edge_pp = posterior - implied

        if abs(edge_pp) < self.min_edge_pp:
            return []

        side = "yes" if edge_pp > 0 else "no"
        kf = quarter_kelly(posterior, implied, side)
        if kf <= 0:
            return []

        return [CalibratedEdge(
            ticker=focus.market, title=focus.event, side=side,
            estimated_fair_prob=posterior,
            current_implied_prob=implied,
            edge_pp=edge_pp if side == "yes" else -edge_pp,
            confidence=min(1.0, len(analogues) / 30.0 + abs(edge_pp) * 3),
            kelly_fraction=kf,
            thesis=(
                f"Empirical base rate {prior:.2%} from {len(analogues)} analogous resolved markets. "
                f"Bayesian posterior given market={implied:.2%} → {posterior:.2%}."
            ),
            source_signal_model=self.name,
            metadata={"prior": prior, "n_analogues": len(analogues)},
        )]

    def _find_analogues(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        history: HistoricalContext,
    ) -> list[float]:
        """
        Pull resolved analogues. Strategy: any LLM-grouped context market that
        has a known settlement_value in `history.resolutions` counts as an analogue.
        Fallback: same focus ticker family resolved before.
        """
        analogues: list[float] = []
        for c in context:
            rec = history.resolutions.get(c.ticker)
            if rec is None:
                continue
            sv = getattr(rec, "settlement_value", None) or (rec.get("settlement_value") if isinstance(rec, dict) else None)
            if sv is not None:
                analogues.append(float(sv))
        return analogues


def _bayesian_update(prior: float, market_implied: float, kappa: float) -> float:
    """
    Beta-style update: turn prior into Beta(α, β) with mean=prior and concentration kappa,
    update by treating market price as one observation worth `kappa` pseudo-counts.
    Closed form: posterior_mean = (kappa * prior + kappa * market) / (2 * kappa) — but
    we want to weight by prior_strength. Use: posterior = w * prior + (1-w) * market_implied,
    where w shrinks as kappa rises (more weight on market).
    """
    w_market = kappa / (kappa + 1.0)
    posterior = (1 - w_market) * prior + w_market * market_implied
    # blend a small pull from the prior toward the market — guards against
    # over-aggressive priors when n_analogues is high but prior is extreme
    posterior = max(0.01, min(0.99, posterior))
    # If there's a meaningful gap and prior is very different, nudge a bit toward prior
    gap = prior - market_implied
    return max(0.01, min(0.99, posterior + 0.10 * gap))
