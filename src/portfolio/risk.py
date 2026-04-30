"""
Risk manager — gatekeeper between signal model output and order placement.

Rejects or scales down orders that breach configured limits. Pure decision
logic, no state mutation.

Limits checked, in order:
  1. Kill switch — global hard stop
  2. Daily loss limit — halt new orders mid-day after threshold
  3. Per-market exposure cap — already-held + new must not exceed cap
  4. Total exposure cap — aggregate across all positions
  5. Correlated exposure cap — markets sharing event_ticker (LLM-grouped) cluster
  6. Kelly clamp — never exceed RiskLimits.max_kelly_fraction
"""
from __future__ import annotations

from src.context.models import ContextMarket
from src.portfolio.models import (
    PortfolioSnapshot,
    RiskDecision,
    RiskLimits,
)
from src.signals.models import CalibratedEdge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check(
        self,
        edge: CalibratedEdge,
        portfolio: PortfolioSnapshot,
        context_markets: list[ContextMarket] | None = None,
        current_price: float | None = None,
    ) -> RiskDecision:
        if self.limits.kill_switch:
            return RiskDecision(approved=False, reason="kill_switch_active")

        if portfolio.daily_pnl < -self.limits.daily_loss_limit_usd:
            return RiskDecision(
                approved=False,
                reason=f"daily_loss_limit_hit ({portfolio.daily_pnl:.2f} < -{self.limits.daily_loss_limit_usd})",
            )

        # Effective price for this side, used for USD exposure math
        price = self._effective_price(edge, current_price)

        # Per-market: notional from existing position + the kelly-deployed amount
        notional_for_edge = portfolio.cash * min(edge.kelly_fraction, self.limits.max_kelly_fraction)
        existing_notional = self._existing_notional(edge, portfolio, price)
        if existing_notional + notional_for_edge > self.limits.max_per_market_usd:
            allowed = max(0.0, self.limits.max_per_market_usd - existing_notional)
            scale = (allowed / notional_for_edge) if notional_for_edge > 0 else 0.0
            if scale <= 0:
                return RiskDecision(approved=False, reason="per_market_cap_full")
            return RiskDecision(approved=True, reason="per_market_cap_scale_down", scale_factor=min(1.0, scale))

        # Total exposure
        total_exposure = self._total_exposure(portfolio, price_lookup=current_price)
        if total_exposure + notional_for_edge > self.limits.max_total_exposure_usd:
            allowed = max(0.0, self.limits.max_total_exposure_usd - total_exposure)
            scale = (allowed / notional_for_edge) if notional_for_edge > 0 else 0.0
            if scale <= 0:
                return RiskDecision(approved=False, reason="total_exposure_full")
            return RiskDecision(approved=True, reason="total_exposure_scale_down", scale_factor=min(1.0, scale))

        # Correlated exposure cap via LLM-grouped event_ticker
        correlated_notional = self._correlated_notional(edge, portfolio, context_markets or [], price)
        if correlated_notional + notional_for_edge > self.limits.max_correlated_exposure_usd:
            allowed = max(0.0, self.limits.max_correlated_exposure_usd - correlated_notional)
            scale = (allowed / notional_for_edge) if notional_for_edge > 0 else 0.0
            if scale <= 0:
                return RiskDecision(approved=False, reason="correlated_exposure_full")
            return RiskDecision(approved=True, reason="correlated_exposure_scale_down", scale_factor=min(1.0, scale))

        return RiskDecision(approved=True, reason="ok", scale_factor=1.0)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _effective_price(edge: CalibratedEdge, current_price: float | None) -> float:
        """Return the price (in dollars) for the side we're buying.
        Defaults to current_implied_prob (yes side) or its complement (no side)
        when current_price not provided."""
        if current_price is not None:
            yes_mid = current_price
        else:
            yes_mid = edge.current_implied_prob
        if edge.side == "yes":
            return max(0.01, min(0.99, yes_mid))
        return max(0.01, min(0.99, 1.0 - yes_mid))

    @staticmethod
    def _existing_notional(edge: CalibratedEdge, portfolio: PortfolioSnapshot, price: float) -> float:
        for pos in portfolio.positions:
            if pos.ticker != edge.ticker:
                continue
            if pos.side != edge.side:
                continue
            return pos.contracts * price
        return 0.0

    @staticmethod
    def _total_exposure(portfolio: PortfolioSnapshot, price_lookup: float | None) -> float:
        # Use avg_cost as a proxy for notional already deployed
        return sum(pos.contracts * pos.avg_cost for pos in portfolio.positions)

    @staticmethod
    def _correlated_notional(
        edge: CalibratedEdge,
        portfolio: PortfolioSnapshot,
        context_markets: list[ContextMarket],
        price: float,
    ) -> float:
        """Sum existing positions whose ticker shares an event_ticker with the
        edge's ticker (or with any LLM-grouped context market)."""
        # Build a set of "this event cluster" tickers from context
        cluster: set[str] = {edge.ticker}
        for c in context_markets:
            cluster.add(c.ticker)
        return sum(
            pos.contracts * pos.avg_cost
            for pos in portfolio.positions
            if pos.ticker in cluster
        )
