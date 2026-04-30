"""Position sizer: turns a CalibratedEdge into a contract count."""
from __future__ import annotations

from src.portfolio.models import RiskDecision
from src.signals.models import CalibratedEdge


class QuarterKellySizer:
    """Quarter-Kelly sizing in contracts.

    The signal model's `kelly_fraction` is already quarter-Kelly clamped to
    [0.0, 0.25]. The risk manager may scale it down further. We then convert
    USD-to-deploy into a contract count using the side-aware price.
    """

    def __init__(self, max_contracts_per_order: int = 10_000):
        self.max_contracts_per_order = max_contracts_per_order

    def size(
        self,
        edge: CalibratedEdge,
        available_cash: float,
        decision: RiskDecision,
        current_price: float,
    ) -> int:
        if not decision.approved:
            return 0
        if available_cash <= 0:
            return 0

        kelly = max(0.0, min(0.25, edge.kelly_fraction)) * decision.scale_factor
        if kelly <= 0:
            return 0

        cash_to_deploy = available_cash * kelly

        # Side-aware price: on yes side pay yes_mid; on no side pay (1 - yes_mid).
        if edge.side == "yes":
            price = max(0.01, min(0.99, current_price))
        else:
            price = max(0.01, min(0.99, 1.0 - current_price))

        contracts = int(cash_to_deploy // price)
        contracts = max(0, min(contracts, self.max_contracts_per_order))
        return contracts
