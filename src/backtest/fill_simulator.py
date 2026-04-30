"""
Fill simulator.

Given a `CalibratedEdge` and the market state at time t, decide whether the
order would have filled and at what price. Two models:

- `midpoint`: fill at midpoint + slippage (always fills). Simple, optimistic.
- `trade_match`: only fills when the recorded trade tape crosses the limit
  price within ±5 minutes of t. Conservative.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from src.backtest.models import Candle, SimulatedFill
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FillSimulator:
    def __init__(
        self,
        model: str = "midpoint",
        slippage_per_contract: float = 0.01,
        fee_per_contract: float = 0.07,
    ):
        if model not in {"midpoint", "trade_match"}:
            raise ValueError(f"Unsupported fill model: {model!r}")
        self.model = model
        self.slippage = slippage_per_contract
        self.fee = fee_per_contract

    def simulate(
        self,
        edge: CalibratedEdge,
        snapshot: MarketSnapshot,
        contracts: int,
        t: datetime,
        recent_candles: list[Candle] | None = None,
    ) -> SimulatedFill | None:
        """Return a SimulatedFill or None if the order didn't fill."""
        if contracts <= 0:
            return None

        if self.model == "midpoint":
            return self._midpoint_fill(edge, snapshot, contracts, t)
        if self.model == "trade_match":
            return self._trade_match_fill(edge, snapshot, contracts, t, recent_candles or [])
        return None

    def _midpoint_fill(
        self,
        edge: CalibratedEdge,
        snapshot: MarketSnapshot,
        contracts: int,
        t: datetime,
    ) -> SimulatedFill:
        # mid in dollars (not cents)
        mid = (snapshot.yes_bid + snapshot.yes_ask) / 2 / 100.0
        if edge.side == "yes":
            fill_price = mid + self.slippage
        else:
            fill_price = (1.0 - mid) + self.slippage
        fill_price = max(0.01, min(0.99, fill_price))

        return SimulatedFill(
            ticker=edge.ticker, side=edge.side, contracts=contracts,
            price=round(fill_price, 4),
            fee=round(self.fee * contracts, 4),
            timestamp=t,
            signal_model=edge.source_signal_model,
            edge_thesis=edge.thesis,
            estimated_fair_prob=edge.estimated_fair_prob,
            current_implied_prob=edge.current_implied_prob,
        )

    def _trade_match_fill(
        self,
        edge: CalibratedEdge,
        snapshot: MarketSnapshot,
        contracts: int,
        t: datetime,
        recent_candles: list[Candle],
    ) -> SimulatedFill | None:
        """Look for a candle within ±5min of t whose range crosses our limit price."""
        window = timedelta(minutes=5)
        nearby = [c for c in recent_candles if abs((c.timestamp - t).total_seconds()) <= window.total_seconds()]
        if not nearby:
            return None

        # Limit price: yes side = pay current ask + slippage; no side = symmetric
        if edge.side == "yes":
            limit_cents = snapshot.yes_ask + self.slippage * 100
            crossed = any(c.low <= limit_cents for c in nearby)
            fill_cents = min(c.low for c in nearby) if crossed else None
        else:
            no_ask = 100 - snapshot.yes_bid
            limit_cents = no_ask + self.slippage * 100
            crossed = any((100 - c.high) <= limit_cents for c in nearby)
            fill_cents = min((100 - c.high) for c in nearby) if crossed else None

        if not crossed or fill_cents is None:
            return None

        fill_price = max(0.01, min(0.99, fill_cents / 100.0))
        return SimulatedFill(
            ticker=edge.ticker, side=edge.side, contracts=contracts,
            price=round(fill_price, 4),
            fee=round(self.fee * contracts, 4),
            timestamp=t,
            signal_model=edge.source_signal_model,
            edge_thesis=edge.thesis,
            estimated_fair_prob=edge.estimated_fair_prob,
            current_implied_prob=edge.current_implied_prob,
        )
