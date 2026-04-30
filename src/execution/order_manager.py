"""
Order manager — translates a CalibratedEdge + sized contract count into an
OrderRequest, places it via the trading client, and persists a WorkingOrder
to portfolio state.

Limit-price strategy is configurable; default is mid-aware passive placement
(yes side: post at current ask; no side: post at 100 - yes_bid).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.execution.kalshi_trading_client import KalshiTradingClient
from src.execution.models import OrderRequest, OrderResult
from src.insight.models import MarketSnapshot
from src.portfolio.models import WorkingOrder
from src.portfolio.state import PortfolioState
from src.signals.models import CalibratedEdge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OrderManager:
    def __init__(
        self,
        client: KalshiTradingClient,
        portfolio: PortfolioState,
        time_in_force: str = "gtc",
    ):
        self.client = client
        self.portfolio = portfolio
        self.tif = time_in_force

    @staticmethod
    def _client_order_id() -> str:
        return f"phase5-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _limit_price(edge: CalibratedEdge, snapshot: MarketSnapshot) -> float:
        """Passive-placement default: yes side posts at current ask;
        no side posts at (100 - yes_bid). In dollars (0..1)."""
        if edge.side == "yes":
            return max(0.01, min(0.99, snapshot.yes_ask / 100.0))
        return max(0.01, min(0.99, (100 - snapshot.yes_bid) / 100.0))

    async def place(
        self,
        edge: CalibratedEdge,
        snapshot: MarketSnapshot,
        contracts: int,
    ) -> OrderResult:
        if contracts <= 0:
            return OrderResult(accepted=False, order_id=None, error="zero_contracts", raw_response={})

        limit = self._limit_price(edge, snapshot)
        request = OrderRequest(
            ticker=edge.ticker, side=edge.side, contracts=contracts,
            order_type="limit", limit_price=limit,
            time_in_force=self.tif,
            client_order_id=self._client_order_id(),
            placed_at=datetime.now(timezone.utc),
        )

        result = await self.client.place_order(request)
        if result.accepted and result.order_id is not None:
            await self.portfolio.add_working_order(WorkingOrder(
                order_id=result.order_id,
                ticker=edge.ticker, side=edge.side,
                contracts=contracts, limit_price=limit,
                placed_at=request.placed_at,
            ))
            logger.info("Order placed and persisted", extra={
                "ticker": edge.ticker, "order_id": result.order_id,
                "contracts": contracts, "limit": limit,
                "signal_model": edge.source_signal_model,
            })
        else:
            logger.warning("Order rejected by client", extra={
                "ticker": edge.ticker, "error": result.error,
            })
        return result
