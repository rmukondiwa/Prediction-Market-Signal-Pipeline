"""
Paper executor.

Wires CalibratedEdge → risk gate → sizer → order manager. Never sends real
orders: the trading client is a stub by default. Simulates fills against the
provided live market snapshot using the same logic as the backtest fill
simulator, then persists the synthetic Fill to PortfolioState under env=paper.

Designed to be invoked once per inference cycle from `scripts/paper_trade.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from src.context.models import ContextMarket
from src.execution.kalshi_trading_client import KalshiTradingClient
from src.execution.order_manager import OrderManager
from src.insight.models import MarketSnapshot
from src.portfolio.models import Fill, RiskLimits
from src.portfolio.risk import RiskManager
from src.portfolio.sizer import QuarterKellySizer
from src.portfolio.state import PortfolioState
from src.signals.models import CalibratedEdge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PaperExecutor:
    def __init__(
        self,
        portfolio: PortfolioState,
        client: KalshiTradingClient,
        risk_limits: RiskLimits,
        slippage_per_contract: float = 0.01,
        fee_per_contract: float = 0.07,
    ):
        self.portfolio = portfolio
        self.client = client
        self.risk_manager = RiskManager(risk_limits)
        self.sizer = QuarterKellySizer()
        self.order_manager = OrderManager(client, portfolio)
        self.slippage = slippage_per_contract
        self.fee = fee_per_contract

    async def process_edges(
        self,
        edges: list[CalibratedEdge],
        snapshot: MarketSnapshot,
        context: list[ContextMarket],
    ) -> list[Fill]:
        """Run each edge through risk → sizer → order placement → simulated fill.
        Returns the list of synthetic fills persisted."""
        portfolio_snapshot = await self.portfolio.snapshot()
        fills: list[Fill] = []

        for edge in edges:
            decision = self.risk_manager.check(edge, portfolio_snapshot, context)
            if not decision.approved:
                logger.info("Edge rejected by risk", extra={
                    "ticker": edge.ticker, "reason": decision.reason,
                    "signal_model": edge.source_signal_model,
                })
                continue

            current_price = snapshot.implied_probability
            contracts = self.sizer.size(edge, portfolio_snapshot.cash, decision, current_price)
            if contracts <= 0:
                continue

            order_result = await self.order_manager.place(edge, snapshot, contracts)
            if not order_result.accepted:
                continue

            # Simulate the fill at midpoint + slippage. The OrderManager already
            # logged the limit price; we use mid for the actual fill price.
            yes_mid = current_price
            if edge.side == "yes":
                fill_price = max(0.01, min(0.99, yes_mid + self.slippage))
            else:
                fill_price = max(0.01, min(0.99, (1.0 - yes_mid) + self.slippage))

            fill = Fill(
                fill_id=f"paper-{uuid.uuid4().hex[:12]}",
                order_id=order_result.order_id,
                ticker=edge.ticker, side=edge.side,
                contracts=contracts, price=round(fill_price, 4),
                fee=round(self.fee * contracts, 4),
                timestamp=datetime.now(timezone.utc),
                signal_model=edge.source_signal_model,
            )
            await self.portfolio.apply_fill(fill)
            # Mark the working order filled
            if order_result.order_id:
                await self.portfolio.update_order(order_result.order_id, "filled", contracts)
            fills.append(fill)
            logger.info("Paper fill applied", extra={
                "ticker": fill.ticker, "contracts": fill.contracts,
                "price": fill.price, "signal_model": fill.signal_model,
            })

        return fills
