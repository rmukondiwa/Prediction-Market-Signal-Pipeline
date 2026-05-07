"""Portfolio data types — positions, working orders, fills, risk limits."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Side = Literal["yes", "no"]
OrderStatus = Literal["working", "partial", "filled", "canceled"]


class Position(BaseModel):
    ticker: str
    side: Side
    contracts: int
    avg_cost: float  # mean price per contract (in dollars, 0..1)
    opened_at: datetime
    last_updated: datetime
    total_fees: float = 0.0  # cumulative entry fees on this position; defaults to 0
                             # for back-compat with pre-fee-tracking serialized state


class WorkingOrder(BaseModel):
    order_id: str
    ticker: str
    side: Side
    contracts: int
    limit_price: float
    placed_at: datetime
    status: OrderStatus = "working"
    filled_contracts: int = 0


class Fill(BaseModel):
    fill_id: str
    order_id: str | None  # None for backtest synthetic fills
    ticker: str
    side: Side
    contracts: int
    price: float
    fee: float
    timestamp: datetime
    signal_model: str  # for post-trade attribution


class RiskLimits(BaseModel):
    max_per_market_usd: float = 500.0
    max_total_exposure_usd: float = 5_000.0
    max_correlated_exposure_usd: float = 1_500.0
    daily_loss_limit_usd: float = 500.0
    max_kelly_fraction: float = 0.25
    kill_switch: bool = False


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    scale_factor: float = 1.0  # in (0..1] when approved-with-scale


class PortfolioSnapshot(BaseModel):
    cash: float
    positions: list[Position] = Field(default_factory=list)
    working_orders: list[WorkingOrder] = Field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    timestamp: datetime
