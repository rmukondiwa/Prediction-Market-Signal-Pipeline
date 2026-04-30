"""Order placement types — symmetric across paper and live executors."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


OrderType = Literal["limit", "market"]
TIF = Literal["ioc", "gtc", "post_only"]


class OrderRequest(BaseModel):
    ticker: str
    side: Literal["yes", "no"]
    contracts: int
    order_type: OrderType
    limit_price: float | None
    time_in_force: TIF
    client_order_id: str  # idempotency key
    placed_at: datetime


class OrderResult(BaseModel):
    accepted: bool
    order_id: str | None
    error: str | None
    raw_response: dict
