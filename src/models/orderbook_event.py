from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class OrderLevel(BaseModel):
    price: int   # cents
    quantity: int


class OrderBookEvent(BaseModel):
    market_id: str
    ticker: str
    event_type: Literal["snapshot", "delta"]
    # Populated on snapshot
    yes_levels: list[OrderLevel] = []
    no_levels: list[OrderLevel] = []
    # Populated on delta
    delta_side: Optional[Literal["yes", "no"]] = None
    delta_price: Optional[int] = None
    delta_quantity: Optional[int] = None   # negative = removal
    source: str
    timestamp: datetime
