from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class TradeEvent(BaseModel):
    market_id: str
    ticker: str
    yes_price: int      # price in cents
    no_price: int
    count: int          # number of contracts
    taker_side: Literal["yes", "no"]
    source: str
    timestamp: datetime
