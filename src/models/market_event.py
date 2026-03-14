from datetime import datetime
from pydantic import BaseModel


class MarketEvent(BaseModel):
    market_id: str
    ticker: str
    yes_bid: int        # price in cents (0–100)
    yes_ask: int
    no_bid: int
    no_ask: int
    last_price: int
    volume: int
    open_interest: int
    source: str         # exchange identifier e.g. "kalshi"
    timestamp: datetime
