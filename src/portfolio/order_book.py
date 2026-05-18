"""
Live in-memory order book aggregator for Kalshi markets.

Applies orderbook_snapshot and orderbook_delta events to maintain a
per-ticker book of {price_cents: quantity} for the YES and NO sides.

Kalshi's order book structure:
  yes levels = bids for YES (people willing to buy YES at a given price)
  no  levels = bids for NO  (people willing to buy NO  at a given price)

Derived quotes:
  best_bid = max(yes level prices)        — highest price to buy YES
  best_ask = 100 - max(no level prices)   — lowest price to sell YES
             (a 65¢ NO bid implies a 35¢ YES ask)
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.orderbook_event import OrderBookEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LiveBook:
    """Per-ticker live order book maintained by applying WS events."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes: dict[int, int] = {}   # price_cents → quantity
        self.no: dict[int, int] = {}
        self.last_updated: datetime = datetime.now(tz=timezone.utc)
        self.event_count: int = 0

    def apply(self, event: OrderBookEvent) -> None:
        if event.event_type == "snapshot":
            self.yes = {lvl.price: lvl.quantity for lvl in event.yes_levels}
            self.no = {lvl.price: lvl.quantity for lvl in event.no_levels}
        else:
            book = self.yes if event.delta_side == "yes" else self.no
            qty = event.delta_quantity
            if qty is None or qty <= 0:
                book.pop(event.delta_price, None)
            else:
                book[event.delta_price] = qty
        self.last_updated = event.timestamp
        self.event_count += 1

    @property
    def best_bid(self) -> int | None:
        """Highest price (cents) at which someone will buy YES."""
        active = [p for p, q in self.yes.items() if q > 0]
        return max(active) if active else None

    @property
    def best_ask(self) -> int | None:
        """Lowest price (cents) at which someone will sell YES."""
        active = [p for p, q in self.no.items() if q > 0]
        return (100 - max(active)) if active else None

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def is_initialised(self) -> bool:
        return bool(self.yes or self.no)


class OrderBookAggregator:
    """Holds a LiveBook for each subscribed ticker and routes incoming events."""

    def __init__(self) -> None:
        self._books: dict[str, LiveBook] = {}

    def on_event(self, event: OrderBookEvent) -> None:
        if event.ticker not in self._books:
            self._books[event.ticker] = LiveBook(event.ticker)
        self._books[event.ticker].apply(event)

    def get_book(self, ticker: str) -> LiveBook | None:
        return self._books.get(ticker)

    def book_count(self) -> int:
        return len(self._books)

    def initialised_count(self) -> int:
        return sum(1 for b in self._books.values() if b.is_initialised())
