from datetime import datetime, timezone
from typing import Union

from src.ingestion.kalshi.message_parser import MessageType, ParsedMessage
from src.models.market_event import MarketEvent
from src.models.orderbook_event import OrderBookEvent, OrderLevel
from src.models.trade_event import TradeEvent
from src.utils.logging import get_logger

logger = get_logger(__name__)

NormalizedEvent = Union[MarketEvent, TradeEvent, OrderBookEvent]

_SOURCE = "kalshi"


def _ts_to_datetime(ts: int | float | None) -> datetime:
    """Convert a Kalshi Unix timestamp (seconds or ms) to a UTC datetime."""
    if ts is None:
        return datetime.now(tz=timezone.utc)
    # Kalshi sends seconds-precision Unix timestamps
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Normalizer:
    """
    Converts Kalshi-specific ParsedMessage instances into exchange-agnostic
    internal event schemas.

    Downstream components must only depend on the output types (MarketEvent,
    TradeEvent, OrderBookEvent) — never on Kalshi wire formats.
    """

    def normalize(self, parsed: ParsedMessage) -> NormalizedEvent | None:
        """
        Dispatch to the appropriate normalizer based on message type.

        Returns a normalized event, or None if the message type carries no
        actionable market data (e.g. subscribed confirmations).
        """
        match parsed.type:
            case MessageType.TICKER:
                return self._normalize_ticker(parsed)
            case MessageType.TRADE:
                return self._normalize_trade(parsed)
            case MessageType.ORDERBOOK_SNAPSHOT:
                return self._normalize_orderbook_snapshot(parsed)
            case MessageType.ORDERBOOK_DELTA:
                return self._normalize_orderbook_delta(parsed)
            case MessageType.SUBSCRIBED:
                logger.info(
                    "Subscription confirmed",
                    extra={"channel": parsed.msg.get("channel"), "seq": parsed.seq},
                )
                return None
            case _:
                logger.debug(
                    "Skipping non-normalizable message",
                    extra={"type": parsed.type.value},
                )
                return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_ticker(self, parsed: ParsedMessage) -> MarketEvent | None:
        msg = parsed.msg
        ticker = msg.get("market_ticker", "")
        if not ticker:
            logger.warning("Ticker message missing market_ticker", extra={"msg": msg})
            return None

        yes_bid: int = msg.get("yes_bid", 0)
        yes_ask: int = msg.get("yes_ask", 0)

        return MarketEvent(
            market_id=msg.get("market_id", ticker),
            ticker=ticker,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=100 - yes_ask,
            no_ask=100 - yes_bid,
            last_price=msg.get("yes_price", msg.get("last_price", 0)),
            volume=msg.get("volume", 0),
            open_interest=msg.get("open_interest", 0),
            source=_SOURCE,
            timestamp=_ts_to_datetime(msg.get("ts")),
        )

    def _normalize_trade(self, parsed: ParsedMessage) -> TradeEvent | None:
        msg = parsed.msg
        ticker = msg.get("market_ticker", "")
        if not ticker:
            logger.warning("Trade message missing market_ticker", extra={"msg": msg})
            return None

        yes_price: int = msg.get("yes_price", 0)
        taker_side: str = msg.get("taker_side", "yes")

        return TradeEvent(
            market_id=msg.get("market_id", ticker),
            ticker=ticker,
            yes_price=yes_price,
            no_price=100 - yes_price,
            count=msg.get("count", 0),
            taker_side=taker_side if taker_side in ("yes", "no") else "yes",
            source=_SOURCE,
            timestamp=_ts_to_datetime(msg.get("ts")),
        )

    def _normalize_orderbook_snapshot(self, parsed: ParsedMessage) -> OrderBookEvent | None:
        msg = parsed.msg
        ticker = msg.get("market_ticker", "")
        if not ticker:
            logger.warning("Orderbook snapshot missing market_ticker", extra={"msg": msg})
            return None

        yes_levels = [
            OrderLevel(price=int(p), quantity=int(q))
            for p, q in msg.get("yes", [])
        ]
        no_levels = [
            OrderLevel(price=int(p), quantity=int(q))
            for p, q in msg.get("no", [])
        ]

        return OrderBookEvent(
            market_id=msg.get("market_id", ticker),
            ticker=ticker,
            event_type="snapshot",
            yes_levels=yes_levels,
            no_levels=no_levels,
            source=_SOURCE,
            timestamp=_ts_to_datetime(msg.get("ts")),
        )

    def _normalize_orderbook_delta(self, parsed: ParsedMessage) -> OrderBookEvent | None:
        msg = parsed.msg
        ticker = msg.get("market_ticker", "")
        if not ticker:
            logger.warning("Orderbook delta missing market_ticker", extra={"msg": msg})
            return None

        side: str = msg.get("side", "yes")

        return OrderBookEvent(
            market_id=msg.get("market_id", ticker),
            ticker=ticker,
            event_type="delta",
            delta_side=side if side in ("yes", "no") else "yes",
            delta_price=msg.get("price"),
            delta_quantity=msg.get("delta"),
            source=_SOURCE,
            timestamp=_ts_to_datetime(msg.get("ts")),
        )
