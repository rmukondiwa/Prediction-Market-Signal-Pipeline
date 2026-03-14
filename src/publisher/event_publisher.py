from typing import Union

import redis.asyncio as aioredis

from src.config.redis_config import RedisConfig
from src.models.market_event import MarketEvent
from src.models.orderbook_event import OrderBookEvent
from src.models.trade_event import TradeEvent
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

NormalizedEvent = Union[MarketEvent, TradeEvent, OrderBookEvent]


class EventPublisher:
    """
    Serializes normalized events and publishes them to Redis Streams.

    Each event type is routed to a dedicated stream:
      - MarketEvent    → market_events
      - TradeEvent     → trade_events
      - OrderBookEvent → orderbook_events

    Downstream consumers (storage engine, signal engine) read from these
    streams independently without coupling to Kalshi wire formats.
    """

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._redis: aioredis.Redis | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._config.url,
            decode_responses=True,
        )
        # Verify connection
        await self._redis.ping()
        log_event(logger, "redis_connected", url=self._config.url)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            log_event(logger, "redis_disconnected")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event: NormalizedEvent) -> None:
        if self._redis is None:
            raise RuntimeError("EventPublisher.connect() must be called before publish()")

        stream, fields = self._route(event)
        await self._redis.xadd(
            stream,
            fields,
            maxlen=self._config.stream_max_len,
            approximate=True,
        )
        logger.debug(
            "Event published",
            extra={
                "stream": stream,
                "event_type": type(event).__name__,
                "ticker": getattr(event, "ticker", ""),
            },
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, event: NormalizedEvent) -> tuple[str, dict[str, str]]:
        """Map an event to its target stream name and field dict."""
        match event:
            case MarketEvent():
                return self._config.market_events_stream, self._market_fields(event)
            case TradeEvent():
                return self._config.trade_events_stream, self._trade_fields(event)
            case OrderBookEvent():
                return self._config.orderbook_events_stream, self._orderbook_fields(event)
            case _:
                raise TypeError(f"Unsupported event type: {type(event)}")

    @staticmethod
    def _market_fields(event: MarketEvent) -> dict[str, str]:
        return {
            "market_id": event.market_id,
            "ticker": event.ticker,
            "yes_bid": str(event.yes_bid),
            "yes_ask": str(event.yes_ask),
            "no_bid": str(event.no_bid),
            "no_ask": str(event.no_ask),
            "last_price": str(event.last_price),
            "volume": str(event.volume),
            "open_interest": str(event.open_interest),
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "data": event.model_dump_json(),
        }

    @staticmethod
    def _trade_fields(event: TradeEvent) -> dict[str, str]:
        return {
            "market_id": event.market_id,
            "ticker": event.ticker,
            "yes_price": str(event.yes_price),
            "no_price": str(event.no_price),
            "count": str(event.count),
            "taker_side": event.taker_side,
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "data": event.model_dump_json(),
        }

    @staticmethod
    def _orderbook_fields(event: OrderBookEvent) -> dict[str, str]:
        return {
            "market_id": event.market_id,
            "ticker": event.ticker,
            "event_type": event.event_type,
            "delta_side": event.delta_side or "",
            "delta_price": str(event.delta_price) if event.delta_price is not None else "",
            "delta_quantity": str(event.delta_quantity) if event.delta_quantity is not None else "",
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "data": event.model_dump_json(),
        }
