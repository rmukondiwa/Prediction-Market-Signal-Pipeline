"""
Prediction Market Signal Pipeline — Ingestion Layer entry point.

Boots the Kalshi WebSocket client and wires it to the Redis event publisher.
All credentials and settings are read from environment variables (see .env.example).
"""

import asyncio
from dotenv import load_dotenv

from src.config.kalshi_config import KalshiConfig
from src.config.redis_config import RedisConfig
from src.ingestion.kalshi.websocket_client import KalshiWebSocketClient
from src.publisher.event_publisher import EventPublisher
from src.utils.logging import get_logger, log_event

load_dotenv()

logger = get_logger(__name__)


async def main() -> None:
    kalshi_cfg = KalshiConfig()
    redis_cfg = RedisConfig()

    publisher = EventPublisher(redis_cfg)
    await publisher.connect()

    log_event(
        logger,
        "pipeline_starting",
        tickers=kalshi_cfg.market_tickers,
        channels=kalshi_cfg.channels,
    )

    client = KalshiWebSocketClient(
        config=kalshi_cfg,
        on_event=publisher.publish,
    )

    try:
        await client.run()
    except asyncio.CancelledError:
        log_event(logger, "pipeline_shutdown")
    finally:
        await publisher.close()


if __name__ == "__main__":
    asyncio.run(main())
