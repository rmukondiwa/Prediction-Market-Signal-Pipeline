import asyncio
import base64
import json
import ssl
import time
from typing import Callable, Awaitable

import certifi
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.kalshi_config import KalshiConfig
from src.ingestion.kalshi.message_parser import MessageParser, ParsedMessage
from src.ingestion.kalshi.normalizer import Normalizer, NormalizedEvent
from src.utils.logging import get_logger, log_event
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)

EventCallback = Callable[[NormalizedEvent], Awaitable[None]]


class KalshiWebSocketClient:
    """
    Maintains a persistent, authenticated WebSocket connection to the Kalshi
    trading API and streams normalized market events to registered callbacks.

    Responsibilities:
    - Connect and authenticate via RSA-PSS signed headers.
    - Subscribe to configured channels and market tickers.
    - Receive and dispatch raw messages through the parser → normalizer chain.
    - Reconnect automatically with exponential backoff on any disconnection.
    - Send periodic pings to detect silent connection drops.

    This class does NOT perform storage, signal generation, or any analytics.
    """

    _WS_PATH = "/trade-api/ws/v2"

    def __init__(
        self,
        config: KalshiConfig,
        on_event: EventCallback,
    ) -> None:
        self._config = config
        self._on_event = on_event
        self._parser = MessageParser()
        self._normalizer = Normalizer()
        self._private_key = self._load_private_key(config.private_key_pem)
        self._subscribe_id = 1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the client loop.  Runs until cancelled.
        Reconnects automatically on any error using exponential backoff.
        """
        log_event(logger, "client_starting", ws_url=self._config.ws_url)
        await retry_with_backoff(
            self._connect_and_stream,
            max_attempts=self._config.reconnect_attempts,
            base_delay=self._config.reconnect_base_delay,
            max_delay=self._config.reconnect_max_delay,
            label="kalshi_websocket",
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        headers = self._build_auth_headers()
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        log_event(logger, "connection_opening", url=self._config.ws_url)

        async with websockets.connect(
            self._config.ws_url,
            additional_headers=headers,
            ssl=ssl_ctx,
            ping_interval=self._config.ping_interval,
            ping_timeout=self._config.ping_interval * 2,
        ) as ws:
            log_event(logger, "connection_opened")
            await self._subscribe(ws)
            await self._receive_loop(ws)

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        if not self._config.market_tickers:
            logger.warning("No market_tickers configured — nothing to subscribe to")
            return

        payload = {
            "id": self._subscribe_id,
            "cmd": "subscribe",
            "params": {
                "channels": self._config.channels,
                "market_tickers": self._config.market_tickers,
            },
        }
        self._subscribe_id += 1
        await ws.send(json.dumps(payload))
        log_event(
            logger,
            "subscription_sent",
            channels=self._config.channels,
            tickers=self._config.market_tickers,
        )

    async def _receive_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue  # ignore binary frames

            parsed: ParsedMessage | None = self._parser.parse(raw)
            if parsed is None:
                continue

            event: NormalizedEvent | None = self._normalizer.normalize(parsed)
            if event is None:
                continue

            try:
                await self._on_event(event)
            except Exception as exc:
                logger.error(
                    "Event callback raised an error",
                    extra={"error": str(exc), "event_type": type(event).__name__},
                )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _build_auth_headers(self) -> dict[str, str]:
        timestamp_ms = str(int(time.time() * 1000))
        msg_to_sign = timestamp_ms + "GET" + self._WS_PATH
        signature = self._sign(msg_to_sign)
        return {
            "KALSHI-ACCESS-KEY": self._config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    def _sign(self, message: str) -> str:
        """Sign a message with the RSA private key using PSS padding."""
        sig_bytes = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig_bytes).decode("utf-8")

    @staticmethod
    def _load_private_key(pem_text: str):
        return serialization.load_pem_private_key(pem_text.encode("utf-8"), password=None)
