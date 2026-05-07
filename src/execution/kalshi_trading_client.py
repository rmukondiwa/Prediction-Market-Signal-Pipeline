"""
Kalshi trading client — stub + live implementations.

Both implementations satisfy the `KalshiTradingClient` Protocol. The stub is
used by paper trading; the live client makes authenticated REST calls to
Kalshi's order endpoints using the same RSA-PSS pattern as the WebSocket
auth (see src/ingestion/kalshi/websocket_client.py).

Live client safety: the constructor logs a clear startup banner; callers
should still gate `--live` behind explicit user confirmation. Endpoints are
configured via KalshiConfig (KALSHI_REST_URL env), defaulting to production.
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from typing import Protocol

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.kalshi_config import KalshiConfig
from src.execution.models import OrderRequest, OrderResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class KalshiTradingClient(Protocol):
    """Interface the executor depends on. Both stub and real impl satisfy it."""

    async def place_order(self, request: OrderRequest) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_balance(self) -> float: ...


class KalshiTradingClientStub:
    """Records placed orders in memory; never sends a real request.

    Useful for paper trading and as a drop-in for testing the executor wiring
    without configuring API auth. Returns a synthetic order_id immediately.
    """

    def __init__(self):
        self.placed: list[OrderRequest] = []
        self.canceled: list[str] = []
        self._balance = 10_000.0

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self.placed.append(request)
        order_id = f"stub-{uuid.uuid4().hex[:8]}"
        logger.info("Stub place_order", extra={
            "ticker": request.ticker, "side": request.side,
            "contracts": request.contracts, "order_id": order_id,
        })
        return OrderResult(
            accepted=True, order_id=order_id, error=None,
            raw_response={"stub": True, "client_order_id": request.client_order_id},
        )

    async def cancel_order(self, order_id: str) -> bool:
        self.canceled.append(order_id)
        logger.info("Stub cancel_order", extra={"order_id": order_id})
        return True

    async def get_balance(self) -> float:
        return self._balance


class KalshiTradingClientLive:
    """Real Kalshi REST trading client.

    Auth: same RSA-PSS scheme as the WebSocket client. Each request signs
    `<timestamp_ms><METHOD><path>` and sends KALSHI-ACCESS-{KEY,TIMESTAMP,SIGNATURE}
    headers.

    Conversions: Kalshi prices are integer cents 1-99 inclusive; OrderRequest
    uses float dollars. We round to nearest cent for the wire and clamp into
    [1, 99]. Contracts are integer counts (Kalshi calls this `count`).

    Safety: this class talks to the live exchange and risks real money. Gate
    instantiation behind an explicit `--live` flag and a typed confirmation
    prompt at the entry point.
    """

    _ORDER_PATH = "/portfolio/orders"
    _BALANCE_PATH = "/portfolio/balance"

    def __init__(self, config: KalshiConfig, *, request_timeout_s: float = 5.0):
        self._config = config
        self._private_key = serialization.load_pem_private_key(
            config.private_key_pem.encode("utf-8"), password=None
        )
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s)
        self._session: aiohttp.ClientSession | None = None
        logger.warning(
            "KalshiTradingClientLive instantiated — live order endpoint enabled",
            extra={"rest_base_url": config.rest_base_url, "api_key_id": config.api_key_id[:8] + "..."},
        )

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()
            self._session = None

    def _sign(self, message: str) -> str:
        sig = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self._config.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts + method + path),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    @staticmethod
    def _to_cents(price_dollars: float) -> int:
        cents = round(price_dollars * 100)
        return max(1, min(99, cents))

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a buy order (the only direction the decay strategy uses).

        Decay always BUYS yes or BUYS no — never sells. Kalshi's API expects
        action='buy' with the price field matching the side (yes_price for
        buy-yes, no_price for buy-no), in cents.
        """
        if request.limit_price is None:
            return OrderResult(accepted=False, order_id=None,
                               error="limit_price required for live orders",
                               raw_response={})
        cents = self._to_cents(request.limit_price)
        body = {
            "ticker": request.ticker,
            "client_order_id": request.client_order_id,
            "side": request.side,
            "action": "buy",
            "type": "limit" if request.order_type == "limit" else "market",
            "count": int(request.contracts),
        }
        if request.side == "yes":
            body["yes_price"] = cents
        else:
            body["no_price"] = cents
        if request.time_in_force == "ioc":
            body["expiration_ts"] = int(time.time())  # Kalshi treats expiration_ts<=now as IOC

        url = self._config.rest_base_url + self._ORDER_PATH
        headers = self._auth_headers("POST", self._ORDER_PATH)
        session = await self._ensure_session()
        try:
            async with session.post(url, headers=headers, json=body) as r:
                payload = await r.json()
                if r.status >= 400:
                    return OrderResult(
                        accepted=False, order_id=None,
                        error=f"HTTP {r.status}: {payload.get('error', payload)}",
                        raw_response=payload,
                    )
                order = payload.get("order", {})
                return OrderResult(
                    accepted=True,
                    order_id=order.get("order_id") or order.get("id"),
                    error=None,
                    raw_response=payload,
                )
        except asyncio.TimeoutError:
            return OrderResult(accepted=False, order_id=None,
                               error="request timeout", raw_response={})
        except aiohttp.ClientError as e:
            return OrderResult(accepted=False, order_id=None,
                               error=f"client error: {e}", raw_response={})

    async def cancel_order(self, order_id: str) -> bool:
        path = f"{self._ORDER_PATH}/{order_id}"
        url = self._config.rest_base_url + path
        headers = self._auth_headers("DELETE", path)
        session = await self._ensure_session()
        try:
            async with session.delete(url, headers=headers) as r:
                return r.status < 400
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return False

    async def get_balance(self) -> float:
        url = self._config.rest_base_url + self._BALANCE_PATH
        headers = self._auth_headers("GET", self._BALANCE_PATH)
        session = await self._ensure_session()
        async with session.get(url, headers=headers) as r:
            r.raise_for_status()
            payload = await r.json()
            # Kalshi returns balance in cents
            return payload.get("balance", 0) / 100.0
