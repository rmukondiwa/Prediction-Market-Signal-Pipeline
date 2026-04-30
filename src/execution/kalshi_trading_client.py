"""
Kalshi trading client — STUB implementation.

The real client extends RSA-PSS auth from src/ingestion/kalshi/websocket_client.py
to authenticated REST trading endpoints (balance, positions, orders, place,
cancel). The exact endpoint paths must be verified against current Kalshi API
docs at build time; do not rely on hand-written paths in this file.

Stage 9 ships the *interface*; the network calls are stubbed. PaperExecutor
uses this interface to place simulated orders without ever leaving local
state, so we surface integration shape now without committing real capital.

When upgrading to live trading, swap `KalshiTradingClientStub` for a real
implementation that wraps `kalshi-python` or hand-rolls auth headers.
"""
from __future__ import annotations

import uuid
from typing import Protocol

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
