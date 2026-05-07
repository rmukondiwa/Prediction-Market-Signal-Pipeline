"""
Polymarket trading client — stub + live implementations.

Polymarket trades on a hybrid: an off-chain Central Limit Order Book (CLOB)
with on-chain settlement via the Conditional Tokens Framework on Polygon.
To place an order:
  1. Build a typed order struct (price, size, side, token_id, etc.)
  2. Sign it via EIP-712 with the user's Polygon private key
  3. POST the signed order to clob.polymarket.com/order

We use the official `py-clob-client` library, which handles signing and
the HTTP layer. Reads (book, prices) are unauthenticated; writes require
a Polygon address with funds + USDC approvals already set up.

Safety:
  - The live client logs a loud banner on construction
  - Wrappers should gate `--live` behind typed user confirmation, mirroring
    the Kalshi pattern in scripts/paper_trade_decay.py
  - Required setup is OUTSIDE this code: Polygon wallet, USDC deposit,
    CLOB allowance approval. We do NOT auto-approve allowances.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from src.execution.models import OrderRequest, OrderResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137


class PolymarketTradingClient(Protocol):
    """Common interface across stub + live implementations."""

    async def place_order(self, request: PolymarketOrderRequest) -> OrderResult: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_balance(self) -> float: ...


@dataclass
class PolymarketOrderRequest:
    """Polymarket-flavoured order. Differs from Kalshi:
       - token_id (ERC-1155) replaces ticker
       - side is BUY/SELL on a specific outcome token
       - price in dollars 0-1
       - size in number of contracts
    """
    token_id: str
    side: str         # "BUY" or "SELL"
    price: float      # 0-1 (USDC per contract)
    size: float       # contracts
    order_type: str = "GTC"  # GTC | FOK | GTD
    client_order_id: str | None = None


class PolymarketTradingClientStub:
    """Records intents in memory; no network. Symmetric to Kalshi stub."""

    def __init__(self):
        self.placed: list[PolymarketOrderRequest] = []
        self.canceled: list[str] = []

    async def place_order(self, request: PolymarketOrderRequest) -> OrderResult:
        self.placed.append(request)
        order_id = f"poly-stub-{uuid.uuid4().hex[:8]}"
        logger.info("Polymarket stub place_order",
                    extra={"token_id": request.token_id[:20], "side": request.side,
                           "price": request.price, "size": request.size,
                           "order_id": order_id})
        return OrderResult(accepted=True, order_id=order_id, error=None,
                           raw_response={"stub": True, "client_order_id": request.client_order_id})

    async def cancel_order(self, order_id: str) -> bool:
        self.canceled.append(order_id)
        return True

    async def get_balance(self) -> float:
        return 0.0  # stub doesn't track real balance


class PolymarketTradingClientLive:
    """Real Polymarket CLOB client via py-clob-client.

    Requires (in env or constructor args):
      POLYMARKET_PRIVATE_KEY   — Polygon EOA private key with USDC + CTF allowances
      POLYMARKET_FUNDER        — optional funder address if using delegated funding
      POLYMARKET_API_KEY,
      POLYMARKET_API_SECRET,
      POLYMARKET_API_PASSPHRASE — L2 (API-key) credentials, generated via
        client.create_or_derive_api_creds(). Optional but recommended for
        less rate-limited reads.

    Setup procedure (one-time, outside this code):
      1. Create a Polygon wallet (MetaMask, etc.)
      2. Deposit USDC (Polygon, native — not bridged USDC.e)
      3. Approve USDC for the CTFExchange contract:
         https://docs.polymarket.com/developers/CLOB/allowances
      4. Approve the CTF (NEG_RISK_CTF too if using neg-risk markets)

    The client logs a startup banner and is gated behind typed user
    confirmation in any wrapper script.
    """

    def __init__(self,
                 private_key: str | None = None,
                 host: str = CLOB_HOST,
                 chain_id: int = POLYGON_CHAIN_ID,
                 api_creds: tuple[str, str, str] | None = None):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError as e:
            raise RuntimeError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            ) from e

        pk = private_key or os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not pk:
            raise RuntimeError(
                "POLYMARKET_PRIVATE_KEY not set in env. Required for live trading."
            )

        creds = None
        if api_creds is None:
            api_key = os.environ.get("POLYMARKET_API_KEY")
            api_secret = os.environ.get("POLYMARKET_API_SECRET")
            api_passphrase = os.environ.get("POLYMARKET_API_PASSPHRASE")
            if all((api_key, api_secret, api_passphrase)):
                creds = ApiCreds(api_key=api_key, api_secret=api_secret,
                                 api_passphrase=api_passphrase)

        self._client = ClobClient(host=host, key=pk, chain_id=chain_id, creds=creds)
        if creds is None:
            # L1 only — derive L2 creds for read endpoints (write still uses L1 sig)
            try:
                self._client.set_api_creds(self._client.create_or_derive_api_creds())
                logger.info("Polymarket L2 API creds derived from L1 key")
            except Exception as e:
                logger.warning("Failed to derive L2 creds — reads may rate-limit",
                               extra={"error": str(e)})

        logger.warning(
            "PolymarketTradingClientLive instantiated — live order endpoint enabled",
            extra={"host": host, "chain_id": chain_id,
                   "address_prefix": self._client.get_address()[:10] if hasattr(self._client, "get_address") else "?"},
        )

    async def place_order(self, request: PolymarketOrderRequest) -> OrderResult:
        """Place a CLOB order. Maps our OrderRequest → py-clob-client OrderArgs."""
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
        except ImportError:
            return OrderResult(accepted=False, order_id=None,
                               error="py-clob-client missing", raw_response={})

        try:
            order_type_map = {"GTC": OrderType.GTC, "FOK": OrderType.FOK, "GTD": OrderType.GTD}
            order_type = order_type_map.get(request.order_type, OrderType.GTC)
            # py-clob-client is sync; run in thread to avoid blocking the loop
            import asyncio
            args = OrderArgs(
                token_id=request.token_id,
                price=float(request.price),
                size=float(request.size),
                side=request.side,
            )
            signed = await asyncio.to_thread(
                self._client.create_order, args
            )
            resp = await asyncio.to_thread(
                self._client.post_order, signed, order_type
            )
            order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
            if not resp.get("success", True) and resp.get("errorMsg"):
                return OrderResult(accepted=False, order_id=None,
                                   error=resp["errorMsg"], raw_response=resp)
            return OrderResult(accepted=True, order_id=order_id, error=None,
                               raw_response=resp)
        except Exception as e:
            return OrderResult(accepted=False, order_id=None,
                               error=f"{type(e).__name__}: {e}", raw_response={})

    async def cancel_order(self, order_id: str) -> bool:
        import asyncio
        try:
            resp = await asyncio.to_thread(self._client.cancel, order_id)
            return bool(resp.get("success", True))
        except Exception as e:
            logger.warning("Polymarket cancel failed",
                           extra={"order_id": order_id, "error": str(e)})
            return False

    async def get_balance(self) -> float:
        """USDC balance available for trading on Polymarket."""
        import asyncio
        try:
            # py-clob-client exposes get_balance_allowance() for combined info
            resp = await asyncio.to_thread(self._client.get_balance_allowance)
            # response format: {"balance": "...", "allowance": "..."} as USDC strings
            return float(resp.get("balance", 0)) / 1e6  # USDC has 6 decimals
        except Exception as e:
            logger.warning("Polymarket balance fetch failed", extra={"error": str(e)})
            return 0.0


def from_kalshi_order(order_request: OrderRequest, token_id: str) -> PolymarketOrderRequest:
    """Convert a generic OrderRequest into a Polymarket-flavored one.

    Useful for symmetric paper-vs-live wiring: you have an OrderRequest from
    the strategy layer, you call this to translate it before passing to
    a Polymarket client. Side mapping:
      OrderRequest.side="yes" → Polymarket BUY at the YES token
      OrderRequest.side="no"  → Polymarket BUY at the NO token
    The caller picks which token_id to pass based on side.
    """
    # On Polymarket, "buying YES" and "buying NO" are buys of different tokens,
    # always at the offered ASK price. We don't sell short here (that's a
    # different action; usually traders just buy the opposite token).
    return PolymarketOrderRequest(
        token_id=token_id,
        side="BUY",
        price=order_request.limit_price or 0.5,
        size=float(order_request.contracts),
        order_type="FOK" if order_request.time_in_force == "ioc" else "GTC",
        client_order_id=order_request.client_order_id,
    )
