"""
Kalshi wallet snapshot.

Fetches account balance and all open positions in batch via the Kalshi REST API.

Usage:
    python3 -m scripts.wallet
    python3 -m scripts.wallet --json        # raw JSON output
"""

import argparse
import asyncio
import base64
import json
import time
from urllib.parse import urlparse
from typing import Any

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

from src.config.kalshi_config import KalshiConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _sign(private_key, timestamp_ms: str, method: str, path: str) -> str:
    msg = timestamp_ms + method.upper() + path
    sig_bytes = private_key.sign(
        msg.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig_bytes).decode("utf-8")


def _auth_headers(private_key, api_key_id: str, method: str, path: str) -> dict[str, str]:
    timestamp_ms = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": _sign(private_key, timestamp_ms, method, path),
        "Content-Type": "application/json",
    }


def _path(base_url: str, endpoint: str) -> str:
    """Extract just the path portion for signing (no host, no query string)."""
    parsed = urlparse(base_url)
    return parsed.path.rstrip("/") + endpoint


async def fetch_balance(session: aiohttp.ClientSession, base_url: str, private_key, api_key_id: str) -> int:
    endpoint = "/portfolio/balance"
    headers = _auth_headers(private_key, api_key_id, "GET", _path(base_url, endpoint))
    async with session.get(f"{base_url}{endpoint}", headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data.get("balance", 0)


async def fetch_all_positions(session: aiohttp.ClientSession, base_url: str, private_key, api_key_id: str) -> tuple[list[dict], float]:
    endpoint = "/portfolio/positions"
    positions: list[dict] = []
    cursor: str | None = None
    t_start = time.perf_counter()

    while True:
        params: dict = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        # Re-sign each request (timestamp must be fresh)
        headers = _auth_headers(private_key, api_key_id, "GET", _path(base_url, endpoint))

        async with session.get(f"{base_url}{endpoint}", headers=headers, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        page = data.get("market_positions", [])
        positions.extend(page)

        cursor = data.get("cursor", "")
        if not cursor:
            break

    elapsed = time.perf_counter() - t_start
    return positions, elapsed


def _print_table(balance_cents: int, positions: list[dict], positions_elapsed: float) -> None:
    balance_dollars = balance_cents / 100

    print(f"\n{'='*60}")
    print(f"  Account balance:   ${balance_dollars:,.2f}")
    print(f"  Open positions:    {len(positions)}")
    print(f"  Positions fetched: {positions_elapsed:.3f}s")
    print(f"{'='*60}\n")

    if not positions:
        print("  No open positions.")
        return

    # Sort by absolute position size descending
    positions = sorted(positions, key=lambda p: abs(p.get("position", 0)), reverse=True)

    header = f"{'TICKER':<35} {'POS':>6} {'SIDE':<4} {'REALIZED PNL':>13}"
    print(header)
    print("-" * len(header))

    for p in positions:
        ticker = p.get("market_ticker") or p.get("ticker", "unknown")
        position = p.get("position", 0)
        realized_pnl = p.get("realized_pnl", 0) / 100  # cents → dollars
        side = "YES" if position > 0 else "NO"

        print(f"{ticker:<35} {abs(position):>6} {side:<4} ${realized_pnl:>12,.2f}")

    total_pnl = sum(p.get("realized_pnl", 0) for p in positions) / 100
    print("-" * len(header))
    print(f"{'TOTAL REALIZED PNL':<46} ${total_pnl:>12,.2f}\n")


async def main(as_json: bool = False) -> None:
    load_dotenv()
    config = KalshiConfig()

    private_key = serialization.load_pem_private_key(
        config.private_key_pem.encode("utf-8"), password=None
    )

    logger.info("Fetching wallet data from Kalshi")

    async with aiohttp.ClientSession() as session:
        balance, (positions, positions_elapsed) = await asyncio.gather(
            fetch_balance(session, config.rest_base_url, private_key, config.api_key_id),
            fetch_all_positions(session, config.rest_base_url, private_key, config.api_key_id),
        )

    logger.info(
        "Wallet fetch complete",
        extra={
            "positions": len(positions),
            "balance_cents": balance,
            "positions_elapsed_s": round(positions_elapsed, 3),
        },
    )

    if as_json:
        print(json.dumps({"balance_cents": balance, "positions": positions, "positions_elapsed_s": round(positions_elapsed, 3)}, indent=2))
    else:
        _print_table(balance, positions, positions_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi wallet snapshot")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    args = parser.parse_args()
    asyncio.run(main(as_json=args.json))
