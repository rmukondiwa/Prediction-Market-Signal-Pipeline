"""
Polymarket CLOB (Central Limit Order Book) read-only client.

Fetches live order books and prices from Polymarket's CLOB API. Read-only:
no orders placed here, no auth required for read endpoints. The Gamma API
gives us market metadata (question, outcomes, tokenIds, AMM mid-price);
this CLOB client gives us real bid/ask/depth for those tokens.

Polymarket markets have 2 outcomes (YES, NO), each with its own ERC-1155
token id (`clobTokenIds[0]` = YES, `[1]` = NO). Each token has its own
CLOB order book — we fetch both to compute arb-relevant top-of-book.

Endpoints used:
  GET https://clob.polymarket.com/book?token_id=<id>     order book snapshot
  GET https://clob.polymarket.com/price?token_id=<id>&side=<sell|buy>  midpoint

Both are public, no auth.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

CLOB_BASE = "https://clob.polymarket.com"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class TokenBook:
    """Top-of-book + depth for one Polymarket outcome token."""
    token_id: str
    bids: list[OrderLevel]  # sorted descending by price
    asks: list[OrderLevel]  # sorted ascending by price

    @property
    def top_bid(self) -> OrderLevel | None:
        return self.bids[0] if self.bids else None

    @property
    def top_ask(self) -> OrderLevel | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.top_bid and self.top_ask:
            return (self.top_bid.price + self.top_ask.price) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.top_bid and self.top_ask:
            return self.top_ask.price - self.top_bid.price
        return None


def _http_json(url: str, timeout: float = 15.0) -> dict | list:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_book(token_id: str) -> TokenBook:
    """Fetch the live order book for a single Polymarket outcome token.

    Returns bids sorted descending by price, asks ascending. Empty lists
    when the book is empty.
    """
    data = _http_json(f"{CLOB_BASE}/book?token_id={token_id}")
    if isinstance(data, dict) and "error" in data:
        return TokenBook(token_id=token_id, bids=[], asks=[])
    bids = sorted(
        [OrderLevel(price=float(b["price"]), size=float(b["size"]))
         for b in data.get("bids", [])],
        key=lambda x: -x.price,
    )
    asks = sorted(
        [OrderLevel(price=float(a["price"]), size=float(a["size"]))
         for a in data.get("asks", [])],
        key=lambda x: x.price,
    )
    return TokenBook(token_id=token_id, bids=bids, asks=asks)


def get_market_books(market: dict) -> tuple[TokenBook, TokenBook] | None:
    """Given a Polymarket Gamma `market` dict, return (yes_book, no_book).

    Returns None if `clobTokenIds` is missing or malformed. Both books may
    have empty bid/ask sides if no orders are resting.
    """
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    try:
        return get_book(raw[0]), get_book(raw[1])
    except Exception:
        return None


def cost_to_buy_amount(book: TokenBook, contracts: float) -> float | None:
    """Walk the ask side to compute the average price to buy `contracts`
    units. Returns None if depth insufficient."""
    if not book.asks:
        return None
    remaining = contracts
    cost = 0.0
    for level in book.asks:
        take = min(remaining, level.size)
        cost += take * level.price
        remaining -= take
        if remaining <= 0:
            return cost / contracts
    return None  # depth insufficient


def best_arb_round_trip(yes_book: TokenBook, no_book: TokenBook) -> dict:
    """Compute the cheapest round-trip on Polymarket for $1 guaranteed payout.

    Buying both YES and NO at their respective asks gives a guaranteed $1
    payout (one of the two settles to $1, the other to $0). If
    yes_ask + no_ask < $1, that's a within-Polymarket arb. Mostly absent
    in efficient markets but useful as a sanity quote for cross-platform.
    """
    out = {
        "yes_top_ask": yes_book.top_ask.price if yes_book.top_ask else None,
        "yes_top_ask_size": yes_book.top_ask.size if yes_book.top_ask else None,
        "no_top_ask": no_book.top_ask.price if no_book.top_ask else None,
        "no_top_ask_size": no_book.top_ask.size if no_book.top_ask else None,
        "yes_top_bid": yes_book.top_bid.price if yes_book.top_bid else None,
        "no_top_bid": no_book.top_bid.price if no_book.top_bid else None,
    }
    ya = out["yes_top_ask"]
    na = out["no_top_ask"]
    if ya is not None and na is not None:
        out["yes_plus_no_ask"] = round(ya + na, 4)
        out["within_poly_arb_edge"] = round(1.0 - (ya + na), 4)
    return out
