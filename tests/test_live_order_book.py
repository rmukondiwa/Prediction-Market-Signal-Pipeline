"""
Tests for the live order book infrastructure:
  - LiveBook (snapshot + delta application, best_bid, best_ask, mid)
  - OrderBookAggregator (routing, counts)
  - check_monotonicity from scripts/scan_live_book.py
"""
from __future__ import annotations

import importlib
from datetime import datetime, timezone

import pytest

from src.catalog.models import CatalogMarket
from src.models.orderbook_event import OrderBookEvent, OrderLevel
from src.portfolio.order_book import LiveBook, OrderBookAggregator

scan_live_book = importlib.import_module("scripts.scan_live_book")
check_monotonicity = scan_live_book.check_monotonicity
_parse_strike = scan_live_book._parse_strike

_TS = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def snapshot(ticker: str, yes: list[tuple[int, int]], no: list[tuple[int, int]]) -> OrderBookEvent:
    return OrderBookEvent(
        market_id=ticker, ticker=ticker, event_type="snapshot",
        yes_levels=[OrderLevel(price=p, quantity=q) for p, q in yes],
        no_levels=[OrderLevel(price=p, quantity=q) for p, q in no],
        source="kalshi", timestamp=_TS,
    )


def delta(ticker: str, side: str, price: int, quantity: int) -> OrderBookEvent:
    return OrderBookEvent(
        market_id=ticker, ticker=ticker, event_type="delta",
        delta_side=side, delta_price=price, delta_quantity=quantity,
        source="kalshi", timestamp=_TS,
    )


def btcd_market(strike: float, yes_bid: int, yes_ask: int) -> CatalogMarket:
    ticker = f"KXBTCD-26MAY1312-T{strike}"
    return CatalogMarket(
        ticker=ticker, event_ticker="KXBTCD-26MAY1312",
        title="BTC daily", subtitle=f"${strike} or above",
        category="Crypto", status="open",
        yes_bid=yes_bid, yes_ask=yes_ask,
        implied_probability=(yes_bid + yes_ask) / 200,
    )


def aggregator_with_books(books: dict[str, tuple[int, int]]) -> OrderBookAggregator:
    """
    Build an aggregator whose books have specific bid/ask values.
    books = { ticker: (best_bid_cents, best_ask_cents) }

    We achieve a desired best_bid by putting one YES level at that price,
    and a desired best_ask by putting one NO level at (100 - ask).
    """
    agg = OrderBookAggregator()
    for ticker, (bid, ask) in books.items():
        # YES bid at `bid` → best_bid == bid
        # NO  bid at (100 - ask) → best_ask == ask
        evt = snapshot(ticker, yes=[(bid, 10)], no=[(100 - ask, 10)])
        agg.on_event(evt)
    return agg


# ---------------------------------------------------------------------------
# LiveBook — initial state
# ---------------------------------------------------------------------------

def test_empty_book_has_no_bid():
    assert LiveBook("T").best_bid is None


def test_empty_book_has_no_ask():
    assert LiveBook("T").best_ask is None


def test_empty_book_has_no_mid():
    assert LiveBook("T").mid is None


def test_empty_book_is_not_initialised():
    assert not LiveBook("T").is_initialised()


# ---------------------------------------------------------------------------
# LiveBook — snapshot
# ---------------------------------------------------------------------------

def test_snapshot_sets_best_bid():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5), (45, 3)], no=[(50, 2)]))
    assert b.best_bid == 45


def test_snapshot_sets_best_ask_via_no_levels():
    # NO bid at 65 → YES ask = 100 - 65 = 35
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(20, 5)], no=[(65, 3)]))
    assert b.best_ask == 35


def test_snapshot_mid():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5)], no=[(40, 5)]))  # bid=40, ask=60
    assert b.mid == 50.0


def test_second_snapshot_replaces_first():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(80, 5)], no=[(10, 5)]))
    b.apply(snapshot("T", yes=[(30, 5)], no=[(60, 5)]))
    assert b.best_bid == 30
    assert b.best_ask == 40


def test_snapshot_marks_initialised():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5)], no=[]))
    assert b.is_initialised()


# ---------------------------------------------------------------------------
# LiveBook — delta
# ---------------------------------------------------------------------------

def test_delta_updates_yes_level():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5)], no=[]))
    b.apply(delta("T", "yes", 40, 20))
    assert b.yes[40] == 20


def test_delta_adds_new_yes_level():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5)], no=[]))
    b.apply(delta("T", "yes", 50, 8))
    assert b.best_bid == 50


def test_delta_removes_level_on_zero_quantity():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5), (50, 3)], no=[]))
    b.apply(delta("T", "yes", 50, 0))
    assert 50 not in b.yes
    assert b.best_bid == 40


def test_delta_removes_level_on_none_quantity():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5), (50, 3)], no=[]))
    b.apply(delta("T", "yes", 50, None))
    assert 50 not in b.yes


def test_delta_updates_no_level_changes_ask():
    b = LiveBook("T")
    # NO bid at 60 → ask = 40
    b.apply(snapshot("T", yes=[(30, 5)], no=[(60, 5)]))
    # Remove the 60 NO bid; add one at 70 → ask = 30
    b.apply(delta("T", "no", 60, 0))
    b.apply(delta("T", "no", 70, 5))
    assert b.best_ask == 30


def test_delta_increments_event_count():
    b = LiveBook("T")
    b.apply(snapshot("T", yes=[(40, 5)], no=[]))
    b.apply(delta("T", "yes", 40, 10))
    assert b.event_count == 2


# ---------------------------------------------------------------------------
# OrderBookAggregator
# ---------------------------------------------------------------------------

def test_aggregator_unknown_ticker_returns_none():
    agg = OrderBookAggregator()
    assert agg.get_book("UNKNOWN") is None


def test_aggregator_creates_book_on_first_event():
    agg = OrderBookAggregator()
    agg.on_event(snapshot("KXBTCD-T1", yes=[(50, 5)], no=[]))
    assert agg.get_book("KXBTCD-T1") is not None


def test_aggregator_routes_to_correct_book():
    agg = OrderBookAggregator()
    agg.on_event(snapshot("A", yes=[(40, 5)], no=[]))
    agg.on_event(snapshot("B", yes=[(70, 5)], no=[]))
    assert agg.get_book("A").best_bid == 40
    assert agg.get_book("B").best_bid == 70


def test_aggregator_book_count():
    agg = OrderBookAggregator()
    agg.on_event(snapshot("A", yes=[(40, 5)], no=[]))
    agg.on_event(snapshot("B", yes=[(70, 5)], no=[]))
    assert agg.book_count() == 2


def test_aggregator_initialised_count():
    agg = OrderBookAggregator()
    agg.on_event(snapshot("A", yes=[(40, 5)], no=[]))
    agg.on_event(snapshot("B", yes=[(70, 5)], no=[]))
    assert agg.initialised_count() == 2


def test_aggregator_delta_applied_to_existing_book():
    agg = OrderBookAggregator()
    agg.on_event(snapshot("A", yes=[(40, 5)], no=[]))
    agg.on_event(delta("A", "yes", 50, 3))
    assert agg.get_book("A").best_bid == 50


# ---------------------------------------------------------------------------
# _parse_strike
# ---------------------------------------------------------------------------

def test_parse_strike_standard():
    assert _parse_strike("KXBTCD-26MAY1312-T89799.99") == 89799.99


def test_parse_strike_integer():
    assert _parse_strike("KXBTCD-26MAY1312-T90000") == 90000.0


def test_parse_strike_no_match_returns_none():
    assert _parse_strike("KXBTCD-26MAY1312") is None
    assert _parse_strike("KXIRANUS-26JUN30") is None


# ---------------------------------------------------------------------------
# check_monotonicity
# ---------------------------------------------------------------------------

def test_no_violation_when_prices_are_valid():
    # bid_low=40 < ask_high=45 → no arb, correct market ordering
    low = btcd_market(89000, 40, 55)
    high = btcd_market(89100, 20, 45)
    agg = aggregator_with_books({low.ticker: (40, 55), high.ticker: (20, 45)})
    result = check_monotonicity("KXBTCD-26MAY1312", [low, high], agg)
    assert result == []


def test_violation_detected_when_bid_exceeds_higher_ask():
    # lower strike bid (70) > higher strike ask (60) — impossible, arb exists
    low = btcd_market(89000, 65, 70)
    high = btcd_market(89100, 55, 65)
    agg = aggregator_with_books({low.ticker: (70, 75), high.ticker: (55, 60)})
    result = check_monotonicity("KXBTCD-26MAY1312", [low, high], agg)
    assert len(result) == 1
    assert result[0]["low_ticker"] == low.ticker
    assert result[0]["high_ticker"] == high.ticker


def test_violation_spread_is_correct():
    low = btcd_market(89000, 65, 70)
    high = btcd_market(89100, 55, 65)
    agg = aggregator_with_books({low.ticker: (70, 75), high.ticker: (55, 60)})
    result = check_monotonicity("KXBTCD-26MAY1312", [low, high], agg)
    assert result[0]["spread_cents"] == 10   # bid=70, ask=60 → 70-60


def test_no_violation_when_bid_equals_ask():
    # bid == ask is NOT a violation (strict > required)
    low = btcd_market(89000, 60, 65)
    high = btcd_market(89100, 40, 45)
    agg = aggregator_with_books({low.ticker: (60, 75), high.ticker: (40, 60)})
    result = check_monotonicity("KXBTCD-26MAY1312", [low, high], agg)
    assert result == []


def test_markets_sorted_by_strike_regardless_of_input_order():
    # Pass markets in reverse order — check should still compare correctly
    low = btcd_market(89000, 65, 70)
    high = btcd_market(89100, 55, 65)
    agg = aggregator_with_books({low.ticker: (70, 75), high.ticker: (55, 60)})
    result = check_monotonicity("KXBTCD-26MAY1312", [high, low], agg)  # reversed input
    assert len(result) == 1
    assert result[0]["low_strike"] == 89000.0
    assert result[0]["high_strike"] == 89100.0


def test_missing_book_is_skipped_without_error():
    markets = [btcd_market(89000, 60, 65), btcd_market(89100, 40, 45)]
    agg = OrderBookAggregator()   # no books at all
    result = check_monotonicity("KXBTCD-26MAY1312", markets, agg)
    assert result == []


def test_multiple_violations_all_returned():
    m1 = btcd_market(89000, 65, 70)
    m2 = btcd_market(89100, 55, 65)
    m3 = btcd_market(89200, 45, 55)
    # bid[m1]=70 > ask[m2]=60 → violation; bid[m2]=65 > ask[m3]=50 → violation
    agg = aggregator_with_books({
        m1.ticker: (70, 75),
        m2.ticker: (65, 60),   # ask=60 < bid[m1]=70 → first violation
        m3.ticker: (45, 50),   # ask=50 < bid[m2]=65 → second violation
    })
    result = check_monotonicity("KXBTCD-26MAY1312", [m1, m2, m3], agg)
    assert len(result) == 2


def test_single_market_group_has_no_violations():
    # Need at least 2 markets to form a pair
    markets = [btcd_market(89000, 60, 65)]
    agg = aggregator_with_books({"KXBTCD-26MAY1312-T89000.0": (60, 70)})
    result = check_monotonicity("KXBTCD-26MAY1312", markets, agg)
    assert result == []
