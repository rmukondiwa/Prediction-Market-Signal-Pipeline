"""
Tests for src/catalog/fetcher.py — specifically fetch_settled_markets().

All HTTP is faked via monkeypatching aiohttp.ClientSession so no real network
calls are made.
"""
from __future__ import annotations

import pytest

from src.catalog.fetcher import fetch_settled_markets


def _make_fake_session(pages: list[dict]):
    """
    Build a fake aiohttp.ClientSession that returns pages in order.
    Each page is {"markets": [...], "cursor": "..."}.
    """
    call_count = [0]

    class FakeResponse:
        def __init__(self, data):
            self._data = data
            self.status = 200

        async def json(self):
            return self._data

        def raise_for_status(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class FakeSession:
        def get(self, url, params=None):
            page = pages[call_count[0] % len(pages)]
            call_count[0] += 1
            return FakeResponse(page)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    return FakeSession, call_count


async def test_fetch_settled_stops_when_cursor_empty(monkeypatch):
    """Two pages returned; second page has empty cursor → stops."""
    pages = [
        {"markets": [{"ticker": f"T{i}"} for i in range(3)], "cursor": "next"},
        {"markets": [{"ticker": f"T{i}"} for i in range(3, 5)], "cursor": ""},
    ]
    FakeSession, _ = _make_fake_session(pages)
    import src.catalog.fetcher as fetcher_mod
    monkeypatch.setattr(fetcher_mod.aiohttp, "ClientSession", FakeSession)

    result = await fetch_settled_markets("https://api.example.com", max_markets=100)

    assert len(result) == 5
    assert result[0]["ticker"] == "T0"
    assert result[4]["ticker"] == "T4"


async def test_fetch_settled_stops_at_max_markets(monkeypatch):
    """max_markets=3 across two pages of 3 → stops after first page."""
    pages = [
        {"markets": [{"ticker": f"T{i}"} for i in range(3)], "cursor": "next"},
        {"markets": [{"ticker": f"T{i}"} for i in range(3, 6)], "cursor": ""},
    ]
    FakeSession, call_count = _make_fake_session(pages)
    import src.catalog.fetcher as fetcher_mod
    monkeypatch.setattr(fetcher_mod.aiohttp, "ClientSession", FakeSession)

    result = await fetch_settled_markets("https://api.example.com", max_markets=3)

    # Should have stopped after page 1 hit the cap
    assert len(result) == 3
    assert call_count[0] == 1


async def test_fetch_settled_trims_to_max_markets(monkeypatch):
    """Page returns more than max_markets total → result is trimmed exactly."""
    pages = [
        {"markets": [{"ticker": f"T{i}"} for i in range(5)], "cursor": "next"},
        {"markets": [{"ticker": f"T{i}"} for i in range(5, 9)], "cursor": ""},
    ]
    FakeSession, _ = _make_fake_session(pages)
    import src.catalog.fetcher as fetcher_mod
    monkeypatch.setattr(fetcher_mod.aiohttp, "ClientSession", FakeSession)

    result = await fetch_settled_markets("https://api.example.com", max_markets=7)

    assert len(result) == 7
    assert result[-1]["ticker"] == "T6"


async def test_fetch_settled_empty_market_list(monkeypatch):
    """API returns empty markets on first page → returns empty list."""
    pages = [{"markets": [], "cursor": ""}]
    FakeSession, _ = _make_fake_session(pages)
    import src.catalog.fetcher as fetcher_mod
    monkeypatch.setattr(fetcher_mod.aiohttp, "ClientSession", FakeSession)

    result = await fetch_settled_markets("https://api.example.com", max_markets=100)

    assert result == []
