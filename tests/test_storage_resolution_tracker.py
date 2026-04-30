from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.storage.resolution_tracker import (
    ResolutionRecord,
    _parse_dt,
    _settlement_value_from_result,
    load_resolutions,
    save_resolutions,
    track_resolutions,
)


def test_settlement_value_yes_no_unknown():
    assert _settlement_value_from_result("yes") == 1.0
    assert _settlement_value_from_result("YES") == 1.0
    assert _settlement_value_from_result("no") == 0.0
    assert _settlement_value_from_result("0") == 0.0
    assert _settlement_value_from_result(None) is None
    assert _settlement_value_from_result("invalid") is None


def test_parse_dt_handles_iso_and_z_suffix():
    parsed = _parse_dt("2026-04-30T12:00:00Z")
    assert parsed is not None
    assert parsed.year == 2026
    assert _parse_dt("") is None
    assert _parse_dt(None) is None


def test_save_and_load_resolutions_roundtrip(tmp_path: Path):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    records = [
        ResolutionRecord(
            ticker="X-1", status="settled", result="yes", settlement_value=1.0,
            close_time=datetime(2026, 4, 1, tzinfo=timezone.utc),
            settled_time=datetime(2026, 4, 2, tzinfo=timezone.utc),
            last_checked=datetime(2026, 4, 30, tzinfo=timezone.utc),
        ),
        ResolutionRecord(
            ticker="X-2", status="open", result=None, settlement_value=None,
            close_time=None, settled_time=None,
            last_checked=datetime(2026, 4, 30, tzinfo=timezone.utc),
        ),
    ]
    save_resolutions(records, archive_root)
    loaded = load_resolutions(archive_root)
    assert set(loaded.keys()) == {"X-1", "X-2"}
    assert loaded["X-1"].settlement_value == 1.0
    assert loaded["X-2"].status == "open"


def test_load_resolutions_empty_when_missing(tmp_path: Path):
    assert load_resolutions(tmp_path) == {}


async def test_track_resolutions_skips_terminal(monkeypatch, tmp_path: Path):
    """Terminal-status records should be skipped; HTTP should not be called for them."""
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    # Pre-populate with one settled and one open
    save_resolutions([
        ResolutionRecord(
            ticker="settled-1", status="settled", result="yes", settlement_value=1.0,
            close_time=None, settled_time=None,
            last_checked=datetime.now(timezone.utc),
        ),
        ResolutionRecord(
            ticker="open-1", status="open", result=None, settlement_value=None,
            close_time=None, settled_time=None,
            last_checked=datetime.now(timezone.utc),
        ),
    ], archive_root)

    fetched: list[str] = []

    async def fake_fetch_market(session, base_url, ticker, semaphore):
        fetched.append(ticker)
        return {"status": "settled", "result": "no", "close_time": None, "settle_time": None}

    import src.storage.resolution_tracker as rt
    monkeypatch.setattr(rt, "_fetch_market", fake_fetch_market)

    # Mock aiohttp.ClientSession to avoid real network
    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None

    monkeypatch.setattr(rt.aiohttp, "ClientSession", FakeSession)

    records = await track_resolutions(
        base_url="https://api.example.com",
        tickers=["settled-1", "open-1", "new-1"],
        archive_root=archive_root,
        concurrency=2,
    )

    # settled-1 should be skipped — only open-1 and new-1 hit the network
    assert "settled-1" not in fetched
    assert set(fetched) == {"open-1", "new-1"}
    # open-1 transitioned to settled with result no → settlement_value 0.0
    by_ticker = {r.ticker: r for r in records}
    assert by_ticker["open-1"].settlement_value == 0.0
    assert by_ticker["new-1"].status == "settled"
    # settled-1 retained
    assert by_ticker["settled-1"].settlement_value == 1.0
