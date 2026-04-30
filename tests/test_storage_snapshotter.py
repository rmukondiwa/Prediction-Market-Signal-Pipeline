"""Tests for src.storage.snapshotter — uses fakeredis for stream simulation."""
from __future__ import annotations

import json
from pathlib import Path

import fakeredis.aioredis
import pytest
import pyarrow.parquet as pq

from src.storage.snapshotter import (
    _next_id,
    _load_checkpoint,
    _save_checkpoint,
    drain_streams,
    run_snapshot,
)


def test_next_id_increments_seq():
    assert _next_id("1700000000000-0") == "1700000000000-1"
    assert _next_id("1700000000000-42") == "1700000000000-43"


def test_next_id_handles_zero():
    assert _next_id("0-0") == "0-1"


def test_checkpoint_roundtrip(tmp_path: Path):
    ckpt = {"market_events": "100-0", "trade_events": "200-3"}
    _save_checkpoint(tmp_path, ckpt)
    loaded = _load_checkpoint(tmp_path)
    assert loaded == ckpt


def test_checkpoint_missing_returns_empty(tmp_path: Path):
    assert _load_checkpoint(tmp_path) == {}


def test_checkpoint_corrupt_returns_empty(tmp_path: Path):
    (tmp_path / ".checkpoint.json").write_text("not valid json {")
    assert _load_checkpoint(tmp_path) == {}


@pytest.fixture
async def fake_redis_with_data():
    """Spin up a fake Redis with three streams pre-populated."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    for i in range(5):
        await r.xadd("market_events", {"ticker": "AAA", "yes_bid": str(20 + i)})
    for i in range(3):
        await r.xadd("trade_events", {"ticker": "AAA", "yes_price": str(50 + i)})
    yield r
    await r.aclose()


async def test_drain_streams_writes_parquet_and_advances_checkpoint(
    fake_redis_with_data, tmp_path: Path, monkeypatch
):
    """drain_streams should write to parquet and return the last id read per stream."""
    # Patch aioredis.from_url so drain_streams uses our fake redis
    import src.storage.snapshotter as snap

    async def fake_from_url(url, **kwargs):
        return fake_redis_with_data

    monkeypatch.setattr(snap.aioredis, "from_url", fake_from_url)

    streams = {
        "market_events": "market_events",
        "trade_events": "trade_events",
        "orderbook_events": "orderbook_events",
    }
    out_dir = tmp_path / "today"
    out_dir.mkdir()

    new_ckpt = await drain_streams(
        redis_url="redis://fake",
        streams=streams,
        since={},
        output_dir=out_dir,
    )

    # market_events and trade_events were populated; orderbook_events was empty
    assert "market_events" in new_ckpt
    assert "trade_events" in new_ckpt
    assert "orderbook_events" not in new_ckpt  # never had any data

    market_path = out_dir / "market_events.parquet"
    assert market_path.exists()
    table = pq.read_table(market_path)
    assert table.num_rows == 5

    trade_path = out_dir / "trade_events.parquet"
    assert trade_path.exists()
    assert pq.read_table(trade_path).num_rows == 3


async def test_drain_streams_idempotent(fake_redis_with_data, tmp_path: Path, monkeypatch):
    """A second drain with the same checkpoint should write zero new rows."""
    import src.storage.snapshotter as snap

    async def fake_from_url(url, **kwargs):
        return fake_redis_with_data

    monkeypatch.setattr(snap.aioredis, "from_url", fake_from_url)

    streams = {"market_events": "market_events"}
    out_dir = tmp_path / "today"
    out_dir.mkdir()

    ckpt1 = await drain_streams("redis://fake", streams, {}, out_dir)
    rows_after_first = pq.read_table(out_dir / "market_events.parquet").num_rows

    ckpt2 = await drain_streams("redis://fake", streams, ckpt1, out_dir)

    assert ckpt2["market_events"] == ckpt1["market_events"]
    # No new rows appended
    assert pq.read_table(out_dir / "market_events.parquet").num_rows == rows_after_first


async def test_drain_streams_reads_only_new_after_checkpoint(
    fake_redis_with_data, tmp_path: Path, monkeypatch
):
    """After checkpoint advance, adding more entries should only drain the new ones."""
    import src.storage.snapshotter as snap

    async def fake_from_url(url, **kwargs):
        return fake_redis_with_data

    monkeypatch.setattr(snap.aioredis, "from_url", fake_from_url)

    streams = {"market_events": "market_events"}
    out_dir = tmp_path / "today"
    out_dir.mkdir()

    ckpt1 = await drain_streams("redis://fake", streams, {}, out_dir)

    # Add 2 more entries
    await fake_redis_with_data.xadd("market_events", {"ticker": "BBB", "yes_bid": "40"})
    await fake_redis_with_data.xadd("market_events", {"ticker": "CCC", "yes_bid": "60"})

    ckpt2 = await drain_streams("redis://fake", streams, ckpt1, out_dir)
    assert ckpt2["market_events"] != ckpt1["market_events"]

    # File now has 5 + 2 = 7 rows total (existing 5 + 2 new appended)
    assert pq.read_table(out_dir / "market_events.parquet").num_rows == 7


async def test_run_snapshot_creates_dated_dir_and_checkpoint(
    fake_redis_with_data, tmp_path: Path, monkeypatch
):
    import src.storage.snapshotter as snap
    from src.config.redis_config import RedisConfig

    async def fake_from_url(url, **kwargs):
        return fake_redis_with_data

    monkeypatch.setattr(snap.aioredis, "from_url", fake_from_url)

    archive_root = tmp_path / "archive"
    cfg = RedisConfig()
    new_ckpt = await run_snapshot(cfg, archive_root)

    # A dated subdirectory should exist
    subdirs = [d for d in archive_root.iterdir() if d.is_dir()]
    assert len(subdirs) == 1
    # market_events.parquet should be present
    assert (subdirs[0] / "market_events.parquet").exists()
    # Checkpoint file exists with the advanced id
    ckpt = json.loads((archive_root / ".checkpoint.json").read_text())
    assert ckpt == new_ckpt
    assert ckpt.get("market_events") is not None
