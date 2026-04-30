"""
Stream snapshotter.

Drains Redis Streams (`market_events`, `trade_events`, `orderbook_events`) into
dated parquet files under data/archive/YYYY-MM-DD/. Idempotent — uses a checkpoint
file to track the last id read per stream so re-runs don't duplicate data.

Designed to be invoked daily via cron / systemd timer. Every day not collecting
is a day of book history we cannot recreate.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis

from src.config.redis_config import RedisConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_STREAM_KEYS = ("market_events_stream", "trade_events_stream", "orderbook_events_stream")
_CHECKPOINT_FILENAME = ".checkpoint.json"
_DEFAULT_BATCH = 5000


def _today_dir(archive_root: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = archive_root / today
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_checkpoint(archive_root: Path) -> dict[str, str]:
    path = archive_root / _CHECKPOINT_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Checkpoint corrupt, starting from 0", extra={"path": str(path)})
        return {}


def _save_checkpoint(archive_root: Path, checkpoint: dict[str, str]) -> None:
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / _CHECKPOINT_FILENAME).write_text(json.dumps(checkpoint, indent=2))


def _entries_to_rows(entries: list[tuple[str, dict[str, str]]]) -> list[dict[str, Any]]:
    """Flatten Redis entries into row-shaped dicts ready for pyarrow."""
    rows: list[dict[str, Any]] = []
    for msg_id, fields in entries:
        row: dict[str, Any] = {"_msg_id": msg_id}
        for k, v in fields.items():
            row[k] = v
        rows.append(row)
    return rows


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to parquet, appending to an existing file when present."""
    if not rows:
        return
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if path.exists():
        existing = pq.read_table(path)
        # pandas concat handles schema reconciliation across rows that may differ
        merged_df = pd.concat([existing.to_pandas(), df], ignore_index=True)
        table = pa.Table.from_pandas(merged_df, preserve_index=False)

    pq.write_table(table, path)


async def drain_streams(
    redis_url: str,
    streams: dict[str, str],
    since: dict[str, str],
    output_dir: Path,
    batch: int = _DEFAULT_BATCH,
) -> dict[str, str]:
    """
    Read entries from each stream from `since[stream]` to `+`, write to parquet
    under output_dir/<stream_basename>.parquet, and return updated checkpoint dict.

    `streams` maps a logical name (e.g. "market_events") to the Redis stream key.
    """
    redis = await aioredis.from_url(redis_url, decode_responses=True)
    new_checkpoint: dict[str, str] = dict(since)

    try:
        for logical, stream_key in streams.items():
            start = since.get(stream_key, "0-0")
            # Redis xrange is inclusive on both ends; advance start by 1 to avoid
            # re-reading the last id from the previous run.
            if start != "0-0":
                start = _next_id(start)

            entries: list[tuple[str, dict[str, str]]] = []
            cursor = start
            while True:
                batch_entries = await redis.xrange(stream_key, min=cursor, max="+", count=batch)
                if not batch_entries:
                    break
                entries.extend(batch_entries)
                if len(batch_entries) < batch:
                    break
                last_id = batch_entries[-1][0]
                cursor = _next_id(last_id)

            if entries:
                rows = _entries_to_rows(entries)
                out = output_dir / f"{logical}.parquet"
                _write_parquet(rows, out)
                last_seen = entries[-1][0]
                new_checkpoint[stream_key] = last_seen
                logger.info(
                    "Stream drained",
                    extra={"stream": stream_key, "count": len(entries), "last_id": last_seen},
                )
            else:
                logger.info("Stream empty", extra={"stream": stream_key, "since": since.get(stream_key, "0-0")})
    finally:
        await redis.aclose()

    return new_checkpoint


def _next_id(stream_id: str) -> str:
    """Compute the next valid Redis Stream id strictly greater than `stream_id`."""
    if "-" in stream_id:
        ms_part, seq_part = stream_id.split("-", 1)
        try:
            seq = int(seq_part)
        except ValueError:
            return stream_id
        return f"{ms_part}-{seq + 1}"
    return stream_id


async def run_snapshot(
    redis_config: RedisConfig,
    archive_root: Path = Path("data/archive"),
) -> dict[str, str]:
    """
    Daily snapshot entry point. Reads checkpoint, drains streams to today's
    directory, advances checkpoint, returns the new checkpoint dict.
    """
    archive_root.mkdir(parents=True, exist_ok=True)
    output_dir = _today_dir(archive_root)
    checkpoint = _load_checkpoint(archive_root)

    streams = {
        "market_events": redis_config.market_events_stream,
        "trade_events": redis_config.trade_events_stream,
        "orderbook_events": redis_config.orderbook_events_stream,
    }

    new_checkpoint = await drain_streams(
        redis_url=redis_config.url,
        streams=streams,
        since=checkpoint,
        output_dir=output_dir,
    )

    _save_checkpoint(archive_root, new_checkpoint)
    logger.info("Snapshot complete", extra={"output_dir": str(output_dir)})
    return new_checkpoint
