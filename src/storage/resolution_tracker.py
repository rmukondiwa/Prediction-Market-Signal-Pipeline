"""
Settlement / resolution tracker.

For each market we have ever observed (in any prior catalog snapshot), poll the
Kalshi REST API for resolution status and append to data/archive/resolutions.parquet.

This is the ground truth dataset every backtest depends on.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from pydantic import BaseModel

from src.utils.logging import get_logger
from src.utils.retry import retry_with_backoff

logger = get_logger(__name__)

_RESOLUTIONS_FILENAME = "resolutions.parquet"
# Treat anything other than active/open as terminal — once we have a result
# we never need to re-poll.
_TERMINAL_STATUSES = {"settled", "finalized", "closed"}


class ResolutionRecord(BaseModel):
    ticker: str
    status: str
    result: str | None
    settlement_value: float | None
    close_time: datetime | None
    settled_time: datetime | None
    last_checked: datetime


def _resolutions_path(archive_root: Path) -> Path:
    return archive_root / _RESOLUTIONS_FILENAME


def load_resolutions(archive_root: Path) -> dict[str, ResolutionRecord]:
    """Return ticker → ResolutionRecord for everything previously written."""
    path = _resolutions_path(archive_root)
    if not path.exists():
        return {}
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    out: dict[str, ResolutionRecord] = {}
    for r in rows:
        out[r["ticker"]] = ResolutionRecord(**r)
    return out


def save_resolutions(records: list[ResolutionRecord], archive_root: Path) -> Path:
    """Overwrite the resolutions parquet with the full set of records."""
    if not records:
        logger.info("No resolution records to save")
        return _resolutions_path(archive_root)
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r.model_dump(mode="python") for r in records]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    path = _resolutions_path(archive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    logger.info("Resolutions saved", extra={"path": str(path), "count": len(records)})
    return path


def _parse_dt(value: Any) -> datetime | None:
    if not value or value in {"", None}:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _settlement_value_from_result(result: str | None) -> float | None:
    if result is None:
        return None
    r = result.lower()
    if r in {"yes", "y", "1", "true"}:
        return 1.0
    if r in {"no", "n", "0", "false"}:
        return 0.0
    return None


async def _fetch_market(
    session: aiohttp.ClientSession,
    base_url: str,
    ticker: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async def _do() -> dict:
        async with semaphore:
            await asyncio.sleep(0.05)
            async with session.get(f"{base_url}/markets/{ticker}") as resp:
                if resp.status == 404:
                    return {}
                resp.raise_for_status()
                return await resp.json()

    try:
        data = await retry_with_backoff(_do, max_attempts=4, base_delay=1.5, label=f"fetch_market:{ticker}")
        return data.get("market") if data else None
    except Exception as exc:
        logger.warning("Resolution fetch failed", extra={"ticker": ticker, "error": str(exc)})
        return None


async def track_resolutions(
    base_url: str,
    tickers: list[str],
    archive_root: Path = Path("data/archive"),
    concurrency: int = 8,
) -> list[ResolutionRecord]:
    """
    For each ticker, fetch its latest market state. Skip those already at a
    terminal status in the existing resolutions table. Write the merged result
    to resolutions.parquet and return the full updated record list.
    """
    archive_root.mkdir(parents=True, exist_ok=True)
    existing = load_resolutions(archive_root)
    to_check = [t for t in tickers if existing.get(t) is None or existing[t].status not in _TERMINAL_STATUSES]
    logger.info(
        "Resolutions pass",
        extra={"total": len(tickers), "terminal_skip": len(tickers) - len(to_check), "to_check": len(to_check)},
    )

    semaphore = asyncio.Semaphore(concurrency)
    now = datetime.now(timezone.utc)

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(_fetch_market(session, base_url, t, semaphore) for t in to_check)
        )

    updated = dict(existing)
    for ticker, market in zip(to_check, results):
        if not market:
            # leave existing record (if any) alone
            if ticker not in updated:
                updated[ticker] = ResolutionRecord(
                    ticker=ticker, status="unknown", result=None,
                    settlement_value=None, close_time=None, settled_time=None,
                    last_checked=now,
                )
            else:
                updated[ticker] = updated[ticker].model_copy(update={"last_checked": now})
            continue

        status = market.get("status", "unknown")
        result = market.get("result")
        if isinstance(result, str) and not result.strip():
            result = None
        updated[ticker] = ResolutionRecord(
            ticker=ticker,
            status=status,
            result=result,
            settlement_value=_settlement_value_from_result(result),
            close_time=_parse_dt(market.get("close_time")),
            settled_time=_parse_dt(market.get("settle_time") or market.get("settled_time")),
            last_checked=now,
        )

    records = sorted(updated.values(), key=lambda r: r.ticker)
    save_resolutions(records, archive_root)
    return records
