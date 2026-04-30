"""
Daily catalog archive.

Captures the Kalshi market universe as it exists today into a parquet file,
so backtests can reconstruct the catalog at any historical date and avoid
look-ahead bias from using today's catalog for past reasoning.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.catalog.fetcher import build_catalog
from src.catalog.models import CatalogMarket
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _today_dir(archive_root: Path) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = archive_root / today
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_catalog_parquet(catalog: list[CatalogMarket], path: Path) -> None:
    """Pure function: write a catalog list to parquet at the given path."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [m.model_dump(mode="json") for m in catalog]
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def load_catalog_parquet(path: Path) -> list[CatalogMarket]:
    """Inverse of write_catalog_parquet."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [CatalogMarket(**r) for r in rows]


async def snapshot_catalog(
    base_url: str,
    archive_root: Path = Path("data/archive"),
    max_markets: int | None = None,
) -> Path:
    """
    Fetch today's catalog from Kalshi REST and write to
    archive_root/YYYY-MM-DD/catalog.parquet. Returns the written path.
    """
    catalog = await build_catalog(base_url, max_markets=max_markets)
    out_path = _today_dir(archive_root) / "catalog.parquet"
    write_catalog_parquet(catalog, out_path)
    logger.info(
        "Catalog snapshot written",
        extra={"path": str(out_path), "markets": len(catalog)},
    )
    return out_path


def load_catalog_for_date(archive_root: Path, target_date: str) -> list[CatalogMarket] | None:
    """
    Load the catalog snapshot for `target_date` (YYYY-MM-DD).
    If that exact day is missing, walks backward up to 14 days; returns None if
    nothing is found in the window. Caller is expected to log the staleness.
    """
    target = datetime.strptime(target_date, "%Y-%m-%d").date()
    for delta in range(0, 15):
        from datetime import timedelta
        candidate = target - timedelta(days=delta)
        path = archive_root / candidate.strftime("%Y-%m-%d") / "catalog.parquet"
        if path.exists():
            if delta > 0:
                logger.warning(
                    "Catalog snapshot stale by N days",
                    extra={"target": target_date, "used": str(candidate), "stale_by": delta},
                )
            return load_catalog_parquet(path)
    return None
