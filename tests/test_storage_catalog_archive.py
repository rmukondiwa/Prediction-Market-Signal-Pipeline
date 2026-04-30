from __future__ import annotations

from pathlib import Path

from src.catalog.models import CatalogMarket
from src.storage.catalog_archive import (
    load_catalog_for_date,
    load_catalog_parquet,
    write_catalog_parquet,
)


def _sample_catalog() -> list[CatalogMarket]:
    return [
        CatalogMarket(
            ticker="A-1", event_ticker="A", title="A title", subtitle="",
            category="Politics", status="open", yes_bid=10, yes_ask=12,
            implied_probability=0.11,
        ),
        CatalogMarket(
            ticker="B-1", event_ticker="B", title="B title", subtitle="Yes",
            category="Finance", status="open", yes_bid=80, yes_ask=82,
            implied_probability=0.81,
        ),
    ]


def test_write_and_load_roundtrip(tmp_path: Path):
    catalog = _sample_catalog()
    p = tmp_path / "catalog.parquet"
    write_catalog_parquet(catalog, p)
    loaded = load_catalog_parquet(p)
    assert len(loaded) == 2
    assert loaded[0].ticker == "A-1"
    assert loaded[1].implied_probability == 0.81


def test_load_catalog_for_date_walks_backward(tmp_path: Path):
    archive_root = tmp_path / "archive"
    # Only write a snapshot for 2026-04-25
    day_dir = archive_root / "2026-04-25"
    day_dir.mkdir(parents=True)
    write_catalog_parquet(_sample_catalog(), day_dir / "catalog.parquet")

    # Asking for 2026-04-30 should walk back and find 2026-04-25
    loaded = load_catalog_for_date(archive_root, "2026-04-30")
    assert loaded is not None
    assert len(loaded) == 2


def test_load_catalog_for_date_returns_none_when_too_old(tmp_path: Path):
    archive_root = tmp_path / "archive"
    day_dir = archive_root / "2025-01-01"
    day_dir.mkdir(parents=True)
    write_catalog_parquet(_sample_catalog(), day_dir / "catalog.parquet")

    # Snapshot is more than 14 days before target → None
    loaded = load_catalog_for_date(archive_root, "2026-04-30")
    assert loaded is None
