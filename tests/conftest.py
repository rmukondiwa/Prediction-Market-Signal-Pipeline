"""Shared pytest fixtures for Phase 5 tests."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `import src...` works regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_archive_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    return root


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "backtest_cache"
    cache.mkdir()
    return cache


@pytest.fixture
def sample_market_snapshot():
    from src.insight.models import MarketSnapshot
    return MarketSnapshot(
        event="Iran strike on US forces",
        market="KXIRANUS-26JUN30-YES",
        outcome="YES",
        quoted_price=23,
        implied_probability=0.23,
        yes_bid=22,
        yes_ask=24,
        volume=12_000,
        open_interest=8_500,
        source="kalshi",
        timestamp=datetime(2026, 4, 15, 14, 30, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_catalog():
    from src.catalog.models import CatalogMarket
    return [
        CatalogMarket(
            ticker="KXIRANUS-26JUN30-YES",
            event_ticker="KXIRANUS-26JUN30",
            title="Iran strike on US forces by June 30, 2026",
            subtitle="Yes",
            category="World",
            status="open",
            yes_bid=22,
            yes_ask=24,
            implied_probability=0.23,
        ),
        CatalogMarket(
            ticker="KXIRANUS-26DEC31-YES",
            event_ticker="KXIRANUS-26DEC31",
            title="Iran strike on US forces by December 31, 2026",
            subtitle="Yes",
            category="World",
            status="open",
            yes_bid=35,
            yes_ask=37,
            implied_probability=0.36,
        ),
        CatalogMarket(
            ticker="KXOIL100-26JUL31-YES",
            event_ticker="KXOIL100-26JUL31",
            title="Brent crude above $100 by July 31, 2026",
            subtitle="Yes",
            category="Commodities",
            status="open",
            yes_bid=45,
            yes_ask=47,
            implied_probability=0.46,
        ),
    ]


@pytest.fixture
def sample_context_markets(sample_catalog):
    from src.context.models import ContextMarket
    return [
        ContextMarket(
            ticker="KXOIL100-26JUL31-YES",
            event_ticker="KXOIL100-26JUL31",
            title="Brent crude above $100 by July 31, 2026",
            subtitle="Yes",
            category="Commodities",
            status="open",
            yes_bid=45,
            yes_ask=47,
            implied_probability=0.46,
            similarity_score=0.71,
            relevance_score=8.5,
            relationship="Iran military action would disrupt supply via the Strait of Hormuz, pushing oil higher.",
        ),
    ]


@pytest.fixture
def sample_inference_report(sample_market_snapshot, sample_context_markets):
    from src.inference.models import (
        DerivedProbability,
        Edge,
        InferenceReport,
        Mispricing,
    )
    return InferenceReport(
        focus_market=sample_market_snapshot,
        context_markets=sample_context_markets,
        consistency_analysis="No axiom violations detected.",
        derived_probabilities=[
            DerivedProbability(
                description="P(oil>100 | Iran strike)",
                value=0.78,
                reasoning="Joint priced at ~0.18, marginal Iran ~0.23 → 0.18/0.23.",
            )
        ],
        detected_mispricings=[
            Mispricing(
                ticker="KXIRANUS-26JUN30-YES",
                title="Iran strike on US forces by June 30, 2026",
                direction="underpriced",
                current_implied_prob=0.23,
                estimated_fair_prob=0.30,
                reasoning="Conditional structure implies higher fair value.",
            )
        ],
        suggested_edges=[
            Edge(
                ticker="KXIRANUS-26JUN30-YES",
                title="Iran strike on US forces by June 30, 2026",
                side="yes",
                confidence="medium",
                thesis="Conditional on oil markets, fair value is ~30%.",
                kelly_fraction=0.05,
            )
        ],
    )
