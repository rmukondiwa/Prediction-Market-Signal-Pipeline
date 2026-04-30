from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.execution.kalshi_trading_client import KalshiTradingClientStub
from src.execution.order_manager import OrderManager
from src.insight.models import MarketSnapshot
from src.portfolio.state import InMemoryBackend, PortfolioState
from src.signals.models import CalibratedEdge


@pytest.fixture
async def manager():
    backend = InMemoryBackend()
    portfolio = PortfolioState(backend, env="test")
    await portfolio.initialize(1_000.0)
    client = KalshiTradingClientStub()
    return OrderManager(client, portfolio), portfolio, client


def _edge(side="yes") -> CalibratedEdge:
    return CalibratedEdge(
        ticker="X-1", title="x", side=side,
        estimated_fair_prob=0.40, current_implied_prob=0.30,
        edge_pp=0.10, confidence=0.7, kelly_fraction=0.05,
        thesis="", source_signal_model="raw_llm",
    )


def _snap() -> MarketSnapshot:
    return MarketSnapshot(
        event="x", market="X-1", outcome="YES",
        quoted_price=30, implied_probability=0.30,
        yes_bid=29, yes_ask=31, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )


async def test_zero_contracts_returns_rejection(manager):
    om, _, client = manager
    result = await om.place(_edge(), _snap(), contracts=0)
    assert not result.accepted
    assert result.error == "zero_contracts"
    assert client.placed == []


async def test_yes_side_posts_at_ask(manager):
    om, portfolio, client = manager
    result = await om.place(_edge("yes"), _snap(), contracts=10)
    assert result.accepted
    assert len(client.placed) == 1
    placed = client.placed[0]
    # yes_ask=31 cents → limit=0.31
    assert abs(placed.limit_price - 0.31) < 1e-9
    # WorkingOrder persisted
    orders = await portfolio.list_working_orders()
    assert len(orders) == 1
    assert orders[0].ticker == "X-1"


async def test_no_side_posts_at_complement_of_yes_bid(manager):
    om, _, client = manager
    await om.place(_edge("no"), _snap(), contracts=10)
    placed = client.placed[0]
    # yes_bid=29 → no limit = (100-29)/100 = 0.71
    assert abs(placed.limit_price - 0.71) < 1e-9


async def test_client_order_id_is_unique(manager):
    om, _, client = manager
    await om.place(_edge(), _snap(), contracts=5)
    await om.place(_edge(), _snap(), contracts=5)
    ids = {p.client_order_id for p in client.placed}
    assert len(ids) == 2
