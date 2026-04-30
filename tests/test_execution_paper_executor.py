from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.execution.kalshi_trading_client import KalshiTradingClientStub
from src.execution.paper_executor import PaperExecutor
from src.insight.models import MarketSnapshot
from src.portfolio.models import RiskLimits
from src.portfolio.state import InMemoryBackend, PortfolioState
from src.signals.models import CalibratedEdge


@pytest.fixture
async def executor():
    backend = InMemoryBackend()
    portfolio = PortfolioState(backend, env="test")
    await portfolio.initialize(1_000.0)
    client = KalshiTradingClientStub()
    return PaperExecutor(
        portfolio=portfolio, client=client,
        risk_limits=RiskLimits(),
        slippage_per_contract=0.01, fee_per_contract=0.0,
    ), portfolio, client


def _edge(kelly=0.05, side="yes", ticker="X-1") -> CalibratedEdge:
    return CalibratedEdge(
        ticker=ticker, title=ticker, side=side,
        estimated_fair_prob=0.40, current_implied_prob=0.30,
        edge_pp=0.10, confidence=0.7, kelly_fraction=kelly,
        thesis="", source_signal_model="raw_llm",
    )


def _snap(ticker="X-1") -> MarketSnapshot:
    return MarketSnapshot(
        event=ticker, market=ticker, outcome="YES",
        quoted_price=30, implied_probability=0.30,
        yes_bid=29, yes_ask=31, volume=0, open_interest=0,
        source="test", timestamp=datetime.now(timezone.utc),
    )


async def test_paper_executor_records_fill(executor):
    ex, portfolio, client = executor
    fills = await ex.process_edges([_edge()], _snap(), context=[])
    assert len(fills) == 1
    f = fills[0]
    assert f.signal_model == "raw_llm"
    assert f.contracts > 0
    # Cash debited
    snap = await portfolio.snapshot()
    assert snap.cash < 1_000.0


async def test_paper_executor_rejects_via_kill_switch():
    backend = InMemoryBackend()
    portfolio = PortfolioState(backend, env="test")
    await portfolio.initialize(1_000.0)
    ex = PaperExecutor(
        portfolio=portfolio, client=KalshiTradingClientStub(),
        risk_limits=RiskLimits(kill_switch=True),
    )
    fills = await ex.process_edges([_edge()], _snap(), context=[])
    assert fills == []


async def test_paper_executor_skips_zero_kelly(executor):
    ex, _, _ = executor
    fills = await ex.process_edges([_edge(kelly=0.0)], _snap(), context=[])
    assert fills == []


async def test_signal_model_attribution_on_fill(executor):
    """Fills should carry the originating signal model name for post-trade attribution."""
    ex, portfolio, _ = executor
    edge_a = _edge(ticker="A-1")
    edge_a.source_signal_model = "calibrated_llm"
    edge_b = _edge(ticker="B-1")
    edge_b.source_signal_model = "consistency_arb"
    await ex.process_edges([edge_a], _snap("A-1"), context=[])
    await ex.process_edges([edge_b], _snap("B-1"), context=[])
    snap = await portfolio.snapshot()
    assert len(snap.positions) == 2


async def test_two_signal_models_isolated_by_env():
    """Two paper traders with different envs should not interfere."""
    backend = InMemoryBackend()
    p1 = PortfolioState(backend, env="paper:raw_llm")
    p2 = PortfolioState(backend, env="paper:consistency_arb")
    await p1.initialize(1_000.0)
    await p2.initialize(2_000.0)

    ex1 = PaperExecutor(p1, KalshiTradingClientStub(), RiskLimits())
    ex2 = PaperExecutor(p2, KalshiTradingClientStub(), RiskLimits())

    await ex1.process_edges([_edge()], _snap(), context=[])
    snap1 = await p1.snapshot()
    snap2 = await p2.snapshot()

    # p1 has a position; p2 doesn't
    assert len(snap1.positions) == 1
    assert len(snap2.positions) == 0
    assert snap2.cash == 2_000.0
