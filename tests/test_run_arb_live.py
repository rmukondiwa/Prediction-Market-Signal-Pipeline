"""Smoke tests for scripts/run_arb_live.py — the orchestrator that ties
scanner → risk gates → multi-leg → reconciler → alerts → state together.

Tests focus on the trade pipeline (maybe_trade_arb) using stubs for both
trading clients. Real scanner subprocess + LLM verification are not
exercised here.
"""
from __future__ import annotations

import asyncio
import pytest

from scripts.run_arb_live import maybe_trade_arb
from src.execution.kalshi_trading_client import KalshiTradingClientStub
from src.execution.polymarket_trading_client import PolymarketTradingClientStub
from src.execution.reconciliation import Reconciler
from src.portfolio.risk_gates import RiskConfig
from src.portfolio.state import InMemoryBackend, PortfolioState


def _arb_dict(edge: float, k_yes_ask: float = 0.40, p_no_ask: float = 0.55) -> dict:
    """Build a synthetic arb candidate as the scanner would emit, with
    depth fields populated so tests don't need live Polymarket fetches."""
    return {
        "kalshi_ticker": "KXTEST-26JAN01-T100",
        "kalshi_title": "Test market — yes/no",
        "poly_id": "0xtest",
        "poly_question": "Test poly question",
        "poly_market": {
            "id": "0xtest",
            "question": "Test poly question",
            "clobTokenIds": '["123456789", "987654321"]',
            "outcomePrices": '["0.45", "0.55"]',
        },
        "k_yes_bid_dollars": 0.39,
        "k_yes_ask_dollars": k_yes_ask,
        "k_implied_prob": 0.40,
        "p_yes_bid": 0.44,
        "p_yes_ask": 0.45,
        "p_no_bid": 0.54,
        "p_no_ask": p_no_ask,
        "p_yes_ask_size": 1000.0,
        "p_no_ask_size": 1000.0,
        "poly_source": "clob",
        "arb_buy_K_yes_P_no_cost": k_yes_ask + p_no_ask,
        "arb_buy_K_no_P_yes_cost": (1.0 - 0.39) + 0.45,
        "best_arb_cost": k_yes_ask + p_no_ask,
        "edge": edge,
    }


@pytest.fixture
async def state():
    backend = InMemoryBackend()
    s = PortfolioState(backend, env="test_arb")
    await s.initialize(500.0)
    return s


async def test_dry_run_returns_dry_marker(state):
    arb = _arb_dict(edge=0.05)
    cfg = RiskConfig()
    rec = Reconciler()
    result = await maybe_trade_arb(
        arb=arb, state=state, risk_cfg=cfg,
        kalshi_client=KalshiTradingClientStub(),
        poly_client=PolymarketTradingClientStub(),
        reconciler=rec, max_trade_usd=50.0, dry_run=True,
    )
    assert result.get("dry_run") is True
    assert result["contracts"] > 0
    assert result["edge"] == 0.05


async def test_live_path_calls_both_clients(state):
    arb = _arb_dict(edge=0.05)
    cfg = RiskConfig()
    rec = Reconciler()
    k_client = KalshiTradingClientStub()
    p_client = PolymarketTradingClientStub()
    result = await maybe_trade_arb(
        arb=arb, state=state, risk_cfg=cfg,
        kalshi_client=k_client, poly_client=p_client,
        reconciler=rec, max_trade_usd=50.0, dry_run=False,
    )
    # Both stubs should have recorded one place_order call
    assert len(k_client.placed) == 1
    assert len(p_client.placed) == 1
    assert result.get("all_legs_succeeded") is True


async def test_kill_switch_skips_trade(state):
    # Force a kill-switch trip via realized_pnl
    backend = state.backend
    await backend.set(state._k("realized_pnl"), "-301.0")
    arb = _arb_dict(edge=0.05)
    cfg = RiskConfig(daily_loss_limit_usd=300.0)
    rec = Reconciler()
    result = await maybe_trade_arb(
        arb=arb, state=state, risk_cfg=cfg,
        kalshi_client=KalshiTradingClientStub(),
        poly_client=PolymarketTradingClientStub(),
        reconciler=rec, max_trade_usd=50.0, dry_run=False,
    )
    assert "skipped" in result


async def test_missing_arb_costs_skipped(state):
    arb = _arb_dict(edge=0.05)
    del arb["arb_buy_K_yes_P_no_cost"]
    cfg = RiskConfig()
    rec = Reconciler()
    result = await maybe_trade_arb(
        arb=arb, state=state, risk_cfg=cfg,
        kalshi_client=KalshiTradingClientStub(),
        poly_client=PolymarketTradingClientStub(),
        reconciler=rec, max_trade_usd=50.0, dry_run=False,
    )
    assert result.get("skipped") == "missing arb costs"
