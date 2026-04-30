from __future__ import annotations

from datetime import datetime, timezone

from src.portfolio.models import (
    PortfolioSnapshot,
    Position,
    RiskLimits,
)
from src.portfolio.risk import RiskManager
from src.signals.models import CalibratedEdge


def _edge(kelly: float = 0.10, ticker: str = "X", side: str = "yes",
          implied: float = 0.50) -> CalibratedEdge:
    return CalibratedEdge(
        ticker=ticker, title="t", side=side,
        estimated_fair_prob=implied + 0.10,
        current_implied_prob=implied,
        edge_pp=0.10, confidence=0.7,
        kelly_fraction=kelly,
        thesis="", source_signal_model="test",
    )


def _empty_snapshot(cash: float = 10_000.0, daily_pnl: float = 0.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=cash, positions=[], working_orders=[],
        realized_pnl=0.0, unrealized_pnl=0.0, daily_pnl=daily_pnl,
        timestamp=datetime.now(timezone.utc),
    )


def test_kill_switch_blocks_everything():
    rm = RiskManager(RiskLimits(kill_switch=True))
    decision = rm.check(_edge(), _empty_snapshot())
    assert not decision.approved
    assert "kill_switch" in decision.reason


def test_daily_loss_limit_blocks_when_breached():
    rm = RiskManager(RiskLimits(daily_loss_limit_usd=500.0))
    decision = rm.check(_edge(), _empty_snapshot(daily_pnl=-501.0))
    assert not decision.approved
    assert "daily_loss_limit" in decision.reason


def test_daily_loss_limit_allows_when_not_breached():
    rm = RiskManager(RiskLimits(daily_loss_limit_usd=500.0))
    decision = rm.check(_edge(), _empty_snapshot(daily_pnl=-499.0))
    assert decision.approved


def test_per_market_cap_scale_down():
    """Existing notional 400 + new 200 > cap 500 → scale to (100/200) = 0.5"""
    rm = RiskManager(RiskLimits(max_per_market_usd=500.0))
    pos = Position(
        ticker="X", side="yes", contracts=800, avg_cost=0.50,
        opened_at=datetime.now(timezone.utc), last_updated=datetime.now(timezone.utc),
    )
    snap = PortfolioSnapshot(
        cash=2_000.0, positions=[pos], working_orders=[],
        realized_pnl=0.0, unrealized_pnl=0.0, daily_pnl=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    # Edge wants 10% of 2000 = 200 notional
    decision = rm.check(_edge(kelly=0.10, ticker="X"), snap)
    assert decision.approved
    # existing position: 800 contracts × 0.50 avg cost = 400 USD notional
    # cap = 500, so available = 100 → scale = 100/200 = 0.5
    assert abs(decision.scale_factor - 0.5) < 1e-9


def test_per_market_cap_full_blocks():
    rm = RiskManager(RiskLimits(max_per_market_usd=500.0))
    pos = Position(
        ticker="X", side="yes", contracts=1000, avg_cost=0.50,
        opened_at=datetime.now(timezone.utc), last_updated=datetime.now(timezone.utc),
    )
    snap = PortfolioSnapshot(
        cash=2_000.0, positions=[pos], working_orders=[],
        realized_pnl=0.0, unrealized_pnl=0.0, daily_pnl=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    decision = rm.check(_edge(kelly=0.10, ticker="X"), snap)
    assert not decision.approved
    assert "per_market_cap_full" in decision.reason


def test_total_exposure_cap_blocks():
    rm = RiskManager(RiskLimits(
        max_per_market_usd=10_000,  # don't trigger per-market
        max_total_exposure_usd=1_000.0,
    ))
    # Existing: 2000 contracts × 0.50 = 1000 already at the cap
    pos = Position(
        ticker="OTHER", side="yes", contracts=2000, avg_cost=0.50,
        opened_at=datetime.now(timezone.utc), last_updated=datetime.now(timezone.utc),
    )
    snap = PortfolioSnapshot(
        cash=10_000.0, positions=[pos], working_orders=[],
        realized_pnl=0.0, unrealized_pnl=0.0, daily_pnl=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    decision = rm.check(_edge(kelly=0.10, ticker="X"), snap)
    assert not decision.approved
    assert "total_exposure" in decision.reason


def test_kelly_clamps_to_max():
    """If edge.kelly > limits.max_kelly_fraction, the cap should still apply
    via downstream sizing — risk decision still approves but notional is
    based on the clamped value."""
    rm = RiskManager(RiskLimits(max_kelly_fraction=0.10))
    decision = rm.check(_edge(kelly=0.25), _empty_snapshot())
    # The risk manager itself returns approved; sizer applies the clamp
    assert decision.approved


def test_clean_path_returns_ok():
    rm = RiskManager(RiskLimits())
    decision = rm.check(_edge(kelly=0.05), _empty_snapshot())
    assert decision.approved
    assert decision.scale_factor == 1.0
    assert decision.reason == "ok"
