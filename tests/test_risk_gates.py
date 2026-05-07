"""Tests for src/portfolio/risk_gates.py — extracted gates that are now
strategy-agnostic (no longer tied to the archived decay trader)."""
from __future__ import annotations

from src.portfolio.risk_gates import (
    RiskConfig,
    RiskState,
    compute_size,
    drawdown_size_multiplier,
    kill_switch_state,
    underlying_of,
    TIER_CAP_FRACTIONS,
)


def test_drawdown_multiplier_brackets():
    assert drawdown_size_multiplier(50, 300) == 1.0
    assert drawdown_size_multiplier(0, 300) == 1.0
    assert drawdown_size_multiplier(-50, 300) == 1.0
    assert drawdown_size_multiplier(-100, 300) == 0.5
    assert drawdown_size_multiplier(-200, 300) == 0.25
    assert drawdown_size_multiplier(-300, 300) == 0.0
    assert drawdown_size_multiplier(-400, 300) == 0.0
    assert drawdown_size_multiplier(-1000, None) == 1.0  # disabled


def test_kill_switch_daily_limit():
    cfg = RiskConfig(daily_loss_limit_usd=300.0, drawdown_limit_usd=None)
    state = RiskState(cash=5000, realized_pnl=-301, peak_realized_pnl=0)
    halt, reason = kill_switch_state(state, cfg)
    assert halt and reason == "daily_loss_limit_exceeded"


def test_kill_switch_drawdown():
    cfg = RiskConfig(daily_loss_limit_usd=1000.0, drawdown_limit_usd=200.0)
    state = RiskState(cash=5000, realized_pnl=-50, peak_realized_pnl=200)
    halt, reason = kill_switch_state(state, cfg)
    assert halt and reason == "drawdown_limit_exceeded"


def test_kill_switch_clean_state():
    cfg = RiskConfig(daily_loss_limit_usd=300, drawdown_limit_usd=200)
    state = RiskState(cash=5000, realized_pnl=50, peak_realized_pnl=100)
    halt, reason = kill_switch_state(state, cfg)
    assert not halt and reason is None


def test_compute_size_clean_state():
    cfg = RiskConfig()
    state = RiskState(cash=5000, realized_pnl=50, peak_realized_pnl=100)
    n, info = compute_size(ask_price=0.50, ask_size=1000, state=state, cfg=cfg, threshold_tier=2)
    assert n > 0
    assert "binding" in info


def test_compute_size_per_market_tier_binds_tier_0():
    """Tier 0 cap (30% of $500 = $150) at $0.50 = 300 contracts max."""
    cfg = RiskConfig(max_per_market_usd=500.0)
    state = RiskState(cash=5000, realized_pnl=0, peak_realized_pnl=0)
    n, info = compute_size(ask_price=0.50, ask_size=10_000, state=state, cfg=cfg, threshold_tier=0)
    # bankroll: 5000*0.05/0.5 = 500. depth: 10000*0.25 = 2500. tier-cap: 150/0.5 = 300.
    # asset cap: 1000/0.5 = 2000. min = 300 (tier 0)
    assert n == 300
    assert info["binding"] == "per_market_tier"


def test_compute_size_per_asset_binds():
    cfg = RiskConfig(max_per_asset_usd=1000.0, max_per_market_usd=10_000.0)
    state = RiskState(cash=50_000, realized_pnl=0, peak_realized_pnl=0,
                      existing_asset_notional=800.0)
    n, info = compute_size(ask_price=0.50, ask_size=10_000, state=state, cfg=cfg, threshold_tier=2)
    # remaining asset: 1000-800 = 200 → 200/0.5 = 400 contracts
    assert n <= 400
    assert info["binding"] == "per_asset"


def test_compute_size_drawdown_ramp_halves():
    cfg = RiskConfig(daily_loss_limit_usd=300.0)
    full_state = RiskState(cash=5000, realized_pnl=0, peak_realized_pnl=0)
    halved_state = RiskState(cash=5000, realized_pnl=-150, peak_realized_pnl=0)
    n_full, _ = compute_size(0.5, 10_000, full_state, cfg, threshold_tier=2)
    n_half, _ = compute_size(0.5, 10_000, halved_state, cfg, threshold_tier=2)
    assert abs(n_half - n_full // 2) <= 1


def test_compute_size_kill_switch_returns_zero():
    cfg = RiskConfig(daily_loss_limit_usd=300.0)
    state = RiskState(cash=5000, realized_pnl=-301, peak_realized_pnl=0)
    n, info = compute_size(0.5, 10_000, state, cfg)
    assert n == 0
    assert info.get("halted") is True


def test_underlying_extraction():
    assert underlying_of("KXBTC15M-26MAY051645-45") == "KXBTC15M"
    assert underlying_of("KXETHD-26MAR0704-T2169.99") == "KXETHD"
    assert underlying_of("NOMARK") == "NOMARK"


def test_tier_caps_are_cumulative():
    """Sanity: tier 0 < tier 1 < tier 2 = 100%."""
    assert TIER_CAP_FRACTIONS[0] < TIER_CAP_FRACTIONS[1] < TIER_CAP_FRACTIONS[2]
    assert TIER_CAP_FRACTIONS[2] == 1.0
