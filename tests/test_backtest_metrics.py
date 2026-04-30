from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.backtest.metrics import (
    brier_score,
    daily_returns_from_curve,
    edge_decay,
    equity_curve,
    hit_rate_by_confidence,
    market_baseline_brier,
    max_drawdown,
    sharpe,
)
from src.backtest.models import SimulatedFill


def _fill(ticker="T", side="yes", price=0.30, contracts=10,
          fair=0.50, implied=0.30, t=None, model="raw_llm") -> SimulatedFill:
    return SimulatedFill(
        ticker=ticker, side=side, contracts=contracts, price=price,
        fee=contracts * 0.05, timestamp=t or datetime.now(timezone.utc),
        signal_model=model, edge_thesis="",
        estimated_fair_prob=fair, current_implied_prob=implied,
    )


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0


def test_brier_score_worst_predictions_is_one():
    assert brier_score([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_brier_score_baseline():
    # All 50% predictions → 0.25 mean squared error
    assert abs(brier_score([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0]) - 0.25) < 1e-9


def test_brier_score_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1.0])


def test_brier_score_empty_returns_zero():
    assert brier_score([], []) == 0.0


def test_market_baseline_brier_alias():
    # Same function, different name for clarity at call sites
    assert market_baseline_brier([0.3], [1.0]) == brier_score([0.3], [1.0])


def test_equity_curve_winning_trade():
    fills = [_fill(ticker="A", side="yes", price=0.30, contracts=10)]
    outcomes = {"A": 1.0}
    curve = equity_curve(fills, outcomes, starting_capital=100.0)
    # Buy 10 yes at $0.30: cost $3.00 + fee $0.50; payout $10 (yes won)
    # PnL = 10*(1-0.30) - 0.50 = 6.50
    assert abs(curve[-1] - 106.50) < 1e-6


def test_equity_curve_losing_trade():
    fills = [_fill(ticker="A", side="yes", price=0.30, contracts=10)]
    outcomes = {"A": 0.0}
    curve = equity_curve(fills, outcomes, starting_capital=100.0)
    # Buy 10 yes at $0.30: cost $3.00 + fee $0.50; payout 0
    # PnL = -3.00 - 0.50 = -3.50
    assert abs(curve[-1] - 96.50) < 1e-6


def test_equity_curve_no_outcome_keeps_equity_flat():
    fills = [_fill(ticker="A", side="yes")]
    curve = equity_curve(fills, outcomes={}, starting_capital=100.0)
    # No settlement → equity unchanged
    assert curve[-1] == 100.0


def test_max_drawdown_zero_for_monotone():
    assert max_drawdown([100, 110, 120, 130]) == 0.0


def test_max_drawdown_simple():
    # Peak 100, trough 80 → 20% drawdown
    assert abs(max_drawdown([100, 80, 90]) - 0.20) < 1e-9


def test_sharpe_constant_returns_is_none():
    """No variance → Sharpe undefined → return None."""
    assert sharpe([0.01, 0.01, 0.01]) is None


def test_sharpe_positive_returns_positive_sharpe():
    sh = sharpe([0.01, 0.02, 0.005, 0.015])
    assert sh is not None
    assert sh > 0


def test_sharpe_too_few_samples():
    assert sharpe([0.01]) is None


def test_daily_returns_from_curve():
    rs = daily_returns_from_curve([100.0, 110.0, 99.0])
    assert len(rs) == 2
    assert abs(rs[0] - 0.10) < 1e-9
    assert abs(rs[1] - (-0.10)) < 1e-9


def test_hit_rate_by_confidence_buckets():
    # Bucket boundaries (in distance from 0.5):
    #   < 0.10  → low
    #   < 0.25  → medium
    #   else    → high
    fills = [
        _fill(ticker="A", side="yes", fair=0.95),  # d=0.45 → high
        _fill(ticker="B", side="yes", fair=0.55),  # d=0.05 → low
        _fill(ticker="C", side="yes", fair=0.65),  # d=0.15 → medium
        _fill(ticker="D", side="yes", fair=0.85),  # d=0.35 → high
    ]
    outcomes = {"A": 1.0, "B": 0.0, "C": 1.0, "D": 0.0}
    by_conf = hit_rate_by_confidence(fills, outcomes)
    assert set(by_conf.keys()) == {"low", "medium", "high"}
    assert by_conf["high"]["trades"] == 2.0
    assert by_conf["high"]["wins"] == 1.0      # only A won
    assert by_conf["high"]["hit_rate"] == 0.5
    assert by_conf["low"]["trades"] == 1.0
    assert by_conf["low"]["wins"] == 0.0       # B lost
    assert by_conf["medium"]["wins"] == 1.0    # C won


def test_edge_decay_buckets_by_hours_to_settlement():
    base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    settle = base + timedelta(hours=10)
    fills = [
        _fill(ticker="A", t=base + timedelta(hours=9)),   # 1h to settle (high)
        _fill(ticker="A", t=base + timedelta(hours=4)),   # 6h to settle (medium)
    ]
    outcomes = {"A": 1.0}
    settlement_times = {"A": settle}
    decay = edge_decay(fills, outcomes, settlement_times, bucket_hours=[1, 6, 24])
    # Both should have positive PnL since "yes" won
    buckets = dict(decay)
    assert 1 in buckets or 6 in buckets
