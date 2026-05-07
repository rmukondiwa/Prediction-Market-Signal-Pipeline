"""Tests for src/execution/reconciliation.py — fill divergence tracker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.execution.reconciliation import (
    ActualFill,
    ExpectedFill,
    Reconciler,
    ReconcilerConfig,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_clean_fill_is_normal():
    rec = Reconciler()
    expected = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85, fee_estimate=0.5)
    actual = ActualFill(ticker="X", side="yes", contracts=100, price=0.85, fee=0.5)
    d = rec.observe(expected, actual)
    assert d.severity == "normal"
    assert rec.divergence_score == 0


def test_qty_divergence_warn():
    rec = Reconciler()
    e = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85)
    a = ActualFill(ticker="X", side="yes", contracts=92, price=0.85)  # -8% qty
    d = rec.observe(e, a)
    assert d.severity == "moderate"
    assert rec.divergence_score == 1


def test_qty_divergence_high():
    rec = Reconciler()
    e = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85)
    a = ActualFill(ticker="X", side="yes", contracts=70, price=0.85)  # -30% qty
    d = rec.observe(e, a)
    assert d.severity == "high"
    assert rec.divergence_score == 2


def test_price_slippage_warn():
    rec = Reconciler()
    e = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85)
    a = ActualFill(ticker="X", side="yes", contracts=100, price=0.86)  # 1¢ slip
    d = rec.observe(e, a)
    assert d.severity == "moderate"


def test_price_slippage_high():
    rec = Reconciler()
    e = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85)
    a = ActualFill(ticker="X", side="yes", contracts=100, price=0.88)  # 3¢ slip
    d = rec.observe(e, a)
    assert d.severity == "high"


def test_should_halt_after_score_threshold():
    rec = Reconciler(ReconcilerConfig(high_score_threshold=3))
    for _ in range(2):
        rec.observe(
            ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85),
            ActualFill(ticker="X", side="yes", contracts=70, price=0.85),  # high
        )
    assert rec.should_halt() is True  # 2 high × 2 = 4 ≥ 3


def test_reset_clears_score():
    rec = Reconciler()
    rec.observe(
        ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85),
        ActualFill(ticker="X", side="yes", contracts=70, price=0.85),
    )
    assert rec.divergence_score > 0
    rec.reset()
    assert rec.divergence_score == 0


def test_latency_recorded():
    rec = Reconciler()
    placed = _now()
    filled = placed + timedelta(seconds=2.5)
    e = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85, placed_at=placed)
    a = ActualFill(ticker="X", side="yes", contracts=100, price=0.85, filled_at=filled)
    d = rec.observe(e, a)
    assert 2.4 < d.latency_seconds < 2.6
