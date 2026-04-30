from __future__ import annotations

from src.portfolio.models import RiskDecision
from src.portfolio.sizer import QuarterKellySizer
from src.signals.models import CalibratedEdge


def _edge(side="yes", kelly=0.10, implied=0.50) -> CalibratedEdge:
    return CalibratedEdge(
        ticker="T", title="t", side=side,
        estimated_fair_prob=implied + 0.10,
        current_implied_prob=implied,
        edge_pp=0.10, confidence=0.7,
        kelly_fraction=kelly,
        thesis="", source_signal_model="test",
    )


def _ok():
    return RiskDecision(approved=True, reason="ok", scale_factor=1.0)


def _scaled(s):
    return RiskDecision(approved=True, reason="scaled", scale_factor=s)


def _rejected():
    return RiskDecision(approved=False, reason="bad")


def test_zero_when_rejected():
    s = QuarterKellySizer()
    assert s.size(_edge(), 1_000.0, _rejected(), 0.30) == 0


def test_zero_when_no_cash():
    s = QuarterKellySizer()
    assert s.size(_edge(), 0.0, _ok(), 0.30) == 0


def test_zero_when_kelly_zero():
    s = QuarterKellySizer()
    assert s.size(_edge(kelly=0.0), 1_000.0, _ok(), 0.30) == 0


def test_yes_side_sizing_with_full_kelly():
    s = QuarterKellySizer()
    # 10% of 1000 = 100 USD; price 0.30 → 333 contracts
    contracts = s.size(_edge(kelly=0.10), 1_000.0, _ok(), 0.30)
    assert contracts == int(100 // 0.30)


def test_no_side_uses_complement_price():
    s = QuarterKellySizer()
    # price = 0.30; no-side cost = 0.70 per contract; deploy 100 USD
    # → 100 / 0.70 = 142 contracts
    contracts = s.size(_edge(side="no", kelly=0.10), 1_000.0, _ok(), 0.30)
    assert contracts == int(100 // 0.70)


def test_scale_factor_reduces_size():
    s = QuarterKellySizer()
    full = s.size(_edge(kelly=0.10), 1_000.0, _ok(), 0.30)
    half = s.size(_edge(kelly=0.10), 1_000.0, _scaled(0.5), 0.30)
    # Half should be approx half of full (within rounding)
    assert half * 2 - full <= 1


def test_quarter_kelly_clamp():
    s = QuarterKellySizer()
    # Edge claims kelly=0.50 but sizer clamps to 0.25
    big = s.size(_edge(kelly=0.50), 1_000.0, _ok(), 0.30)
    cap = s.size(_edge(kelly=0.25), 1_000.0, _ok(), 0.30)
    assert big == cap


def test_max_contracts_per_order_clamp():
    s = QuarterKellySizer(max_contracts_per_order=100)
    # Lots of cash, low price, kelly 0.25 → would otherwise be huge
    contracts = s.size(_edge(kelly=0.25), 1_000_000.0, _ok(), 0.10)
    assert contracts == 100
