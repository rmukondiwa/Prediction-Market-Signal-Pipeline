"""
Order reconciliation — compare expected fill vs actual fill.

Live trading diverges from paper:
  - Limit orders may fill partially or not at all
  - Market orders walk the book; effective price differs from quoted
  - Sometimes orders are rejected (insufficient balance, etc.)
  - Sometimes the venue confirms an order_id but the fill payload differs

This module:
  1. Captures `(expected, actual)` pairs at fill time
  2. Computes deltas (price, quantity, fee, latency)
  3. Surfaces divergences above thresholds for operator review
  4. Maintains a running "divergence score" used for circuit breakers

Use:
    rec = Reconciler()
    expected = ExpectedFill(ticker="X", side="yes", contracts=100, price=0.85)
    # ... place order, get back actual fill data ...
    actual = ActualFill(...)
    divergence = rec.observe(expected, actual)
    if divergence.severity == "high":
        # alert + possibly halt trading
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from src.utils.logging import get_logger

logger = get_logger(__name__)

Severity = Literal["normal", "moderate", "high"]


@dataclass
class ExpectedFill:
    ticker: str
    side: str
    contracts: int
    price: float       # dollars 0-1
    fee_estimate: float = 0.0
    placed_at: datetime | None = None


@dataclass
class ActualFill:
    ticker: str
    side: str
    contracts: int
    price: float       # average effective fill price
    fee: float = 0.0
    filled_at: datetime | None = None


@dataclass
class FillDivergence:
    ticker: str
    qty_delta_pct: float         # (actual - expected) / expected
    price_delta_dollars: float   # actual - expected
    fee_delta_dollars: float
    latency_seconds: float
    severity: Severity
    notes: list[str] = field(default_factory=list)


@dataclass
class ReconcilerConfig:
    qty_warn_pct: float = 0.05            # >5% qty delta is a yellow flag
    qty_high_pct: float = 0.20            # >20% qty delta is red
    price_warn_dollars: float = 0.005     # >½¢ slippage warning
    price_high_dollars: float = 0.02      # >2¢ slippage = red
    fee_warn_pct: float = 0.50            # >50% fee surprise = warning
    high_score_threshold: int = 5         # cumulative high+moderate count


class Reconciler:
    """Tracks divergences across many fills. Maintains a rolling score.

    When the score exceeds `cfg.high_score_threshold`, callers should
    consider halting trading — the live↔paper gap is too wide to trust.
    """

    def __init__(self, cfg: ReconcilerConfig | None = None):
        self.cfg = cfg or ReconcilerConfig()
        self._score = 0
        self._observations: list[FillDivergence] = []

    def observe(self, expected: ExpectedFill, actual: ActualFill) -> FillDivergence:
        cfg = self.cfg
        notes: list[str] = []

        if expected.contracts > 0:
            qty_delta_pct = (actual.contracts - expected.contracts) / expected.contracts
        else:
            qty_delta_pct = 0.0
        price_delta = actual.price - expected.price
        fee_delta = actual.fee - expected.fee_estimate

        if expected.placed_at and actual.filled_at:
            latency = (actual.filled_at - expected.placed_at).total_seconds()
        else:
            latency = 0.0

        severity: Severity = "normal"
        if abs(qty_delta_pct) >= cfg.qty_high_pct:
            severity = "high"
            notes.append(f"qty delta {qty_delta_pct*100:.1f}% ≥ high threshold")
        elif abs(qty_delta_pct) >= cfg.qty_warn_pct:
            severity = max_sev(severity, "moderate")
            notes.append(f"qty delta {qty_delta_pct*100:.1f}% ≥ warn threshold")

        if abs(price_delta) >= cfg.price_high_dollars:
            severity = "high"
            notes.append(f"price delta ${price_delta:+.4f} ≥ high threshold")
        elif abs(price_delta) >= cfg.price_warn_dollars:
            severity = max_sev(severity, "moderate")
            notes.append(f"price delta ${price_delta:+.4f} ≥ warn threshold")

        if expected.fee_estimate > 0 and abs(fee_delta) >= cfg.fee_warn_pct * expected.fee_estimate:
            severity = max_sev(severity, "moderate")
            notes.append(f"fee delta ${fee_delta:+.4f} ≥ {cfg.fee_warn_pct*100:.0f}%")

        d = FillDivergence(
            ticker=expected.ticker,
            qty_delta_pct=qty_delta_pct,
            price_delta_dollars=price_delta,
            fee_delta_dollars=fee_delta,
            latency_seconds=latency,
            severity=severity,
            notes=notes,
        )
        self._observations.append(d)
        if severity == "high":
            self._score += 2
        elif severity == "moderate":
            self._score += 1

        if severity != "normal":
            logger.warning("Fill divergence", extra={
                "ticker": d.ticker,
                "severity": severity,
                "qty_delta_pct": round(qty_delta_pct, 4),
                "price_delta": round(price_delta, 4),
                "notes": notes,
            })
        return d

    @property
    def divergence_score(self) -> int:
        return self._score

    def should_halt(self) -> bool:
        return self._score >= self.cfg.high_score_threshold

    def reset(self) -> None:
        self._score = 0
        self._observations.clear()


def max_sev(a: Severity, b: Severity) -> Severity:
    """Return the more severe of two severities."""
    order = {"normal": 0, "moderate": 1, "high": 2}
    return a if order[a] >= order[b] else b
