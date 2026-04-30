"""
Backtest metrics.

Pure functions over fills + outcomes. No I/O, no global state — backtests
compute their metrics from the in-memory fill list at the end of each run.

Brier score is THE metric for this system. It measures calibration:
"are the probabilities our signal model emits actually right on average?"
A signal model that doesn't beat `market_baseline_brier` is not adding
information beyond what the market already prices.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from src.backtest.models import SimulatedFill


def brier_score(predicted_probs: list[float], outcomes: list[float]) -> float:
    """Mean squared error between predictions and binary outcomes (0 or 1)."""
    if not predicted_probs:
        return 0.0
    if len(predicted_probs) != len(outcomes):
        raise ValueError("predicted_probs and outcomes must be same length")
    return sum((p - o) ** 2 for p, o in zip(predicted_probs, outcomes)) / len(predicted_probs)


def market_baseline_brier(implied_probs: list[float], outcomes: list[float]) -> float:
    """Brier of the 'just trust the market price' baseline."""
    return brier_score(implied_probs, outcomes)


def hit_rate_by_confidence(
    fills: list[SimulatedFill],
    outcomes: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Bucket fills by confidence proxy (estimated_fair_prob distance from 0.5)
    and report hit rate + mean P&L per bucket."""
    def bucket(p: float) -> str:
        d = abs(p - 0.5)
        if d < 0.1:
            return "low"
        if d < 0.25:
            return "medium"
        return "high"

    buckets: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for f in fills:
        outcome = outcomes.get(f.ticker)
        if outcome is None:
            continue
        b = bucket(f.estimated_fair_prob)
        won = (f.side == "yes" and outcome == 1.0) or (f.side == "no" and outcome == 0.0)
        # P&L per contract: +1*(1-price) on win, -price on loss, minus fee
        if won:
            pnl = f.contracts * (1.0 - f.price) - f.fee
        else:
            pnl = -(f.contracts * f.price) - f.fee
        buckets[b].append((won, pnl))

    out: dict[str, dict[str, float]] = {}
    for b, results in buckets.items():
        wins = sum(1 for w, _ in results if w)
        total = len(results)
        total_pnl = sum(p for _, p in results)
        out[b] = {
            "trades": float(total),
            "wins": float(wins),
            "hit_rate": (wins / total) if total > 0 else 0.0,
            "pnl": total_pnl,
        }
    return out


def equity_curve(fills: list[SimulatedFill], outcomes: dict[str, float], starting_capital: float) -> list[float]:
    """Build the equity curve assuming everything resolves at settlement.
    Order: fills are applied chronologically; each fill's P&L resolves once
    its ticker shows up in `outcomes`."""
    settled: dict[str, list[SimulatedFill]] = defaultdict(list)
    for f in fills:
        settled[f.ticker].append(f)

    equity = starting_capital
    curve: list[float] = [equity]
    sorted_fills = sorted(fills, key=lambda f: f.timestamp)
    for f in sorted_fills:
        outcome = outcomes.get(f.ticker)
        if outcome is None:
            # No settlement yet → leave equity unchanged for now
            curve.append(equity)
            continue
        won = (f.side == "yes" and outcome == 1.0) or (f.side == "no" and outcome == 0.0)
        if won:
            pnl = f.contracts * (1.0 - f.price) - f.fee
        else:
            pnl = -(f.contracts * f.price) - f.fee
        equity += pnl
        curve.append(equity)
    return curve


def max_drawdown(equity: list[float]) -> float:
    """Return the largest peak-to-trough drawdown as a fraction (e.g. 0.12 = 12%)."""
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak <= 0:
            continue
        dd = (peak - v) / peak
        if dd > worst:
            worst = dd
    return worst


def sharpe(daily_returns: list[float], periods_per_year: int = 365) -> float | None:
    """Annualized Sharpe ratio; None if insufficient data or zero variance."""
    if len(daily_returns) < 2:
        return None
    n = len(daily_returns)
    mean = sum(daily_returns) / n
    variance = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    if variance <= 0:
        return None
    std = variance ** 0.5
    return (mean / std) * (periods_per_year ** 0.5)


def daily_returns_from_curve(equity: list[float]) -> list[float]:
    """Convert an equity curve into per-step return rates."""
    out: list[float] = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        if prev <= 0:
            continue
        out.append((equity[i] - prev) / prev)
    return out


def edge_decay(
    fills: list[SimulatedFill],
    outcomes: dict[str, float],
    settlement_times: dict[str, datetime],
    bucket_hours: list[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Group fills by hours-from-settlement and report mean P&L per bucket.

    Useful to answer "how stale does our signal get?" — if mean P&L drops as
    the bucket gets further from settlement, the signal works near the close
    but rots when held longer.
    """
    if bucket_hours is None:
        bucket_hours = [1, 6, 24, 72, 168, 720]

    buckets: dict[int, list[float]] = defaultdict(list)
    for f in fills:
        outcome = outcomes.get(f.ticker)
        settle_t = settlement_times.get(f.ticker)
        if outcome is None or settle_t is None:
            continue
        hours_to_settle = (settle_t - f.timestamp).total_seconds() / 3600
        # find smallest bucket that contains it
        b = next((h for h in bucket_hours if hours_to_settle <= h), bucket_hours[-1] + 1)
        won = (f.side == "yes" and outcome == 1.0) or (f.side == "no" and outcome == 0.0)
        if won:
            pnl = f.contracts * (1.0 - f.price) - f.fee
        else:
            pnl = -(f.contracts * f.price) - f.fee
        buckets[b].append(pnl)

    out: list[tuple[int, float]] = []
    for h in sorted(buckets.keys()):
        out.append((h, sum(buckets[h]) / len(buckets[h])))
    return out
