"""
Generic risk gates extracted from the (now-archived) decay paper trader.

These are strategy-agnostic — they take a desired position, an existing
state snapshot, and a configuration, and return either an allowed size or
a rejection reason. Any trader (decay, arb, manual) can use them.

Layers:
  1. Per-fill bankroll cap        — single trade ≤ X% of cash
  2. Depth cap                    — single fill ≤ Y% of inside ask depth
  3. Per-market cumulative cap    — total notional in one ticker
  4. Per-asset cumulative cap     — total notional across same-underlying tickers
  5. Per-tier cumulative cap      — for trail-up strategies, smaller size on
                                    early thresholds
  6. Drawdown ramp                — progressive de-risking near daily limit
  7. Daily loss kill switch       — hard halt at threshold
  8. Drawdown trail               — halt at peak-to-trough threshold

Most callers just want `RiskGates.compute_size(...)` — that runs everything
in order and returns final contract count + a list of reasons.
"""
from __future__ import annotations

from dataclasses import dataclass


# Cumulative per-market cap fractions by trail-up tier. Tier 0 = first
# (earliest) threshold, tier 2 = terminal. Back-loading bounds the
# "reversal-after-early-fill" loss pattern that ate the decay strategy.
TIER_CAP_FRACTIONS = (0.30, 0.60, 1.00)


@dataclass
class RiskConfig:
    max_fraction_per_fill: float = 0.05
    depth_take_fraction: float = 0.25
    max_contracts_per_order: int = 5_000
    max_per_market_usd: float = 500.0
    max_per_asset_usd: float = 1_000.0
    daily_loss_limit_usd: float | None = 300.0
    drawdown_limit_usd: float | None = 200.0


@dataclass
class RiskState:
    """Snapshot of the state that risk gates need to evaluate a fill."""
    cash: float
    realized_pnl: float
    peak_realized_pnl: float
    existing_market_notional: float = 0.0  # cumulative on this ticker
    existing_asset_notional: float = 0.0   # cumulative on this underlying


def drawdown_size_multiplier(realized_pnl: float,
                              daily_loss_limit_usd: float | None) -> float:
    """Step-function de-risking based on % of daily loss budget consumed.

    Returns 1.00 if flat or up.
    Returns 1.00 while drawdown ≤ 33% of limit (no derisk yet).
    Returns 0.50 while drawdown is 33-66% of limit.
    Returns 0.25 while drawdown is 66-99% of limit.
    Returns 0.00 at 100%+ (hard halt).
    """
    if daily_loss_limit_usd is None or realized_pnl >= 0:
        return 1.0
    pct_used = -realized_pnl / abs(daily_loss_limit_usd)
    if pct_used >= 1.0:
        return 0.0
    if pct_used >= 0.66:
        return 0.25
    if pct_used >= 0.33:
        return 0.50
    return 1.0


def kill_switch_state(state: RiskState, cfg: RiskConfig) -> tuple[bool, str | None]:
    """Returns (should_halt, reason_or_None)."""
    if cfg.daily_loss_limit_usd is not None and state.realized_pnl <= -abs(cfg.daily_loss_limit_usd):
        return True, "daily_loss_limit_exceeded"
    if cfg.drawdown_limit_usd is not None:
        drawdown = state.peak_realized_pnl - state.realized_pnl
        if drawdown >= abs(cfg.drawdown_limit_usd):
            return True, "drawdown_limit_exceeded"
    return False, None


def compute_size(ask_price: float, ask_size: float,
                 state: RiskState, cfg: RiskConfig,
                 threshold_tier: int = 2) -> tuple[int, dict]:
    """Run all risk gates and return (contracts, debug_info).

    debug_info is a dict naming each binding constraint and the value it
    bound at. Useful for logging why a fill ended up smaller than asked.

    Strategy-agnostic. The decay-specific tier system is here too because
    most binary-event strategies benefit from progressively-confident sizing.
    """
    if ask_price <= 0:
        return 0, {"reason": "ask_price <= 0"}

    halted, halt_reason = kill_switch_state(state, cfg)
    if halted:
        return 0, {"reason": halt_reason, "halted": True}

    multiplier = drawdown_size_multiplier(state.realized_pnl, cfg.daily_loss_limit_usd)
    if multiplier <= 0:
        return 0, {"reason": "drawdown_multiplier_zero", "multiplier": 0.0}

    by_bankroll = (state.cash * cfg.max_fraction_per_fill) / ask_price
    by_depth = ask_size * cfg.depth_take_fraction

    tier_idx = max(0, min(threshold_tier, len(TIER_CAP_FRACTIONS) - 1))
    tier_cap_usd = cfg.max_per_market_usd * TIER_CAP_FRACTIONS[tier_idx]
    remaining_market = max(0.0, tier_cap_usd - state.existing_market_notional)
    remaining_asset = max(0.0, cfg.max_per_asset_usd - state.existing_asset_notional)
    by_market = remaining_market / ask_price
    by_asset = remaining_asset / ask_price

    raw = min(by_bankroll, by_depth, by_market, by_asset, cfg.max_contracts_per_order)
    contracts = int(raw * multiplier)

    info = {
        "ask_price": ask_price,
        "by_bankroll": by_bankroll, "by_depth": by_depth,
        "by_market": by_market, "by_asset": by_asset,
        "by_tier_cap": tier_cap_usd, "tier_idx": tier_idx,
        "drawdown_multiplier": multiplier,
        "raw_min": raw, "final_contracts": contracts,
    }
    # Tag the binding constraint
    binding = min(by_bankroll, by_depth, by_market, by_asset)
    if binding == by_bankroll: info["binding"] = "bankroll"
    elif binding == by_depth: info["binding"] = "depth"
    elif binding == by_market: info["binding"] = "per_market_tier"
    elif binding == by_asset: info["binding"] = "per_asset"
    else: info["binding"] = "other"
    return max(0, contracts), info


def underlying_of(ticker: str) -> str:
    """Extract the underlying-asset key from a Kalshi ticker.

    KXBTC15M-26MAY051645-45 → 'KXBTC15M'
    KXBTCD-26MAR0704-T2169.99 → 'KXBTCD'
    """
    return ticker.split("-", 1)[0]
