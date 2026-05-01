"""
Backtest the SettlementDecaySignal strategy on historical Kalshi candles.

Reads:
    data/decay_universe_meta.json  — settled market metadata
    data/decay_candles.json        — hourly candles per market

For each market, walks hourly through its lifetime; at each hour where the
strategy's signal fires, simulates buying at ask + slippage and holds to
settlement. Aggregates win rate, mean P&L, Sharpe, and bootstrap CIs.

Sweeps (--sweep flag) over (min_implied_prob × max_hours_to_close) and
prints a parameter heatmap.

Honest caveats:
    - Survivorship-free: the universe is "all settled markets in the last
      365 days that had vol>=100 and lifetime>=24h" — we include all
      outcomes, not just winning ones.
    - Look-ahead-free: each entry decision uses ONLY the candle close at
      time t and the known close_time (which is set at market creation,
      not after-the-fact).
    - Fill assumption: midpoint + slippage, conservative on the take side.
      Real fills may be worse on illiquid markets.
    - Fees: Kalshi's actual fee schedule is per-contract trade-fee + a
      withdrawal cut on profits. We model the per-trade cost only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, stdev

META_PATH = Path("data/decay_universe_meta.json")
CANDLES_PATH = Path("data/decay_candles.json")

# Kalshi fee schedule (approx, retail tier as of 2025-2026):
#   - Trade fee: round(0.07 * contracts * yes_price * (1 - yes_price)) capped per
#     contract. Worst case ~$0.0175/contract at price 0.50; near zero at extremes.
#   - Withdrawal: ~10% take on net profits (Series B funding round changed this;
#     stay conservative).
# Modelled separately below so we can sanity-check sensitivity.
FEE_PER_CONTRACT_BASE = 0.02   # conservative average effective trade fee/ct
SLIPPAGE = 0.005                # 0.5 cent worse than ask on the take side
WITHDRAWAL_FEE_RATE = 0.10      # 10% on net profits at withdrawal
MIN_CANDLE_VOLUME = 1           # skip stale-quote candles with zero volume
CONTRACTS_PER_TRADE = 100       # nominal sizing for P&L scale


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def candle_quotes(c: dict) -> tuple[float | None, float | None, float]:
    """Return (yes_bid, yes_ask, volume) in dollars/contracts. None on bad data."""
    yb = c.get("yes_bid", {})
    ya = c.get("yes_ask", {})
    bid = yb.get("close_dollars")
    ask = ya.get("close_dollars")
    if bid is None or ask is None:
        return None, None, 0.0
    try:
        bid_f = float(bid)
        ask_f = float(ask)
        vol = float(c.get("volume_fp", 0) or 0)
    except (TypeError, ValueError):
        return None, None, 0.0
    if bid_f <= 0 and ask_f <= 0:
        return None, None, vol
    if bid_f > ask_f:
        return None, None, vol
    return bid_f, ask_f, vol


def trade_fee(price: float, contracts: int) -> float:
    """Kalshi-style per-trade fee: round(0.07 * contracts * price * (1-price)).
    Returns total fee in dollars (not per-contract)."""
    p = max(0.001, min(0.999, price))
    return round(0.07 * contracts * p * (1 - p), 4)


def simulate_one(
    meta: dict,
    candles: list[dict],
    min_implied: float,
    max_hours: float,
    no_max_implied: float | None = None,
) -> list[dict]:
    """Return all entries this strategy would have made on this market.
    Each entry dict has price, side, hours_to_close, won, pnl_per_contract."""
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()  # "yes" / "no" / ""
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None:
        return []

    entries: list[dict] = []
    seen_signal = False  # only ONE entry per market — first qualifying hour
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        hours_to_close = (close_t - t).total_seconds() / 3600.0
        if hours_to_close <= 0 or hours_to_close > max_hours:
            continue

        bid, ask, volume = candle_quotes(c)
        if bid is None or ask is None:
            continue
        # Volume gate: skip candles where nothing actually traded —
        # quotes there are stale and our "fill at ask" assumption is fiction.
        if volume < MIN_CANDLE_VOLUME:
            continue

        mid = (bid + ask) / 2

        # YES-side decay: enter at YES ask + slippage
        if mid >= min_implied and not seen_signal:
            fill_price = min(0.99, ask + SLIPPAGE)
            won = (settlement_value == 1.0)
            payoff = 1.0 if won else 0.0
            gross_pnl_per_ct = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            pnl = gross_pnl_per_ct - fee
            entries.append({
                "ticker": meta["ticker"], "side": "yes",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close,
                "outcome": settlement_value,
                "won": won,
                "gross_pnl_per_contract": gross_pnl_per_ct,
                "pnl_per_contract": pnl,
                "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
            })
            seen_signal = True
            break

        # NO-side decay: buy NO. Real no_ask ≈ 1 - yes_bid (best yes-sell price).
        if no_max_implied is not None and mid <= no_max_implied and not seen_signal:
            no_ask = 1.0 - bid  # honest NO ask from the bid-ask relationship
            fill_price = min(0.99, no_ask + SLIPPAGE)
            won = (settlement_value == 0.0)
            payoff = 1.0 if won else 0.0
            gross_pnl_per_ct = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            pnl = gross_pnl_per_ct - fee
            entries.append({
                "ticker": meta["ticker"], "side": "no",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close,
                "outcome": settlement_value,
                "won": won,
                "gross_pnl_per_contract": gross_pnl_per_ct,
                "pnl_per_contract": pnl,
                "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
            })
            seen_signal = True
            break

    return entries


def bootstrap_mean_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lower, upper) percentile CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return mean(values), means[lo_idx], means[hi_idx]


def equity_curve_drawdown(per_trade_pnl: list[float]) -> tuple[float, float]:
    """Return (max_drawdown_dollars, sharpe_unannualized).
    per_trade_pnl is per-contract P&L; we treat trades as independent + equal-sized."""
    if not per_trade_pnl:
        return 0.0, 0.0
    equity = [0.0]
    for p in per_trade_pnl:
        equity.append(equity[-1] + p * CONTRACTS_PER_TRADE)
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    if len(per_trade_pnl) > 1 and stdev(per_trade_pnl) > 0:
        sharpe = mean(per_trade_pnl) / stdev(per_trade_pnl) * math.sqrt(len(per_trade_pnl))
    else:
        sharpe = 0.0
    return max_dd, sharpe


def run_one_config(meta: dict, candles: dict, min_implied: float, max_hours: float,
                   include_no_side: bool = False) -> dict:
    """Run one parameter combo and return summary stats."""
    no_max = (1.0 - min_implied) if include_no_side else None
    all_entries: list[dict] = []
    for ticker, m in meta.items():
        cs = candles.get(ticker, [])
        if not cs:
            continue
        all_entries.extend(simulate_one(m, cs, min_implied, max_hours, no_max))

    if not all_entries:
        return {"n": 0, "min_implied": min_implied, "max_hours": max_hours}

    pnls = [e["pnl_per_contract"] for e in all_entries]
    costs = [e["fill_price"] for e in all_entries]
    # Return on capital per trade = pnl / cost (dollar in, dollar back out at settle)
    roc = [p / c if c > 0 else 0.0 for p, c in zip(pnls, costs)]

    wins = sum(1 for e in all_entries if e["won"])
    mean_pnl, lo, hi = bootstrap_mean_ci(pnls)
    mean_roc, roc_lo, roc_hi = bootstrap_mean_ci(roc)
    max_dd, sharpe = equity_curve_drawdown(pnls)

    # Compound with FRACTIONAL sizing: bet a small % of bankroll per trade.
    # Full-bankroll sizing bankrupts at the first loss (one trade resolving
    # against a 0.85 bet wipes 85% of capital). Kelly for 97% win-rate /
    # 12% payoff is roughly 2-3%; we use 2% as the test sizing.
    SIZING_FRACTION = 0.02
    eq = 1.0
    for r in roc:
        eq *= (1 + SIZING_FRACTION * r)
    fractional_compound_return = eq - 1

    # Total $ P&L net of withdrawal fee on net profits
    gross_total_pnl = sum(pnls) * CONTRACTS_PER_TRADE
    net_total_pnl = gross_total_pnl * (1 - WITHDRAWAL_FEE_RATE) if gross_total_pnl > 0 else gross_total_pnl
    total_capital = sum(costs) * CONTRACTS_PER_TRADE
    portfolio_roc = (net_total_pnl / total_capital) if total_capital > 0 else 0.0

    # Estimate max concurrent positions (capital lock-up estimate)
    # Sort entries by entry timestamp, count overlap with hours_to_close as
    # the holding period.
    entries_sorted = sorted(all_entries, key=lambda e: e.get("hours_to_close", 0), reverse=True)
    # We don't have actual entry timestamps in the entry dicts; approximate
    # by spreading uniformly over the year and counting concurrency given the
    # avg holding period.
    avg_hold_h = mean([e["hours_to_close"] for e in all_entries]) if all_entries else 0
    avg_concurrent = (len(all_entries) * avg_hold_h) / (365 * 24) if all_entries else 0
    estimated_working_capital = avg_concurrent * mean(costs) * CONTRACTS_PER_TRADE if all_entries else 0

    # By category breakdown
    by_cat: dict[str, list[float]] = defaultdict(list)
    for e in all_entries:
        by_cat[e["category"]].append(e["pnl_per_contract"])

    return {
        "min_implied": min_implied,
        "max_hours": max_hours,
        "n": len(all_entries),
        "wins": wins,
        "win_rate": wins / len(all_entries),
        "mean_pnl_per_contract": mean_pnl,
        "ci_low": lo, "ci_high": hi,
        "median_pnl": median(pnls),
        "mean_cost_per_contract": mean(costs),
        "mean_roc_per_trade": mean_roc,
        "roc_ci_low": roc_lo, "roc_ci_high": roc_hi,
        "fractional_compound_return": fractional_compound_return,  # 2% sizing
        "portfolio_roc_net": portfolio_roc,  # net of withdrawal fee
        "gross_total_pnl_dollars": gross_total_pnl,
        "net_total_pnl_dollars": net_total_pnl,
        "total_capital_deployed": total_capital,
        "avg_concurrent_positions": avg_concurrent,
        "estimated_working_capital": estimated_working_capital,
        "max_drawdown_dollars": max_dd,
        "sharpe": sharpe,
        "by_category": {cat: {"n": len(v), "win_rate": sum(1 for x in v if x>0)/len(v),
                              "mean_pnl": mean(v)} for cat, v in by_cat.items() if len(v) >= 2},
        "entries": all_entries,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-implied", type=float, default=0.95)
    p.add_argument("--max-hours", type=float, default=72.0)
    p.add_argument("--include-no-side", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over (min_implied × max_hours)")
    p.add_argument("--output", type=Path, default=Path("reports/decay_backtest.json"))
    return p.parse_args()


def fmt(x: float, w: int = 8) -> str:
    return f"{x:>{w}.4f}"


def print_one(r: dict) -> None:
    if r["n"] == 0:
        print(f"  No entries at min_implied={r['min_implied']}, max_hours={r['max_hours']}")
        return
    print(f"\n  Config: min_implied≥{r['min_implied']:.2f}  max_hours≤{r['max_hours']:.0f}h")
    print(f"  Entries: {r['n']}   Wins: {r['wins']}   Win rate: {r['win_rate']:.1%}")
    print(f"  Mean P&L per contract: {r['mean_pnl_per_contract']:+.4f}  "
          f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  Median P&L per contract: {r['median_pnl']:+.4f}")
    print(f"  Total P&L (assuming {CONTRACTS_PER_TRADE} contracts/trade): ${r['total_pnl_dollars']:+.2f}")
    print(f"  Max drawdown: ${r['max_drawdown_dollars']:.2f}   Sharpe (unannualized): {r['sharpe']:+.2f}")
    if r.get("by_category"):
        print(f"  By category (n≥2):")
        cats = sorted(r["by_category"].items(), key=lambda kv: -kv[1]["mean_pnl"])
        for cat, stats in cats[:10]:
            print(f"    {cat:25} n={stats['n']:>3}  win_rate={stats['win_rate']:.0%}  mean_pnl={stats['mean_pnl']:+.4f}")


def main() -> None:
    args = parse_args()
    meta = {m["ticker"]: m for m in json.loads(META_PATH.read_text())}
    candles = json.loads(CANDLES_PATH.read_text())
    print(f"Universe: {len(meta)} markets, {sum(1 for v in candles.values() if v)} have candle data\n")

    if not args.sweep:
        r = run_one_config(meta, candles, args.min_implied, args.max_hours, args.include_no_side)
        print_one(r)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Drop entries from disk write to keep file small; keep summary only
        out = {k: v for k, v in r.items() if k != "entries"}
        out["entry_count"] = len(r.get("entries", []))
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\n  Report: {args.output}")
        return

    print("=== Parameter sweep (sound execution: ask+slip+Kalshi fees+10% withdrawal+volume gate) ===")
    print(f"{'min_imp':>8} {'max_h':>5} {'n':>4} {'win%':>5} "
          f"{'pnl/ct':>8} {'roc/tr':>7} {'compd2%':>8} {'sharpe':>6} "
          f"{'net_pnl':>8} {'work_cap':>9} {'max_dd':>7}")
    print("  " + "-" * 95)
    sweep_results = []
    for thr in [0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.93, 0.95, 0.97]:
        for hrs in [12, 24, 48, 72, 168]:
            r = run_one_config(meta, candles, thr, float(hrs), args.include_no_side)
            sweep_results.append({k: v for k, v in r.items() if k != "entries"})
            if r["n"] == 0:
                print(f"  {thr:>6.2f} {hrs:>5.0f} {0:>4} {'-':>5} "
                      f"{'-':>8} {'-':>7} {'-':>9} {'-':>6} "
                      f"{'-':>8} {'-':>9} {'-':>7}")
            else:
                print(f"  {thr:>6.2f} {hrs:>5.0f} {r['n']:>4} "
                      f"{r['win_rate']*100:>4.1f}% "
                      f"{r['mean_pnl_per_contract']:>+8.4f} "
                      f"{r['mean_roc_per_trade']*100:>+6.2f}% "
                      f"{r['fractional_compound_return']*100:>+7.2f}% "
                      f"{r['sharpe']:>+6.2f} "
                      f"${r['net_total_pnl_dollars']:>+7.1f} "
                      f"${r['estimated_working_capital']:>8.0f} "
                      f"${r['max_drawdown_dollars']:>6.0f}")

    valid = [r for r in sweep_results if r.get("n", 0) >= 30 and r.get("net_total_pnl_dollars", 0) > 0]
    if valid:
        print("\n  Top 5 by SHARPE (n>=30, positive net P&L):")
        for r in sorted(valid, key=lambda x: -x["sharpe"])[:5]:
            print(f"    min_imp={r['min_implied']:.2f}  max_h={r['max_hours']:.0f}h  "
                  f"sharpe={r['sharpe']:+.2f}  net_pnl=${r['net_total_pnl_dollars']:+.1f}  "
                  f"work_cap=${r['estimated_working_capital']:.0f}  "
                  f"compd2%={r['fractional_compound_return']*100:+.2f}%")
        print("\n  Top 5 by ABSOLUTE NET $ P&L (Kalshi fees + 10% withdrawal):")
        for r in sorted(valid, key=lambda x: -x["net_total_pnl_dollars"])[:5]:
            print(f"    min_imp={r['min_implied']:.2f}  max_h={r['max_hours']:.0f}h  "
                  f"net_pnl=${r['net_total_pnl_dollars']:+.1f}  sharpe={r['sharpe']:+.2f}  "
                  f"work_cap=${r['estimated_working_capital']:.0f}  "
                  f"return_on_work_cap={r['net_total_pnl_dollars']/max(r['estimated_working_capital'],1)*100:.0f}%")
        print("\n  Top 5 by RETURN ON WORKING CAPITAL (annual, conservative):")
        for r in sorted(valid, key=lambda x: -(x["net_total_pnl_dollars"]/max(x["estimated_working_capital"],1)))[:5]:
            roc = r["net_total_pnl_dollars"] / max(r["estimated_working_capital"], 1) * 100
            print(f"    min_imp={r['min_implied']:.2f}  max_h={r['max_hours']:.0f}h  "
                  f"return_on_work_cap={roc:.0f}%  sharpe={r['sharpe']:+.2f}  "
                  f"net_pnl=${r['net_total_pnl_dollars']:+.1f}  work_cap=${r['estimated_working_capital']:.0f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sweep": sweep_results,
                                       "fee_per_contract": FEE_PER_CONTRACT,
                                       "slippage": SLIPPAGE,
                                       "contracts_per_trade": CONTRACTS_PER_TRADE}, indent=2))
    print(f"\n  Sweep written to {args.output}")


if __name__ == "__main__":
    main()
