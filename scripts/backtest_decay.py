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

FEE_PER_CONTRACT = 0.01  # placeholder; Kalshi's fee schedule is roughly $0.01-0.02
SLIPPAGE = 0.005          # 0.5 cent on the take side
CONTRACTS_PER_TRADE = 100  # nominal sizing for P&L scale


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def candle_mid_ask(c: dict) -> tuple[float | None, float | None]:
    """Return (mid, ask) in dollars. None if missing."""
    yb = c.get("yes_bid", {})
    ya = c.get("yes_ask", {})
    bid = yb.get("close_dollars")
    ask = ya.get("close_dollars")
    if bid is None or ask is None:
        return None, None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except (TypeError, ValueError):
        return None, None
    if bid_f <= 0 and ask_f <= 0:
        return None, None
    if bid_f > ask_f:  # malformed entry
        return None, None
    return (bid_f + ask_f) / 2, ask_f


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

        mid, ask = candle_mid_ask(c)
        if mid is None or ask is None:
            continue

        # YES-side decay
        if mid >= min_implied and not seen_signal:
            fill_price = min(0.99, ask + SLIPPAGE)
            won = (settlement_value == 1.0)
            payoff = 1.0 if won else 0.0
            pnl = (payoff - fill_price) - FEE_PER_CONTRACT
            entries.append({
                "ticker": meta["ticker"], "side": "yes",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close,
                "outcome": settlement_value,
                "won": won, "pnl_per_contract": pnl,
                "category": meta.get("event_ticker", "").split("-")[0],
            })
            seen_signal = True
            break  # one entry per market

        # NO-side decay (mirror)
        if no_max_implied is not None and mid <= no_max_implied and not seen_signal:
            no_ask = 1.0 - (mid - 0.005)  # approximate no-side ask
            fill_price = min(0.99, no_ask + SLIPPAGE)
            won = (settlement_value == 0.0)
            payoff = 1.0 if won else 0.0
            pnl = (payoff - fill_price) - FEE_PER_CONTRACT
            entries.append({
                "ticker": meta["ticker"], "side": "no",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close,
                "outcome": settlement_value,
                "won": won, "pnl_per_contract": pnl,
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
    wins = sum(1 for e in all_entries if e["won"])
    mean_pnl, lo, hi = bootstrap_mean_ci(pnls)
    max_dd, sharpe = equity_curve_drawdown(pnls)

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
        "total_pnl_dollars": sum(pnls) * CONTRACTS_PER_TRADE,
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

    print("=== Parameter sweep ===")
    print(f"{'min_imp':>8}  {'max_h':>6}  {'n':>4}  {'wins':>4}  {'win%':>6}  "
          f"{'mean_pnl':>9}  {'sharpe':>7}  {'max_dd$':>8}")
    print("  " + "-" * 76)
    sweep_results = []
    for thr in [0.85, 0.90, 0.93, 0.95, 0.97]:
        for hrs in [12, 24, 48, 72, 168]:
            r = run_one_config(meta, candles, thr, float(hrs), args.include_no_side)
            sweep_results.append({k: v for k, v in r.items() if k != "entries"})
            if r["n"] == 0:
                print(f"  {thr:>6.2f}  {hrs:>6.0f}  {0:>4}  {0:>4}  {'-':>6}  {'-':>9}  {'-':>7}  {'-':>8}")
            else:
                print(f"  {thr:>6.2f}  {hrs:>6.0f}  {r['n']:>4}  {r['wins']:>4}  "
                      f"{r['win_rate']*100:>5.1f}%  {r['mean_pnl_per_contract']:>+9.4f}  "
                      f"{r['sharpe']:>+7.2f}  {r['max_drawdown_dollars']:>8.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sweep": sweep_results,
                                       "fee_per_contract": FEE_PER_CONTRACT,
                                       "slippage": SLIPPAGE,
                                       "contracts_per_trade": CONTRACTS_PER_TRADE}, indent=2))
    print(f"\n  Sweep written to {args.output}")


if __name__ == "__main__":
    main()
