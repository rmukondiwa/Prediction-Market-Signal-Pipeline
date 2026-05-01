"""
Settlement-decay backtest on the high-frequency 15-min crypto universe.

Same execution model as backtest_decay.py (Kalshi fees, withdrawal cut,
volume gate, ask-side fills) but parameterized in MINUTES-to-close instead
of hours. Built for the 15-min crypto markets where the entire lifetime is
measured in minutes.

Why this matters for "more money at higher frequency": the daily-decay
strategy is capped at ~400 trades/year because that's the universe size of
liquid sports/weather markets that hit our threshold. The 15-min crypto
universe has 4000+/year, with the same favorite-longshot structure if it
holds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

META_PATH = Path("data/hf_universe_meta.json")
CANDLES_PATH = Path("data/hf_candles.json")

SLIPPAGE = 0.005
WITHDRAWAL_FEE_RATE = 0.10
MIN_CANDLE_VOLUME = 1
CONTRACTS_PER_TRADE = 100


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def candle_quotes(c: dict):
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
    p = max(0.001, min(0.999, price))
    return round(0.07 * contracts * p * (1 - p), 4)


def simulate_one(meta: dict, candles: list[dict],
                 min_implied: float, max_minutes: float,
                 no_max_implied: float | None = None) -> list[dict]:
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None:
        return []

    entries: list[dict] = []
    seen_signal = False
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        minutes_to_close = (close_t - t).total_seconds() / 60.0
        if minutes_to_close <= 0 or minutes_to_close > max_minutes:
            continue

        bid, ask, volume = candle_quotes(c)
        if bid is None or ask is None:
            continue
        if volume < MIN_CANDLE_VOLUME:
            continue
        mid = (bid + ask) / 2

        if mid >= min_implied and not seen_signal:
            fill_price = min(0.99, ask + SLIPPAGE)
            won = (settlement_value == 1.0)
            payoff = 1.0 if won else 0.0
            gross = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            entries.append({
                "ticker": meta["ticker"], "side": "yes",
                "fill_price": fill_price, "implied_at_entry": mid,
                "minutes_to_close": minutes_to_close,
                "won": won, "gross_pnl_per_contract": gross,
                "pnl_per_contract": gross - fee, "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
                "volume": volume,
            })
            seen_signal = True
            break

        if no_max_implied is not None and mid <= no_max_implied and not seen_signal:
            no_ask = 1.0 - bid
            fill_price = min(0.99, no_ask + SLIPPAGE)
            won = (settlement_value == 0.0)
            payoff = 1.0 if won else 0.0
            gross = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            entries.append({
                "ticker": meta["ticker"], "side": "no",
                "fill_price": fill_price, "implied_at_entry": mid,
                "minutes_to_close": minutes_to_close,
                "won": won, "gross_pnl_per_contract": gross,
                "pnl_per_contract": gross - fee, "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
                "volume": volume,
            })
            seen_signal = True
            break

    return entries


def bootstrap_mean_ci(values, n_boot=1000, alpha=0.05):
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot)]
    means.sort()
    return mean(values), means[int((alpha/2)*n_boot)], means[int((1-alpha/2)*n_boot)-1]


def equity_drawdown_sharpe(pnls):
    if not pnls:
        return 0.0, 0.0
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p * CONTRACTS_PER_TRADE)
    peak = eq[0]; mx = 0.0
    for v in eq:
        peak = max(peak, v)
        mx = max(mx, peak - v)
    sh = (mean(pnls)/stdev(pnls)*math.sqrt(len(pnls))) if len(pnls) > 1 and stdev(pnls) > 0 else 0.0
    return mx, sh


def run_one(meta_dict, candles, min_implied, max_minutes, include_no_side, sample_days):
    no_max = (1 - min_implied) if include_no_side else None
    entries = []
    for tk, m in meta_dict.items():
        cs = candles.get(tk, [])
        if cs:
            entries.extend(simulate_one(m, cs, min_implied, max_minutes, no_max))
    if not entries:
        return {"n": 0, "min_implied": min_implied, "max_minutes": max_minutes}

    pnls = [e["pnl_per_contract"] for e in entries]
    costs = [e["fill_price"] for e in entries]
    roc = [p/c if c > 0 else 0.0 for p, c in zip(pnls, costs)]
    wins = sum(1 for e in entries if e["won"])
    mean_pnl, lo, hi = bootstrap_mean_ci(pnls)
    mean_roc, _, _ = bootstrap_mean_ci(roc)
    mx_dd, sharpe = equity_drawdown_sharpe(pnls)

    # Annualize: project the sample's trade frequency to a year
    annual_scale = 365 / sample_days
    annual_trades = len(entries) * annual_scale

    # Fractional sizing for compound (2% per trade)
    SIZ = 0.02
    eq = 1.0
    for r in roc:
        eq *= (1 + SIZ * r)
    fc_return = eq - 1

    gross = sum(pnls) * CONTRACTS_PER_TRADE
    net = gross * (1 - WITHDRAWAL_FEE_RATE) if gross > 0 else gross
    annual_net = net * annual_scale

    avg_hold = mean([e["minutes_to_close"] for e in entries])  # minutes
    avg_concurrent = (len(entries) * avg_hold/60) / (sample_days * 24) if sample_days > 0 else 0
    work_cap = avg_concurrent * mean(costs) * CONTRACTS_PER_TRADE if entries else 0

    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e["pnl_per_contract"])

    return {
        "min_implied": min_implied,
        "max_minutes": max_minutes,
        "n": len(entries),
        "wins": wins,
        "win_rate": wins/len(entries),
        "mean_pnl_per_contract": mean_pnl, "ci_low": lo, "ci_high": hi,
        "mean_roc_per_trade": mean_roc,
        "fractional_compound_return": fc_return,
        "gross_total_pnl": gross,
        "net_total_pnl": net,
        "annualized_net_pnl": annual_net,
        "annualized_trades": annual_trades,
        "estimated_working_capital": work_cap,
        "max_drawdown_dollars": mx_dd,
        "sharpe": sharpe,
        "avg_hold_minutes": avg_hold,
        "by_category": {c: {"n": len(v), "win_rate": sum(1 for x in v if x>0)/len(v),
                            "mean_pnl": mean(v)} for c, v in by_cat.items() if len(v) >= 2},
    }


def fmt_pct(x: float, w: int = 7) -> str:
    return f"{x*100:>+{w-1}.2f}%"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--include-no-side", action="store_true")
    p.add_argument("--sample-days", type=float, default=90.0,
                   help="How many days the sample spans (for annualization)")
    p.add_argument("--output", type=Path, default=Path("reports/hf_decay_backtest.json"))
    return p.parse_args()


def main():
    args = parse_args()
    meta = {m["ticker"]: m for m in json.loads(META_PATH.read_text())}
    candles = json.loads(CANDLES_PATH.read_text())
    have = sum(1 for v in candles.values() if v)
    print(f"HF universe: {len(meta)} markets, {have} have candles, sample_days={args.sample_days}\n")

    if not args.sweep:
        r = run_one(meta, candles, 0.85, 5, args.include_no_side, args.sample_days)
        for k, v in r.items():
            if k == "by_category":
                continue
            print(f"  {k}: {v}")
        return

    print("=== HF sweep (minutes-to-close) ===")
    print(f"{'min_imp':>7} {'max_m':>5} {'n':>4} {'win%':>5} {'pnl/ct':>8} {'roc/tr':>7} "
          f"{'sharpe':>6} {'net_$':>8} {'ann_$':>9} {'ann_n':>6} {'work_cap':>9} {'dd_$':>6}")
    print("  " + "-"*100)
    sweep = []
    for thr in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        for mins in [1, 2, 3, 5, 7, 10, 15]:
            r = run_one(meta, candles, thr, float(mins), args.include_no_side, args.sample_days)
            sweep.append(r)
            if r["n"] == 0:
                print(f"  {thr:>5.2f} {mins:>5.0f} {0:>4} {'-':>5} {'-':>8} {'-':>7} "
                      f"{'-':>6} {'-':>8} {'-':>9} {'-':>6} {'-':>9} {'-':>6}")
            else:
                print(f"  {thr:>5.2f} {mins:>5.0f} {r['n']:>4} "
                      f"{r['win_rate']*100:>4.1f}% "
                      f"{r['mean_pnl_per_contract']:>+8.4f} "
                      f"{r['mean_roc_per_trade']*100:>+6.2f}% "
                      f"{r['sharpe']:>+6.2f} "
                      f"${r['net_total_pnl']:>+7.1f} "
                      f"${r['annualized_net_pnl']:>+8.0f} "
                      f"{r['annualized_trades']:>6.0f} "
                      f"${r['estimated_working_capital']:>8.0f} "
                      f"${r['max_drawdown_dollars']:>5.0f}")

    valid = [r for r in sweep if r.get("n", 0) >= 30 and r.get("net_total_pnl", 0) > 0]
    if valid:
        print("\n  Top 5 by ANNUALIZED NET P&L:")
        for r in sorted(valid, key=lambda x: -x["annualized_net_pnl"])[:5]:
            print(f"    min_imp={r['min_implied']:.2f}  max_min={r['max_minutes']:.0f}m  "
                  f"ann_pnl=${r['annualized_net_pnl']:.0f}  sharpe={r['sharpe']:+.2f}  "
                  f"win%={r['win_rate']*100:.1f}  trades/yr={r['annualized_trades']:.0f}  "
                  f"work_cap=${r['estimated_working_capital']:.0f}")
        print("\n  Top 5 by SHARPE:")
        for r in sorted(valid, key=lambda x: -x["sharpe"])[:5]:
            print(f"    min_imp={r['min_implied']:.2f}  max_min={r['max_minutes']:.0f}m  "
                  f"sharpe={r['sharpe']:+.2f}  ann_pnl=${r['annualized_net_pnl']:.0f}  "
                  f"win%={r['win_rate']*100:.1f}  trades/yr={r['annualized_trades']:.0f}")

    # strip raw entries from JSON output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sweep": sweep, "sample_days": args.sample_days}, indent=2))
    print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()
