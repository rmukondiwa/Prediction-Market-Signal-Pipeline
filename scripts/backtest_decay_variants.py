"""
Test entry-logic variants on the existing HF crypto data without expanding
the universe.

Variants:
  1. baseline      — first qualifying entry, 1 per market
  2. multi_entry   — re-enter each time mid crosses threshold from below
                     (with cooldown to avoid same-tick double-fills)
  3. last_window   — enter only in the FINAL N minutes (e.g., last 5)
  4. trail_up      — enter at threshold T, optionally re-enter at T+epsilon
                     if implied keeps climbing

Compares trades/year, win%, Sharpe, and net annualized P&L for each.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from scripts.backtest_decay_hf import (
    SLIPPAGE, WITHDRAWAL_FEE_RATE, MIN_CANDLE_VOLUME,
    CONTRACTS_PER_TRADE, candle_quotes, trade_fee, parse_iso,
)

META_PATH = Path("data/hf_universe_meta.json")
CANDLES_PATH = Path("data/hf_candles.json")


def make_entry(meta, c, fill_price, side, mid, mins_to_close, won, settlement_value):
    gross = (1.0 if won else 0.0) - fill_price
    fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
    return {
        "ticker": meta["ticker"], "side": side,
        "fill_price": fill_price, "implied_at_entry": mid,
        "minutes_to_close": mins_to_close, "won": won,
        "pnl_per_contract": gross - fee, "fee_per_contract": fee,
        "category": meta.get("event_ticker", "").split("-")[0],
    }


def baseline(meta, candles, min_implied, max_minutes):
    """Original: one entry per market on first qualifying candle."""
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None: return []
    for c in candles:
        ts = c.get("end_period_ts")
        if not ts: continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        m2c = (close_t - t).total_seconds()/60
        if m2c <= 0 or m2c > max_minutes: continue
        bid, ask, vol = candle_quotes(c)
        if bid is None or vol < MIN_CANDLE_VOLUME: continue
        mid = (bid + ask) / 2
        if mid >= min_implied:
            fill = min(0.99, ask + SLIPPAGE)
            won = settlement_value == 1.0
            return [make_entry(meta, c, fill, "yes", mid, m2c, won, settlement_value)]
    return []


def multi_entry(meta, candles, min_implied, max_minutes, cooldown_min=2):
    """Multiple entries per market — re-enter each time mid crosses threshold
    from below after a cooldown."""
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None: return []
    out = []
    last_fill_t: datetime | None = None
    prev_mid = None
    for c in candles:
        ts = c.get("end_period_ts")
        if not ts: continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        m2c = (close_t - t).total_seconds()/60
        if m2c <= 0 or m2c > max_minutes: continue
        bid, ask, vol = candle_quotes(c)
        if bid is None or vol < MIN_CANDLE_VOLUME: continue
        mid = (bid + ask) / 2
        # Trigger: mid crossed from below threshold to above
        crossed_up = prev_mid is not None and prev_mid < min_implied <= mid
        first_signal = prev_mid is None and mid >= min_implied
        cooldown_ok = (last_fill_t is None
                       or (t - last_fill_t).total_seconds() >= cooldown_min*60)
        if (crossed_up or first_signal) and cooldown_ok:
            fill = min(0.99, ask + SLIPPAGE)
            won = settlement_value == 1.0
            out.append(make_entry(meta, c, fill, "yes", mid, m2c, won, settlement_value))
            last_fill_t = t
        prev_mid = mid
    return out


def last_window(meta, candles, min_implied, last_n_minutes):
    """Enter only in the FINAL N minutes."""
    return baseline(meta, candles, min_implied, last_n_minutes)


def trail_up(meta, candles, thresholds: list[float], max_minutes,
             include_no_side: bool = False):
    """Enter at each threshold step (e.g. [0.80, 0.90]) if mid hits it.
    Multiple entries possible, one per threshold step. Optionally mirrors
    on the NO side at (1 - thr) thresholds (buy NO when mid is very low)."""
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None: return []
    out = []
    hit_yes = set()
    hit_no = set()
    no_thresholds = [1 - t for t in thresholds] if include_no_side else []
    for c in candles:
        ts = c.get("end_period_ts")
        if not ts: continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        m2c = (close_t - t).total_seconds()/60
        if m2c <= 0 or m2c > max_minutes: continue
        bid, ask, vol = candle_quotes(c)
        if bid is None or vol < MIN_CANDLE_VOLUME: continue
        mid = (bid + ask) / 2
        for thr in thresholds:
            if thr in hit_yes: continue
            if mid >= thr:
                fill = min(0.99, ask + SLIPPAGE)
                won = settlement_value == 1.0
                out.append(make_entry(meta, c, fill, "yes", mid, m2c, won, settlement_value))
                hit_yes.add(thr)
        for nt in no_thresholds:
            if nt in hit_no: continue
            if mid <= nt:
                no_ask = 1.0 - bid
                fill = min(0.99, no_ask + SLIPPAGE)
                won = settlement_value == 0.0
                out.append(make_entry(meta, c, fill, "no", mid, m2c, won, settlement_value))
                hit_no.add(nt)
    return out


def aggregate(entries, sample_days=90.0, label=""):
    if not entries:
        return {"label": label, "n": 0}
    pnls = [e["pnl_per_contract"] for e in entries]
    costs = [e["fill_price"] for e in entries]
    rocs = [p/c if c > 0 else 0.0 for p, c in zip(pnls, costs)]
    wins = sum(1 for e in entries if e["won"])
    gross = sum(pnls) * CONTRACTS_PER_TRADE
    net = gross * (1 - WITHDRAWAL_FEE_RATE) if gross > 0 else gross
    annual_scale = 365 / sample_days
    sh = (mean(pnls)/stdev(pnls)*math.sqrt(len(pnls))) if len(pnls)>1 and stdev(pnls)>0 else 0.0
    return {
        "label": label,
        "n": len(entries),
        "annualized_trades": len(entries) * annual_scale,
        "win_rate": wins/len(entries),
        "mean_pnl_per_contract": mean(pnls),
        "mean_roc": mean(rocs),
        "sharpe": sh,
        "annualized_net_pnl": net * annual_scale,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-days", type=float, default=90.0)
    args = p.parse_args()

    meta_dict = {m["ticker"]: m for m in json.loads(META_PATH.read_text())}
    candles = json.loads(CANDLES_PATH.read_text())

    print(f"=== HF entry-logic variants (sample={args.sample_days}d) ===\n")
    print(f"{'variant':40} {'n':>4} {'ann_n':>6} {'win%':>5} {'roc':>6} {'sharpe':>6} {'ann_$':>9}")
    print("  " + "-"*82)

    def run_variant(name, fn):
        all_e = []
        for tk, m in meta_dict.items():
            cs = candles.get(tk, [])
            if not cs: continue
            all_e.extend(fn(m, cs))
        r = aggregate(all_e, args.sample_days, name)
        if r["n"] == 0:
            print(f"  {name:40} {0:>4} {'-':>6} {'-':>5} {'-':>6} {'-':>6} {'-':>9}")
            return
        print(f"  {name:40} {r['n']:>4} {r['annualized_trades']:>6.0f} "
              f"{r['win_rate']*100:>4.1f}% {r['mean_roc']*100:>+5.2f}% "
              f"{r['sharpe']:>+6.2f} ${r['annualized_net_pnl']:>+8.0f}")

    # Baseline
    run_variant("baseline (thr=0.80, max_min=15)",
                lambda m, c: baseline(m, c, 0.80, 15))

    # Multi-entry variants
    for cd in [1, 2, 3, 5]:
        run_variant(f"multi_entry (thr=0.80, cooldown={cd}m)",
                    lambda m, c, _cd=cd: multi_entry(m, c, 0.80, 15, _cd))

    # Last-window variants
    for n in [1, 2, 3, 5, 7, 10]:
        run_variant(f"last_window (thr=0.80, last={n}m)",
                    lambda m, c, _n=n: last_window(m, c, 0.80, _n))

    # Trail-up variants (yes-side only)
    run_variant("trail_up [0.80, 0.90]",
                lambda m, c: trail_up(m, c, [0.80, 0.90], 15))
    run_variant("trail_up [0.75, 0.85, 0.95]",
                lambda m, c: trail_up(m, c, [0.75, 0.85, 0.95], 15))
    run_variant("trail_up [0.70, 0.80, 0.90]",
                lambda m, c: trail_up(m, c, [0.70, 0.80, 0.90], 15))

    # Trail-up + NO-side mirror (the production candidate)
    run_variant("trail_up [0.80, 0.90] +NO",
                lambda m, c: trail_up(m, c, [0.80, 0.90], 15, include_no_side=True))
    run_variant("trail_up [0.75, 0.85, 0.95] +NO",
                lambda m, c: trail_up(m, c, [0.75, 0.85, 0.95], 15, include_no_side=True))
    run_variant("trail_up [0.70, 0.80, 0.90] +NO",
                lambda m, c: trail_up(m, c, [0.70, 0.80, 0.90], 15, include_no_side=True))


if __name__ == "__main__":
    main()
