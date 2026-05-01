"""
Combined-universe settlement-decay backtest.

Loads ALL three universes (daily decay, 15M HF, expanded) and runs the
strategy uniformly in hours-to-close. Reports per-universe and combined
annualized P&L with sound execution model.

This is the "real number" for what the strategy looks like across our entire
known universe. Sample is the most recent 90 days for HF/expanded and
365 days for the original daily set — annualization handles the mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

# Reuse the audited execution constants
from scripts.backtest_decay import (
    SLIPPAGE, WITHDRAWAL_FEE_RATE, MIN_CANDLE_VOLUME,
    CONTRACTS_PER_TRADE, candle_quotes, trade_fee, parse_iso,
)

UNIVERSES = [
    {"name": "daily",    "meta": "data/decay_universe_meta.json",
     "candles": "data/decay_candles.json", "sample_days": 365.0},
    {"name": "hf_15m",   "meta": "data/hf_universe_meta.json",
     "candles": "data/hf_candles.json", "sample_days": 90.0},
    {"name": "expanded", "meta": "data/expanded_universe_meta.json",
     "candles": "data/expanded_candles.json", "sample_days": 90.0},
]


def simulate_market(meta: dict, candles: list[dict],
                    min_implied: float, max_hours: float,
                    no_max_implied: float | None) -> list[dict]:
    """Hours-based simulator (works for any timescale; max_hours=0.25 for
    15-min markets)."""
    close_t = parse_iso(meta["close_time"])
    result = (meta.get("result") or "").lower()
    settlement_value = 1.0 if result == "yes" else (0.0 if result == "no" else None)
    if settlement_value is None:
        return []

    entries = []
    seen = False
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None:
            continue
        t = datetime.fromtimestamp(ts, tz=timezone.utc)
        hours_to_close = (close_t - t).total_seconds() / 3600.0
        if hours_to_close <= 0 or hours_to_close > max_hours:
            continue
        bid, ask, vol = candle_quotes(c)
        if bid is None or ask is None:
            continue
        if vol < MIN_CANDLE_VOLUME:
            continue
        mid = (bid + ask) / 2

        if mid >= min_implied and not seen:
            fill_price = min(0.99, ask + SLIPPAGE)
            won = (settlement_value == 1.0)
            payoff = 1.0 if won else 0.0
            gross = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            entries.append({
                "ticker": meta["ticker"], "side": "yes",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close, "won": won,
                "pnl_per_contract": gross - fee, "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
            })
            seen = True
            break
        if no_max_implied is not None and mid <= no_max_implied and not seen:
            no_ask = 1.0 - bid
            fill_price = min(0.99, no_ask + SLIPPAGE)
            won = (settlement_value == 0.0)
            payoff = 1.0 if won else 0.0
            gross = payoff - fill_price
            fee = trade_fee(fill_price, CONTRACTS_PER_TRADE) / CONTRACTS_PER_TRADE
            entries.append({
                "ticker": meta["ticker"], "side": "no",
                "fill_price": fill_price, "implied_at_entry": mid,
                "hours_to_close": hours_to_close, "won": won,
                "pnl_per_contract": gross - fee, "fee_per_contract": fee,
                "category": meta.get("event_ticker", "").split("-")[0],
            })
            seen = True
            break
    return entries


def kelly(p: float, avg_fill: float) -> float:
    if avg_fill <= 0 or avg_fill >= 1:
        return 0.0
    return max(0.0, (p - avg_fill) / (1 - avg_fill))


def compound_at_size(roc_seq: list[float], size_fraction: float) -> tuple[float, float]:
    eq = 1.0; peak = 1.0; mx = 0.0
    for r in roc_seq:
        eq *= (1 + size_fraction * r)
        if eq <= 0:
            return 0.0, 1.0
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        mx = max(mx, dd)
    return eq, mx


def run_universe(name: str, meta_path: str, candles_path: str,
                 sample_days: float, min_implied: float, max_hours: float,
                 include_no: bool) -> dict:
    meta = {m["ticker"]: m for m in json.loads(Path(meta_path).read_text())}
    candles = json.loads(Path(candles_path).read_text())
    no_max = (1 - min_implied) if include_no else None

    entries = []
    for tk, m in meta.items():
        cs = candles.get(tk, [])
        if cs:
            entries.extend(simulate_market(m, cs, min_implied, max_hours, no_max))

    if not entries:
        return {"name": name, "n": 0, "annualized_net_pnl": 0.0,
                "win_rate": 0.0, "sharpe": 0.0}

    pnls = [e["pnl_per_contract"] for e in entries]
    costs = [e["fill_price"] for e in entries]
    rocs = [p/c if c > 0 else 0.0 for p, c in zip(pnls, costs)]
    wins = sum(1 for e in entries if e["won"])
    p_emp = wins / len(entries)
    avg_fill = mean(costs)

    # Kelly + sized compound at multiple fractions
    f_kelly = kelly(p_emp, avg_fill)
    sizing = {}
    for label, f in [("5%", 0.05), ("10%", 0.10),
                     ("Quarter-Kelly", f_kelly * 0.25),
                     ("Half-Kelly", f_kelly * 0.50),
                     ("Full-Kelly", f_kelly)]:
        if f <= 0:
            sizing[label] = {"f": 0, "mult": 1, "max_dd_pct": 0}
            continue
        m, dd = compound_at_size(rocs, f)
        sizing[label] = {"f": round(f, 4), "mult": round(m, 3),
                         "max_dd_pct": round(dd*100, 1)}

    annual_scale = 365 / sample_days
    gross = sum(pnls) * CONTRACTS_PER_TRADE
    net = gross * (1 - WITHDRAWAL_FEE_RATE) if gross > 0 else gross
    annual_net = net * annual_scale

    sharpe = (mean(pnls)/stdev(pnls)*math.sqrt(len(pnls))) if len(pnls) > 1 and stdev(pnls) > 0 else 0.0

    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e["pnl_per_contract"])

    return {
        "name": name,
        "n": len(entries),
        "win_rate": p_emp,
        "avg_fill": avg_fill,
        "full_kelly": f_kelly,
        "annualized_trades": len(entries) * annual_scale,
        "gross_total_pnl": gross,
        "net_total_pnl": net,
        "annualized_net_pnl": annual_net,
        "sharpe": sharpe,
        "sizing": sizing,
        "top_categories": sorted(
            [{"cat": c, "n": len(v), "win%": sum(1 for x in v if x>0)/len(v),
              "mean_pnl": mean(v)} for c, v in by_cat.items() if len(v) >= 5],
            key=lambda x: -x["n"]
        )[:10],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-implied", type=float, default=0.80)
    p.add_argument("--max-hours", type=float, default=12.0,
                   help="Entry window in hours (use 0.25 for 15-min markets)")
    p.add_argument("--per-universe-tuning", action="store_true",
                   help="Use universe-specific max_hours: daily=12h, 15M=0.25h, expanded=4h")
    p.add_argument("--include-no-side", action="store_true", default=True)
    p.add_argument("--output", type=Path, default=Path("reports/combined_backtest.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"=== Combined-universe settlement-decay backtest ===\n")

    results = []
    for u in UNIVERSES:
        if args.per_universe_tuning:
            mh = {"daily": 12.0, "hf_15m": 0.25, "expanded": 4.0}[u["name"]]
        else:
            mh = args.max_hours
        if not Path(u["meta"]).exists() or not Path(u["candles"]).exists():
            print(f"  Skipping {u['name']} — files missing")
            continue
        r = run_universe(u["name"], u["meta"], u["candles"], u["sample_days"],
                         args.min_implied, mh, args.include_no_side)
        r["max_hours_used"] = mh
        results.append(r)
        print(f"\n--- {u['name'].upper()} (max_hours={mh}h, sample_days={u['sample_days']}) ---")
        if r["n"] == 0:
            print(f"  No entries.")
            continue
        print(f"  N={r['n']} (annualized: {r['annualized_trades']:.0f}/yr)  "
              f"win%={r['win_rate']*100:.1f}  avg_fill={r['avg_fill']:.3f}")
        print(f"  full_kelly={r['full_kelly']*100:.1f}%  sharpe={r['sharpe']:+.2f}")
        print(f"  net_total=${r['net_total_pnl']:+.0f}  annualized=${r['annualized_net_pnl']:+.0f}")
        print(f"  Sizing → final_mult / max_dd:")
        for label, s in r["sizing"].items():
            print(f"    {label:>14}  f={s['f']*100:>5.1f}%  mult={s['mult']:>6}  max_dd={s['max_dd_pct']:>5.1f}%")

    combined_pnl = sum(r["annualized_net_pnl"] for r in results)
    combined_trades = sum(r["annualized_trades"] for r in results)
    print(f"\n=== COMBINED across all universes (annualized) ===")
    print(f"  Trades/year:   {combined_trades:.0f}")
    print(f"  Net P&L/year:  ${combined_pnl:+.0f}  (after Kalshi fees + 10% withdrawal)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_universe": results,
                                       "combined_annualized_pnl": combined_pnl,
                                       "combined_annualized_trades": combined_trades,
                                       "params": vars(args)}, indent=2, default=str))
    print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()
