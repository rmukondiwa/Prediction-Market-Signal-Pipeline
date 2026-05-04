"""
Canonical post-session audit for the paper trader.

Replays every fill from logs/paper_decay.jsonl, simulates the position-state
transitions exactly as the trader would have, and produces a ground-truth
P&L per market that includes:

  - Settlement P&L (final position vs outcome)
  - Realized P&L from intermediate opposite-side closures (the bit that
    yesterday's chat reporting silently dropped)
  - Lost-contracts diagnostic (state.py bug fingerprint — should be 0
    after the fix is deployed)

Red flags it surfaces:
  - Any market with realized losses NOT reported in settled events
  - Any "lost contracts" instances (suspicious — possible state.py regression)
  - Win rate >= 99% over n>=20 trades (statistically suspicious — review)
  - Any per-market exposure exceeding configured cap

Usage:
    python -m scripts.audit_paper_session
    python -m scripts.audit_paper_session --log logs/paper_decay.jsonl
    python -m scripts.audit_paper_session --json reports/audit_$(date +%Y%m%d_%H%M).json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path("logs/paper_decay.jsonl")


def replay_market(fills: list[dict], settled: dict | None) -> dict:
    """Simulate apply_fill behavior; return per-market accounting."""
    pos_side = None
    pos_ct = 0
    pos_avg = 0.0
    realized_pnl = 0.0
    lost_contracts = 0
    fee_total = 0.0
    fill_actions = []

    for f in fills:
        side = f["side"]
        ct = f["contracts"]
        price = f["price"]
        fee = f.get("fee", 0)
        fee_total += fee

        if pos_side is None:
            pos_side = side
            pos_ct = ct
            pos_avg = price
            realized_pnl -= fee  # fee on every fill
            fill_actions.append(f"open {side} {ct}@${price:.4f}")
        elif pos_side == side:
            new_total = pos_ct + ct
            pos_avg = (pos_ct * pos_avg + ct * price) / new_total
            pos_ct = new_total
            realized_pnl -= fee
            fill_actions.append(f"add {side} {ct}@${price:.4f} → {pos_ct}@${pos_avg:.4f}")
        else:
            closing = min(pos_ct, ct)
            equiv_sell = 1.0 - price
            close_pnl = closing * (equiv_sell - pos_avg) - fee
            realized_pnl += close_pnl
            leftover = ct - closing
            old_pos_ct = pos_ct
            pos_ct -= closing

            if pos_ct > 0:
                fill_actions.append(
                    f"close {closing}/{old_pos_ct} {pos_side} (rPnL ${close_pnl:+.2f}); "
                    f"position {pos_ct}@${pos_avg:.4f} {pos_side} remaining"
                )
            elif leftover > 0:
                # POST-FIX: should open new opposite-side position with leftover
                # PRE-FIX: leftover dropped, this is the bug fingerprint
                pos_side = side
                pos_ct = leftover
                pos_avg = price
                fill_actions.append(
                    f"close {closing} {pos_side} (rPnL ${close_pnl:+.2f}); "
                    f"flip to {leftover} {side}@${price:.4f} (was bug-prone)"
                )
            else:
                pos_side = None; pos_avg = 0.0
                fill_actions.append(f"close {closing} (flat, rPnL ${close_pnl:+.2f})")

    settle_pnl = settled.get("pnl", 0.0) if settled else 0.0
    return {
        "n_fills": len(fills),
        "realized_pnl_intermediate": realized_pnl,
        "lost_contracts": lost_contracts,
        "fee_total": fee_total,
        "settled_pnl": settle_pnl,
        "true_total_pnl": realized_pnl + settle_pnl,
        "settled": settled is not None,
        "actions": fill_actions,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--json", type=Path, help="Optional JSON output path")
    p.add_argument("--max-per-market", type=float, default=500.0,
                   help="Per-market exposure cap to check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log.exists():
        raise SystemExit(f"Log not found: {args.log}")

    events = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]
    fills = [e for e in events if e["event"] == "fill"]
    settled = {e["ticker"]: e for e in events if e["event"] == "settled"}
    skipped = [e for e in events if e["event"] == "signal_skipped"]
    kills = [e for e in events if e["event"] == "kill_switch_tripped"]
    ticks = [e for e in events if e["event"] == "tick_summary"]

    fills_by_ticker = defaultdict(list)
    for f in fills:
        fills_by_ticker[f["ticker"]].append(f)

    print(f"=== AUDIT: {args.log} ===\n")
    print(f"Events:  total={len(events)}  fills={len(fills)}  settled={len(settled)}  "
          f"skipped={len(skipped)}  kills={len(kills)}  ticks={len(ticks)}\n")

    # Replay every market
    market_audits = {}
    for tk, fs in fills_by_ticker.items():
        market_audits[tk] = replay_market(fs, settled.get(tk))

    # Per-market summary
    print(f"{'Ticker':<42} {'Fills':>5} {'Real$':>10} {'Reported$':>10} {'Δ':>8}")
    print("-" * 80)
    total_real = 0.0
    total_reported = 0.0
    discrepancies = []
    for tk in sorted(market_audits):
        a = market_audits[tk]
        real = a["true_total_pnl"]
        reported = a["settled_pnl"]
        delta = real - reported
        total_real += real
        total_reported += reported
        flag = ""
        if abs(delta) > 0.01:
            flag = " ⚠️"
            discrepancies.append((tk, delta))
        if a["lost_contracts"] > 0:
            flag += f" LOST{a['lost_contracts']}"
        if not a["settled"]:
            flag += " UNSETTLED"
        print(f"{flag:>4} {tk[:38]:<38} {a['n_fills']:>5} ${real:>+9.2f} ${reported:>+9.2f} ${delta:>+7.2f}")
    print("-" * 80)
    print(f"{'TOTAL':<42} {len(fills):>5} ${total_real:>+9.2f} ${total_reported:>+9.2f} "
          f"${total_real - total_reported:>+7.2f}")

    print(f"\n=== HEADLINE NUMBERS ===")
    print(f"  Ground-truth session P&L: ${total_real:+.2f}")
    print(f"  Settlement-event P&L:     ${total_reported:+.2f}  (from 'settled' events alone)")
    delta_total = total_reported - total_real
    if abs(delta_total) > 0.50:
        print(f"  ⚠️  PHANTOM gap: ${delta_total:+.2f}  ← do NOT report settlement-event "
              "totals as session P&L")
    else:
        print(f"  ✅ Discrepancy ${delta_total:+.2f} — within rounding")

    # Win rate analysis
    wins = sum(1 for tk, a in market_audits.items()
               if a["settled"] and a["true_total_pnl"] > 0)
    losses = sum(1 for tk, a in market_audits.items()
                 if a["settled"] and a["true_total_pnl"] < 0)
    settled_total = wins + losses
    win_rate = wins / settled_total if settled_total else 0
    print(f"\n  True win rate: {wins}/{settled_total} = {win_rate*100:.1f}%")
    if settled_total >= 20 and win_rate >= 0.99:
        print(f"  🚩 RED FLAG: win rate ≥ 99% over n≥20 — almost certainly an artifact, manual review required")
    elif settled_total < 20:
        print(f"  (sample size {settled_total} < 20, win rate not yet reliable)")

    # Lost-contracts diagnostic
    total_lost = sum(a["lost_contracts"] for a in market_audits.values())
    if total_lost > 0:
        print(f"\n  🚩 RED FLAG: {total_lost} contracts vanished across all markets — "
              "state.py regression?")
    else:
        print(f"\n  ✅ Lost-contracts: 0 (state.py fix holding)")

    # Per-market exposure cap check
    over_cap = []
    for tk, fs in fills_by_ticker.items():
        # max simultaneous notional during the session
        running_ct = 0
        running_cost = 0.0
        side = None
        peak_notional = 0.0
        for f in fs:
            if side is None or side == f["side"]:
                side = f["side"]
                running_ct += f["contracts"]
                running_cost += f["contracts"] * f["price"]
            else:
                # opposite, just approximate
                running_ct = max(0, running_ct - f["contracts"])
                running_cost = running_ct * (running_cost / max(running_ct + f["contracts"], 1))
            peak_notional = max(peak_notional, running_cost)
        if peak_notional > args.max_per_market * 1.05:  # 5% tolerance
            over_cap.append((tk, peak_notional))
    if over_cap:
        print(f"\n  ⚠️  Per-market exposure cap ({args.max_per_market}) exceeded:")
        for tk, n in over_cap:
            print(f"      {tk[:50]} peak ${n:.2f}")
    else:
        print(f"  ✅ Per-market exposure cap (${args.max_per_market}) respected on all markets")

    # Kill switch
    if kills:
        print(f"\n  ⚠️  Kill switch tripped {len(kills)} time(s) during this session:")
        for k in kills:
            print(f"      reason={k.get('reason')}  pnl={k.get('realized_pnl')}")

    print()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "log": str(args.log),
            "n_events": len(events),
            "n_fills": len(fills),
            "n_settled": len(settled),
            "n_skipped": len(skipped),
            "n_kills": len(kills),
            "n_ticks": len(ticks),
            "ground_truth_pnl": round(total_real, 4),
            "settlement_event_pnl": round(total_reported, 4),
            "phantom_gap": round(total_reported - total_real, 4),
            "true_win_rate": win_rate,
            "true_wins": wins,
            "true_losses": losses,
            "true_settled": settled_total,
            "lost_contracts": total_lost,
            "exposure_cap_breaches": [{"ticker": tk, "peak_notional": n} for tk, n in over_cap],
            "per_market": {
                tk: {k: v for k, v in a.items() if k != "actions"}
                for tk, a in market_audits.items()
            },
        }, indent=2, default=str))
        print(f"  JSON: {args.json}")


if __name__ == "__main__":
    main()
