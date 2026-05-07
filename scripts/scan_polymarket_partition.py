"""
Polymarket within-platform partition scanner.

For each Polymarket EVENT (which can have N binary markets, e.g.
"who wins the primary?" → one market per candidate), pull live CLOB
top-of-book on each market and check the partition condition:

  sum(yes_top_ask across all markets in event) ≥ $1.00

If sum < $1, you can buy YES on every market for less than $1 and one
of them WILL settle to $1 — riskless arb (modulo gas).

If sum > $1.00 (much more common), the spread is normal MM behavior.

Threshold-only events (only one market) are skipped; only multi-market
events (≥2 candidates / outcomes) are analyzed.

Output: reports/polymarket_partition.json with `arb_candidates` for
sums < $1.

Usage:
    python -m scripts.scan_polymarket_partition --min-event-volume 10000
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.polymarket.clob_client import get_book

GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_events(limit: int, min_volume: float) -> list[dict]:
    """Pull active Polymarket events. Filter + sort client-side since the
    Gamma /events endpoint doesn't accept volume ordering."""
    out: list[dict] = []
    offset = 0
    pages_pulled = 0
    while len(out) < limit and pages_pulled < 30:
        url = f"{GAMMA}/events?active=true&closed=false&limit=100&offset={offset}"
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                page = json.loads(r.read())
        except Exception as e:
            print(f"events fetch failed: {e}")
            break
        pages_pulled += 1
        if not page:
            break
        for e in page:
            try:
                v = float(e.get("volume", 0) or 0)
            except (TypeError, ValueError):
                v = 0
            if v < min_volume:
                continue
            mkts = e.get("markets") or []
            # Filter to events with multiple un-closed markets
            active_mkts = [m for m in mkts if not m.get("closed") and not m.get("resolved")]
            if len(active_mkts) < 2:
                continue
            e["_active_markets"] = active_mkts
            e["_volume_num"] = v
            out.append(e)
        if len(page) < 100:
            break
        offset += 100
    # Sort by volume desc, take top N
    out.sort(key=lambda x: -x["_volume_num"])
    return out[:limit]


def first_token(market: dict) -> str | None:
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    return raw[0]  # YES token


def analyze_event(event: dict) -> dict:
    """Compute the partition stat for one event."""
    markets = event.get("markets") or []
    rows = []
    sum_yes_ask = 0.0
    sum_yes_bid = 0.0
    n_with_book = 0
    for m in markets:
        # Skip resolved markets
        if m.get("closed") or m.get("resolved"):
            continue
        token = first_token(m)
        if not token:
            continue
        try:
            book = get_book(token)
        except Exception:
            continue
        if not book.top_ask or not book.top_bid:
            continue
        ya = book.top_ask.price
        yb = book.top_bid.price
        sum_yes_ask += ya
        sum_yes_bid += yb
        n_with_book += 1
        rows.append({
            "market": m.get("question") or m.get("groupItemTitle", ""),
            "yes_bid": yb, "yes_ask": ya,
            "yes_bid_size": book.top_bid.size,
            "yes_ask_size": book.top_ask.size,
        })
    return {
        "event_id": event.get("id"),
        "event_slug": event.get("slug"),
        "event_title": event.get("title", "")[:120],
        "n_markets_total": len(markets),
        "n_markets_with_book": n_with_book,
        "sum_yes_ask": round(sum_yes_ask, 4),
        "sum_yes_bid": round(sum_yes_bid, 4),
        "spread": round(sum_yes_ask - sum_yes_bid, 4),
        "partition_arb_edge": round(1.0 - sum_yes_ask, 4),  # >0 means free arb
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--min-event-volume", type=float, default=10_000.0)
    p.add_argument("--output", type=Path, default=Path("reports/polymarket_partition.json"))
    args = p.parse_args()

    print(f"=== Polymarket within-platform partition scan ===")
    print(f"  Pulling top {args.limit} events by volume (min ${args.min_event_volume:,.0f})...")
    events = fetch_events(args.limit, args.min_event_volume)
    print(f"  Got {len(events)} multi-market events")

    results = []
    for i, e in enumerate(events, 1):
        res = analyze_event(e)
        results.append(res)
        if i % 5 == 0:
            print(f"    {i}/{len(events)} analyzed")

    # Sort by edge (positive = arb)
    results.sort(key=lambda r: -r["partition_arb_edge"])

    arbs = [r for r in results if r["partition_arb_edge"] > 0.0 and r["n_markets_with_book"] >= 2]
    near_arbs = [r for r in results if -0.05 < r["partition_arb_edge"] <= 0.0]

    print(f"\n=== RESULTS ===")
    print(f"  Events analyzed:           {len(results)}")
    print(f"  With ≥2 booked markets:    {sum(1 for r in results if r['n_markets_with_book'] >= 2)}")
    print(f"  PARTITION ARBS (sum<$1):   {len(arbs)}")
    print(f"  Near-arbs (sum within 5¢): {len(near_arbs)}")

    if arbs:
        print(f"\n  Top 5 partition arb candidates:")
        for r in arbs[:5]:
            print(f"    edge=${r['partition_arb_edge']:+.4f}  sum_yes_ask=${r['sum_yes_ask']:.4f}  "
                  f"n={r['n_markets_with_book']}  '{r['event_title'][:60]}'")

    print(f"\n  Tightest distributions (sum_yes_ask closest to $1):")
    sorted_by_tightness = sorted(results, key=lambda r: abs(r["partition_arb_edge"]))
    for r in sorted_by_tightness[:5]:
        print(f"    edge=${r['partition_arb_edge']:+.4f}  sum=${r['sum_yes_ask']:.4f}  "
              f"n={r['n_markets_with_book']}  '{r['event_title'][:60]}'")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_events": len(results),
        "arbs": arbs,
        "near_arbs": near_arbs,
        "all_results": results,
    }, indent=2, default=str))
    print(f"\n  → {args.output}")


if __name__ == "__main__":
    main()
