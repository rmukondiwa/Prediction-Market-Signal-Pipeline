"""
Cross-platform arbitrage scanner: Kalshi vs Polymarket.

Pulls active markets from both platforms, matches them by underlying event
(BTC/ETH price by date, currently — easy to extend to elections, weather,
politics), and reports cases where the combined YES + complementary side
costs less than $1.00 (which would lock in risk-free profit).

The arb math:
    - Buy YES on Kalshi at K_yes_ask
    - Buy NO on Polymarket at (1 - P_yes_bid)
    - If K_yes_ask + (1 - P_yes_bid) < 1.00, we paid <$1 for guaranteed $1 payout.

Or symmetrically:
    - Buy NO on Kalshi at (1 - K_yes_bid)
    - Buy YES on Polymarket at P_yes_ask
    - If (1 - K_yes_bid) + P_yes_ask < 1.00, same arb in the other direction.

NOT executed automatically — output is a candidate list to be reviewed and
manually traded (until we build full execution on both platforms).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA = "https://gamma-api.polymarket.com"

UA_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Match these keywords in market questions to identify BTC/ETH price markets
BTC_PATTERNS = [r"\bbtc\b", r"\bbitcoin\b"]
ETH_PATTERNS = [r"\beth\b", r"\bethereum\b"]
PRICE_PATTERN = re.compile(r"\$?(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:k|K)?")
DATE_PATTERN = re.compile(r"(?:by|on|before)\s+([A-Z][a-z]+ \d+|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d+,? \d{4})", re.IGNORECASE)


def fetch_polymarket(limit: int = 500, min_volume: float = 1000.0) -> list[dict]:
    """Pull active, non-closed Polymarket markets sorted by volume."""
    out = []
    offset = 0
    while len(out) < limit:
        url = f"{POLY_GAMMA}/markets?limit=100&offset={offset}&active=true&closed=false&order=volumeNum&ascending=false"
        req = urllib.request.Request(url, headers=UA_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                page = json.loads(r.read())
        except Exception as e:
            print(f"poly fetch failed: {e}")
            break
        if not page:
            break
        for m in page:
            try:
                vol = float(m.get("volumeNum", 0) or 0)
            except (ValueError, TypeError):
                vol = 0
            if vol < min_volume:
                continue
            out.append(m)
        if len(page) < 100:
            break
        offset += 100
    return out


def fetch_kalshi_open(series_filter: set[str] | None = None,
                     pages: int = 25) -> list[dict]:
    """Pull open Kalshi markets in target series via per-series events query
    (much faster than paginating the entire 15k+ open universe)."""
    out = []
    if series_filter:
        for series in series_filter:
            url = f"{KALSHI}/events?series_ticker={series}&status=open&limit=200"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    data = json.loads(r.read())
            except Exception as e:
                continue
            for ev in data.get("events", []):
                # Inline markets sometimes present
                for m in (ev.get("markets") or []):
                    if m.get("status") in {"open", "active"}:
                        out.append(m)
                # Otherwise fetch event detail
                if not ev.get("markets"):
                    et = ev.get("event_ticker", "")
                    try:
                        u2 = f"{KALSHI}/events/{et}"
                        with urllib.request.urlopen(u2, timeout=10) as r2:
                            ed = json.loads(r2.read())
                        for m in (ed.get("markets") or []):
                            if m.get("status") in {"open", "active"}:
                                out.append(m)
                    except Exception:
                        pass
    else:
        cursor = None
        for _ in range(pages):
            url = f"{KALSHI}/markets?status=open&limit=1000"
            if cursor:
                url += f"&cursor={cursor}"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    data = json.loads(r.read())
            except Exception as e:
                break
            out.extend(data.get("markets", []))
            cursor = data.get("cursor", "")
            if not cursor:
                break
    return out


_BAD_NUMBERS = {2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030,
                2031, 2032}


def extract_price_threshold(text: str) -> float | None:
    """Pull the dollar threshold out of a price-prediction question.
    Returns None unless the number is clearly a price (preceded by $ or
    explicit threshold language like 'above'/'reach' AND not a year)."""
    if not text:
        return None
    t = text.lower()
    # Strict: must have a dollar sign or threshold word adjacent to the number
    strict = re.compile(
        r"(?:\$|above\s+|over\s+|reach(?:es)?\s+|hit(?:s)?\s+|>=?\s*|at\s+least\s+)"
        r"\$?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(k)?",
        re.IGNORECASE,
    )
    matches = strict.findall(t)
    for raw, k in matches:
        s = raw.replace(",", "")
        try:
            v = float(s)
        except ValueError:
            continue
        if k:
            v *= 1000
        if int(v) in _BAD_NUMBERS:
            continue
        if 100 <= v <= 1_000_000:
            return v
    return None


def is_price_question(text: str) -> bool:
    """Filter for actual 'X price by date' questions (not 'sells', 'mentions',
    'announces', 'tweets', etc.)."""
    if not text:
        return False
    t = text.lower()
    # Must mention price/value AND a number
    if not any(w in t for w in ["price", "above", "below", "reach", "hit", ">=", "<=", "or above", "or below"]):
        return False
    # Filter out derivative-event questions
    if any(w in t for w in ["sells", "buys", "announces", "tweets", "mentions",
                            "approves", "files", "indicted", "ban"]):
        return False
    return True


def extract_close_date(text: str) -> str | None:
    if not text:
        return None
    m = DATE_PATTERN.search(text)
    return m.group(1) if m else None


def find_btc_arbs(poly_markets: list[dict], kalshi_markets: list[dict]) -> list[dict]:
    """Match BTC price markets across the two platforms and report arb candidates."""
    # Tag Polymarket BTC markets
    poly_btc = []
    for m in poly_markets:
        q = (m.get("question") or "").lower()
        if not any(re.search(p, q) for p in BTC_PATTERNS):
            continue
        if not is_price_question(q):
            continue
        threshold = extract_price_threshold(q)
        if threshold is None:
            continue
        try:
            prices = json.loads(m.get("outcomePrices", "[]"))
            if len(prices) != 2:
                continue
            yes_price = float(prices[0])
        except (ValueError, json.JSONDecodeError):
            continue
        end = m.get("endDate", "")
        poly_btc.append({
            "platform": "poly", "question": m.get("question", ""),
            "threshold": threshold, "yes_price": yes_price,
            "no_price": 1 - yes_price, "end_date": end,
            "volume": m.get("volumeNum", 0),
            "id": m.get("id"), "slug": m.get("slug"),
        })

    # Tag Kalshi BTC markets
    kalshi_btc = []
    for m in kalshi_markets:
        title = (m.get("title") or "") + " " + (m.get("subtitle") or "")
        title_l = title.lower()
        if not any(re.search(p, title_l) for p in BTC_PATTERNS):
            continue
        if not is_price_question(title):
            continue
        threshold = extract_price_threshold(title)
        if threshold is None:
            continue
        try:
            yes_bid = float(m.get("yes_bid_dollars", 0) or 0)
            yes_ask = float(m.get("yes_ask_dollars", 0) or 0)
        except (ValueError, TypeError):
            continue
        if yes_bid <= 0 or yes_ask <= 0:
            continue
        kalshi_btc.append({
            "platform": "kalshi", "question": title.strip(),
            "threshold": threshold,
            "yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid": 1 - yes_ask, "no_ask": 1 - yes_bid,
            "close_time": m.get("close_time", ""),
            "ticker": m.get("ticker"),
        })

    print(f"  Poly BTC markets w/ extracted threshold: {len(poly_btc)}")
    print(f"  Kalshi BTC markets w/ extracted threshold: {len(kalshi_btc)}")

    # Look for matched pairs (same threshold ±5%)
    arbs = []
    for p in poly_btc:
        for k in kalshi_btc:
            ratio = max(p["threshold"], k["threshold"]) / min(p["threshold"], k["threshold"])
            if ratio > 1.05:
                continue
            # Two arb directions:
            # 1. Buy YES on kalshi, NO on poly: cost = k_yes_ask + p_no_price
            cost_a = k["yes_ask"] + p["no_price"]
            # 2. Buy NO on kalshi, YES on poly: cost = k_no_ask + p_yes_price
            cost_b = k["no_ask"] + p["yes_price"]
            best_cost = min(cost_a, cost_b)
            best_dir = "kalshi_YES + poly_NO" if cost_a < cost_b else "kalshi_NO + poly_YES"
            edge = 1.0 - best_cost  # positive = arb opportunity
            arbs.append({
                "edge_cents": round(edge * 100, 2),
                "direction": best_dir,
                "kalshi_ticker": k["ticker"],
                "poly_id": p["id"],
                "poly_slug": p["slug"],
                "kalshi_threshold": k["threshold"],
                "poly_threshold": p["threshold"],
                "kalshi_yes_bid_ask": (k["yes_bid"], k["yes_ask"]),
                "poly_yes_price": p["yes_price"],
                "kalshi_q": k["question"][:80],
                "poly_q": p["question"][:80],
                "kalshi_close": k["close_time"],
                "poly_end": p["end_date"],
            })
    arbs.sort(key=lambda a: -a["edge_cents"])
    return arbs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--poly-min-vol", type=float, default=5000.0)
    p.add_argument("--poly-limit", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("reports/cross_platform_arb.json"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"=== Cross-platform arbitrage scan ===\n")
    print(f"[1/3] Pulling Polymarket (vol≥${args.poly_min_vol})...")
    poly = fetch_polymarket(args.poly_limit, args.poly_min_vol)
    print(f"      Got {len(poly)} markets")
    print(f"[2/3] Pulling Kalshi crypto-related series...")
    kalshi = fetch_kalshi_open(series_filter={"KXBTCD", "KXETHD", "KXSOLD",
                                              "KXBTC", "KXETH", "KXSOL",
                                              "KXBTC15M", "KXETH15M", "KXSOL15M"})
    print(f"      Got {len(kalshi)} markets")

    print(f"\n[3/3] Matching BTC price markets...")
    arbs = find_btc_arbs(poly, kalshi)
    print(f"      Generated {len(arbs)} candidate pairs")

    positive = [a for a in arbs if a["edge_cents"] > 0]
    print(f"\n=== Positive-edge candidates (genuine arbs): {len(positive)} ===\n")
    for a in positive[:20]:
        print(f"  +{a['edge_cents']:>5.2f}¢  {a['direction']}")
        print(f"     Kalshi (T={a['kalshi_threshold']:>7.0f}): {a['kalshi_q']}")
        print(f"       bid/ask = ({a['kalshi_yes_bid_ask'][0]:.3f}, {a['kalshi_yes_bid_ask'][1]:.3f})")
        print(f"     Poly (T={a['poly_threshold']:>7.0f}, yes={a['poly_yes_price']:.3f}): {a['poly_q']}")
        print(f"     poly slug: {a['poly_slug']}")
        print()

    # Even near-arbs are interesting (within 5c of breakeven means small directional edge)
    near = [a for a in arbs if -5 <= a["edge_cents"] <= 0 and abs(a["kalshi_threshold"] - a["poly_threshold"])/a["kalshi_threshold"] < 0.02]
    print(f"=== Near-arb (within 5c of breakeven): {len(near)} ===")
    for a in near[:5]:
        print(f"  {a['edge_cents']:+.2f}¢  K@T={a['kalshi_threshold']:.0f} vs P@T={a['poly_threshold']:.0f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n_total_pairs": len(arbs),
        "n_positive_edge": len(positive),
        "n_near_arbs": len(near),
        "candidates": positive[:50],
    }, indent=2, default=str))
    print(f"\n  Report: {args.output}")


if __name__ == "__main__":
    main()
