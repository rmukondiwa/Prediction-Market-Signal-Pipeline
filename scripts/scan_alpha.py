"""
Alpha scanner: structural arbitrage on the live Kalshi catalog.

No LLM calls, no API keys, no waiting for backtest data. Pure axiom-checking
on the existing data/catalog.json snapshot.

Two checks:

  (1) MONOTONICITY VIOLATION on stacked thresholds.
      Within one event, "Above 100" must be at least as likely as "Above 200"
      (and "Below 200" at least as likely as "Below 100"). When the order book
      contradicts this — bid on the less-likely side strictly above ask on the
      more-likely side — there is a riskless lock-in. Edge = bid_high - ask_low
      cents per matched pair, paid out unconditionally at settlement.

  (2) RANGE-PARTITION VIOLATION on mutually exclusive bins.
      When an event partitions the outcome space ("$3000-$3500", "$3500-$4000",
      etc.) the sum of bids must be ≤ 100¢ (else: sell all = free money) and the
      sum of asks must be ≥ 100¢ (else: buy all = free money).

Output: ranked list of candidate trades with worst-case P&L per matched
contract pair, plus a JSON dump for downstream tooling. Reverify in real
time before placing — catalog snapshot ages quickly.

Usage:
    python -m scripts.scan_alpha
    python -m scripts.scan_alpha --min-edge 2 --min-bid 5 --output reports/alpha.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# We deliberately avoid importing anything that needs network/API keys.
from src.catalog.models import CatalogMarket
from src.catalog.store import load_catalog
from src.utils.logging import get_logger

logger = get_logger(__name__)


_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_ABOVE_PAT = re.compile(r"^\s*(?:above|greater\s+than|higher\s+than|over|>=?)\s+", re.IGNORECASE)
_BELOW_PAT = re.compile(r"^\s*(?:below|less\s+than|lower\s+than|under|<=?)\s+", re.IGNORECASE)
_RANGE_PAT = re.compile(
    r"^\s*(?:between\s+)?(\$?[-+]?[\d,.]+(?:\s*[a-zA-Z]+)?)\s*(?:to|-|–|—|and)\s*(\$?[-+]?[\d,.]+(?:\s*[a-zA-Z]+)?)",
    re.IGNORECASE,
)
# Unit suffix multipliers. Order matters — longest first so "billion" matches
# before "b" inside the same scan.
_UNIT_MULTIPLIERS = [
    ("trillion", 1e12), ("trillions", 1e12),
    ("billion", 1e9), ("billions", 1e9),
    ("million", 1e6), ("millions", 1e6),
    ("thousand", 1e3), ("thousands", 1e3),
    ("tn", 1e12),
    ("bn", 1e9), ("bln", 1e9),
    ("mn", 1e6), ("mln", 1e6), ("mm", 1e6),
    ("k", 1e3),
    ("b", 1e9), ("m", 1e6), ("t", 1e12),  # bare suffixes last (greediest)
]


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except (ValueError, AttributeError):
        return None


def _parse_amount(text: str) -> float | None:
    """Parse a numeric amount with optional unit suffix.
    Examples: "1.1M" → 1_100_000.0; "$220 billion" → 2.2e11; "700k" → 700_000.
    Returns None if no number found."""
    if not text:
        return None
    cleaned = text.replace("$", "").strip().lower()
    nums = _NUMBER.findall(cleaned)
    if not nums:
        return None
    base = _to_float(nums[0])
    if base is None:
        return None
    # Look for a unit suffix anywhere after the first number
    after_num = cleaned.split(nums[0], 1)[1] if nums[0] in cleaned else ""
    for unit, mult in _UNIT_MULTIPLIERS:
        # Word-boundary on long unit names; raw match on bare letters
        if len(unit) > 1:
            if re.search(rf"\b{re.escape(unit)}\b", after_num):
                return base * mult
        else:
            # Bare suffix: "1.1m" or "700k". Must touch the number directly
            # (no whitespace) to avoid false positives like "1 in 700".
            if re.match(rf"\s*{re.escape(unit)}\b", after_num):
                return base * mult
    return base


def _parse_threshold(subtitle: str) -> tuple[str, float] | None:
    """Return ('above', X) or ('below', X) when the subtitle encodes one."""
    if not subtitle:
        return None
    s = subtitle.strip()
    m = _ABOVE_PAT.match(s)
    if m:
        v = _parse_amount(s[m.end():])
        if v is not None:
            return ("above", v)
    m = _BELOW_PAT.match(s)
    if m:
        v = _parse_amount(s[m.end():])
        if v is not None:
            return ("below", v)
    return None


def _parse_range(subtitle: str) -> tuple[float, float] | None:
    if not subtitle:
        return None
    m = _RANGE_PAT.match(subtitle.strip())
    if not m:
        return None
    lo = _parse_amount(m.group(1))
    hi = _parse_amount(m.group(2))
    if lo is None or hi is None:
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def _is_liquid(m: CatalogMarket, max_spread: int = 15, min_bid: int = 2, max_ask: int = 98) -> bool:
    if m.yes_bid <= 0 or m.yes_ask <= 0:
        return False
    spread = m.yes_ask - m.yes_bid
    if spread > max_spread or spread < 0:
        return False
    if m.yes_bid < min_bid or m.yes_ask > max_ask:
        return False
    return True


def scan_monotonicity(
    catalog: list[CatalogMarket],
    min_edge_cents: int = 1,
    max_spread: int = 15,
    min_bid: int = 2,
) -> list[dict]:
    """For each event group, find pairs (low, high) of same-direction thresholds
    where the order book contradicts monotonicity. Returns dicts ranked by edge."""
    by_event: dict[str, list[CatalogMarket]] = defaultdict(list)
    for m in catalog:
        if not _is_liquid(m, max_spread=max_spread, min_bid=min_bid):
            continue
        by_event[m.event_ticker].append(m)

    arbs: list[dict] = []

    for event_ticker, markets in by_event.items():
        # Bucket by direction
        above: list[tuple[float, CatalogMarket]] = []
        below: list[tuple[float, CatalogMarket]] = []
        for m in markets:
            parsed = _parse_threshold(m.subtitle)
            if parsed is None:
                continue
            direction, threshold = parsed
            if direction == "above":
                above.append((threshold, m))
            else:
                below.append((threshold, m))

        # ABOVE direction: t_low < t_high → P(above t_low) ≥ P(above t_high).
        # Arb when bid(t_high) > ask(t_low) — sell yes(t_high), buy yes(t_low).
        if len(above) >= 2:
            above.sort(key=lambda x: x[0])
            for i in range(len(above)):
                for j in range(i + 1, len(above)):
                    t_low, m_low = above[i]
                    t_high, m_high = above[j]
                    edge = m_high.yes_bid - m_low.yes_ask
                    if edge >= min_edge_cents:
                        arbs.append(_record_arb(event_ticker, "above", t_low, m_low, t_high, m_high, edge))

        # BELOW direction: t_low < t_high → P(below t_low) ≤ P(below t_high).
        # Arb when bid(t_low) > ask(t_high) — sell yes(t_low), buy yes(t_high).
        if len(below) >= 2:
            below.sort(key=lambda x: x[0])
            for i in range(len(below)):
                for j in range(i + 1, len(below)):
                    t_low, m_low = below[i]
                    t_high, m_high = below[j]
                    edge = m_low.yes_bid - m_high.yes_ask
                    if edge >= min_edge_cents:
                        arbs.append(_record_arb(event_ticker, "below", t_low, m_low, t_high, m_high, edge))

    arbs.sort(key=lambda a: a["edge_cents"], reverse=True)
    return arbs


def _record_arb(event: str, direction: str, t_low: float, m_low: CatalogMarket,
                t_high: float, m_high: CatalogMarket, edge: int) -> dict:
    """Build a structured arb record describing the trade."""
    if direction == "above":
        # sell yes(t_high), buy yes(t_low)
        sell, sell_t = m_high, t_high
        buy, buy_t = m_low, t_low
    else:
        sell, sell_t = m_low, t_low
        buy, buy_t = m_high, t_high
    return {
        "type": "monotonicity",
        "event_ticker": event,
        "direction": direction,
        "edge_cents": edge,
        "sell_yes": {
            "ticker": sell.ticker,
            "subtitle": sell.subtitle,
            "threshold": sell_t,
            "bid": sell.yes_bid, "ask": sell.yes_ask,
        },
        "buy_yes": {
            "ticker": buy.ticker,
            "subtitle": buy.subtitle,
            "threshold": buy_t,
            "bid": buy.yes_bid, "ask": buy.yes_ask,
        },
        "title": sell.title,
        "category": sell.category,
        "trade": (
            f"SELL YES {sell.ticker} @ {sell.yes_bid}¢ / "
            f"BUY YES {buy.ticker} @ {buy.yes_ask}¢ → "
            f"locks {edge}¢ per pair"
        ),
    }


_TIME_HINTS = re.compile(r"\b(am|pm|hour|hourly|minute|second|window|window?s?)\b", re.IGNORECASE)


def _looks_like_time_bin(subtitle: str) -> bool:
    return bool(subtitle and _TIME_HINTS.search(subtitle))


def scan_range_partitions(
    catalog: list[CatalogMarket],
    min_violation_cents: int = 2,
    max_spread: int = 15,
    min_bid: int = 2,
) -> list[dict]:
    """Find events whose markets form a complete numeric partition
    (e.g. $X-$Y bins) and check sum-of-bids ≤ 100, sum-of-asks ≥ 100.

    Heuristics to avoid false positives:
      - Skip time-of-day bins ("1-2 AM"). Multiple hours within one tracking
        window are usually independent events, not mutually exclusive.
      - Require bin labels to span a meaningful numeric range — total span
        must be ≥ 50 in the underlying units, ruling out tiny categorical
        partitions that happen to look numeric.
    """
    by_event: dict[str, list[tuple[float, float, CatalogMarket]]] = defaultdict(list)
    for m in catalog:
        if not _is_liquid(m, max_spread=max_spread, min_bid=min_bid):
            continue
        if _looks_like_time_bin(m.subtitle):
            continue
        rng = _parse_range(m.subtitle or "")
        if rng is None:
            continue
        by_event[m.event_ticker].append((rng[0], rng[1], m))

    arbs: list[dict] = []
    for event_ticker, items in by_event.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[0])
        contiguous = all(
            abs(items[i][1] - items[i + 1][0]) < 1e-6
            for i in range(len(items) - 1)
        )
        if not contiguous:
            continue
        # Require the partition to span enough numeric ground to plausibly
        # represent a measure rather than discrete categories. Cheap, but
        # filters out obviously-not-money cases.
        total_span = items[-1][1] - items[0][0]
        if total_span < 50:
            continue

        bids = sum(it[2].yes_bid for it in items)
        asks = sum(it[2].yes_ask for it in items)

        if bids > 100 + min_violation_cents:
            arbs.append({
                "type": "partition_overpriced",
                "event_ticker": event_ticker,
                "edge_cents": bids - 100,
                "sum_bids": bids, "sum_asks": asks,
                "n_bins": len(items),
                "total_span": total_span,
                "title": items[0][2].title,
                "trade": f"SELL YES on all {len(items)} bins → collect {bids}¢ for $1 of liability",
                "bins": [{"ticker": it[2].ticker, "range": [it[0], it[1]],
                          "bid": it[2].yes_bid, "ask": it[2].yes_ask} for it in items],
            })
        elif asks < 100 - min_violation_cents:
            arbs.append({
                "type": "partition_underpriced",
                "event_ticker": event_ticker,
                "edge_cents": 100 - asks,
                "sum_bids": bids, "sum_asks": asks,
                "n_bins": len(items),
                "total_span": total_span,
                "title": items[0][2].title,
                "trade": f"BUY YES on all {len(items)} bins → pay {asks}¢ for guaranteed $1",
                "bins": [{"ticker": it[2].ticker, "range": [it[0], it[1]],
                          "bid": it[2].yes_bid, "ask": it[2].yes_ask} for it in items],
            })

    arbs.sort(key=lambda a: a["edge_cents"], reverse=True)
    return arbs


def scan_soft_kinks(
    catalog: list[CatalogMarket],
    min_kink_pp: int = 5,
    max_spread: int = 15,
    min_bid: int = 2,
) -> list[dict]:
    """Soft-signal scanner: NOT risk-free arbs. Looks for non-monotone bumps in
    the implied-prob curve across stacked thresholds. A true axiom violation
    would be bid(high) > ask(low); a 'kink' is when mid(high) > mid(low) by
    less than the spread — directional alpha if the lower-priced side is
    actually correct.
    """
    by_event: dict[str, list[tuple[float, str, CatalogMarket]]] = defaultdict(list)
    for m in catalog:
        if not _is_liquid(m, max_spread=max_spread, min_bid=min_bid):
            continue
        parsed = _parse_threshold(m.subtitle)
        if parsed is None:
            continue
        direction, threshold = parsed
        by_event[(m.event_ticker, direction)].append((threshold, direction, m))

    kinks: list[dict] = []
    for (event_ticker, direction), items in by_event.items():
        if len(items) < 3:
            continue
        items.sort(key=lambda x: x[0])

        for i in range(len(items) - 1):
            t1, _, m1 = items[i]
            t2, _, m2 = items[i + 1]
            # mid in cents
            mid1 = (m1.yes_bid + m1.yes_ask) / 2
            mid2 = (m2.yes_bid + m2.yes_ask) / 2

            # For "above" direction: mid should be DECREASING as threshold rises.
            # For "below" direction: mid should be INCREASING as threshold rises.
            if direction == "above":
                expected_diff = mid1 - mid2
            else:
                expected_diff = mid2 - mid1

            if expected_diff >= 0:
                continue

            # Kink: probability went the wrong way. Magnitude in pp.
            magnitude = -expected_diff
            if magnitude < min_kink_pp:
                continue

            # Identify which side is more likely "correct" using neighbors:
            # if t_(i-1) and t_(i+1)'s mids straddle, that's stronger evidence.
            kinks.append({
                "type": "kink",
                "event_ticker": event_ticker,
                "direction": direction,
                "magnitude_pp": round(magnitude, 1),
                "title": m1.title,
                "category": m1.category,
                "lower_threshold": {
                    "ticker": m1.ticker, "subtitle": m1.subtitle,
                    "threshold": t1, "bid": m1.yes_bid, "ask": m1.yes_ask, "mid": mid1,
                },
                "higher_threshold": {
                    "ticker": m2.ticker, "subtitle": m2.subtitle,
                    "threshold": t2, "bid": m2.yes_bid, "ask": m2.yes_ask, "mid": mid2,
                },
                "thesis": (
                    f"{direction.upper()} curve has a kink between t={t1:g} ({mid1:.1f}¢) and "
                    f"t={t2:g} ({mid2:.1f}¢). Direction violates monotonicity by {magnitude:.1f}pp. "
                    f"One of these two prints is wrong; trade the side you have other reasons to trust."
                ),
            })
    kinks.sort(key=lambda k: k["magnitude_pp"], reverse=True)
    return kinks


def _print_top_monotonicity(arbs: list[dict], top: int = 20) -> None:
    if not arbs:
        print("\n  (no monotonicity violations found)")
        return
    print(f"\n=== MONOTONICITY VIOLATIONS (top {min(top, len(arbs))} of {len(arbs)}) ===\n")
    for i, a in enumerate(arbs[:top], 1):
        print(f"[{i:2}] +{a['edge_cents']}¢/pair  {a['event_ticker']}  ({a['direction']})")
        print(f"     {a['title'][:80]}")
        s = a["sell_yes"]
        b = a["buy_yes"]
        print(f"     SELL YES {s['ticker']:<45} @ bid={s['bid']:>3}¢  ({s['subtitle']})")
        print(f"     BUY  YES {b['ticker']:<45} @ ask={b['ask']:>3}¢  ({b['subtitle']})")
        print()


def _print_top_partition(arbs: list[dict], top: int = 20) -> None:
    if not arbs:
        print("\n  (no range-partition violations found)")
        return
    print(f"\n=== RANGE-PARTITION VIOLATIONS (top {min(top, len(arbs))} of {len(arbs)}) ===\n")
    for i, a in enumerate(arbs[:top], 1):
        print(f"[{i:2}] +{a['edge_cents']}¢  {a['event_ticker']}  ({a['type']})")
        print(f"     {a['title'][:80]}")
        print(f"     n_bins={a['n_bins']}  sum_bids={a['sum_bids']}¢  sum_asks={a['sum_asks']}¢")
        print(f"     {a['trade']}")
        print()


def _print_top_kinks(kinks: list[dict], top: int = 20) -> None:
    if not kinks:
        print("\n  (no soft kinks found)")
        return
    print(f"\n=== SOFT KINKS (top {min(top, len(kinks))} of {len(kinks)}, NOT riskless) ===\n")
    for i, k in enumerate(kinks[:top], 1):
        lo = k["lower_threshold"]
        hi = k["higher_threshold"]
        print(f"[{i:2}] kink={k['magnitude_pp']}pp  {k['event_ticker']}  ({k['direction']})")
        print(f"     {k['title'][:80]}")
        print(f"     {lo['subtitle']:30} bid={lo['bid']:>3}¢ ask={lo['ask']:>3}¢ mid={lo['mid']:.1f}¢")
        print(f"     {hi['subtitle']:30} bid={hi['bid']:>3}¢ ask={hi['ask']:>3}¢ mid={hi['mid']:.1f}¢")
        print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan Kalshi catalog for structural arbitrage")
    p.add_argument("--catalog", type=Path, default=Path("data/catalog.json"))
    p.add_argument("--output", type=Path, default=Path("reports/alpha.json"))
    p.add_argument("--min-edge", type=int, default=1, help="Minimum edge in cents")
    p.add_argument("--min-bid", type=int, default=2, help="Filter out illiquid markets (bid below this)")
    p.add_argument("--max-spread", type=int, default=15, help="Filter out wide-spread markets (cents)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--min-kink", type=int, default=5, help="Soft-kink magnitude threshold (pp)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = load_catalog(args.catalog)
    print(f"Loaded {len(catalog)} markets from {args.catalog}")

    monotonicity = scan_monotonicity(
        catalog, min_edge_cents=args.min_edge,
        max_spread=args.max_spread, min_bid=args.min_bid,
    )
    partitions = scan_range_partitions(
        catalog, min_violation_cents=args.min_edge,
        max_spread=args.max_spread, min_bid=args.min_bid,
    )
    kinks = scan_soft_kinks(
        catalog, min_kink_pp=args.min_kink,
        max_spread=args.max_spread, min_bid=args.min_bid,
    )

    _print_top_monotonicity(monotonicity, top=args.top)
    _print_top_partition(partitions, top=args.top)
    _print_top_kinks(kinks, top=args.top)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "monotonicity_violations": monotonicity,
        "partition_violations": partitions,
        "soft_kinks": kinks,
        "filters": {
            "min_edge_cents": args.min_edge,
            "min_bid": args.min_bid,
            "max_spread": args.max_spread,
            "min_kink_pp": args.min_kink,
        },
        "n_monotonicity": len(monotonicity),
        "n_partition": len(partitions),
        "n_kinks": len(kinks),
    }, indent=2, default=str))

    total_edge = sum(a["edge_cents"] for a in monotonicity) + sum(a["edge_cents"] for a in partitions)
    print(f"\nRiskless arbs: {len(monotonicity)} monotonicity + {len(partitions)} partition")
    print(f"Soft kinks (directional, not riskless): {len(kinks)}")
    print(f"Total riskless locked edge: {total_edge}¢ (gross, before fees)")
    print(f"Report written to {args.output}")
    print("\nWARNING: catalog ages quickly. Reverify each trade live before placing.")


if __name__ == "__main__":
    main()
