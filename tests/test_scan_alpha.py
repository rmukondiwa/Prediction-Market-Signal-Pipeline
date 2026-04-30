"""Tests for the alpha scanner — especially the threshold parser, since a bug
there silently produces fake arbs."""
from __future__ import annotations

import importlib

scan_alpha = importlib.import_module("scripts.scan_alpha")


def test_parse_amount_plain_number():
    assert scan_alpha._parse_amount("1294.97") == 1294.97
    assert scan_alpha._parse_amount("100000") == 100000


def test_parse_amount_with_unit_suffixes():
    """The bug that triggered this test: '1.1M' parsed as 1.1, breaking
    the sort against '700k' parsed as 700, generating fake arbs."""
    assert scan_alpha._parse_amount("1.1M") == 1_100_000
    assert scan_alpha._parse_amount("700k") == 700_000
    assert scan_alpha._parse_amount("1.0M") == 1_000_000
    assert scan_alpha._parse_amount("$220 billion") == 220_000_000_000
    assert scan_alpha._parse_amount("1 billion") == 1_000_000_000
    assert scan_alpha._parse_amount("940 million") == 940_000_000


def test_parse_amount_ordering_consistency():
    """The whole point of fixing this: 1.1M must compare > 700k after parsing."""
    assert scan_alpha._parse_amount("1.1M") > scan_alpha._parse_amount("700k")
    assert scan_alpha._parse_amount("1 billion") > scan_alpha._parse_amount("999 million")


def test_parse_amount_with_dollar_signs():
    assert scan_alpha._parse_amount("$3.5 billion") == 3_500_000_000


def test_parse_amount_returns_none_when_empty():
    assert scan_alpha._parse_amount("") is None
    assert scan_alpha._parse_amount("nothing here") is None


def test_parse_threshold_above():
    assert scan_alpha._parse_threshold("Above 1.1M") == ("above", 1_100_000)
    assert scan_alpha._parse_threshold("above 700k") == ("above", 700_000)
    assert scan_alpha._parse_threshold("Above 25000") == ("above", 25000)


def test_parse_threshold_below():
    assert scan_alpha._parse_threshold("Below 1.0M") == ("below", 1_000_000)
    assert scan_alpha._parse_threshold("less than 100") == ("below", 100)


def test_parse_threshold_no_match():
    assert scan_alpha._parse_threshold("Yes") is None
    assert scan_alpha._parse_threshold("Between 100 and 200") is None
    assert scan_alpha._parse_threshold("") is None
    assert scan_alpha._parse_threshold(None) is None


def test_parse_range_with_units():
    assert scan_alpha._parse_range("$3000 to $3500") == (3000, 3500)
    assert scan_alpha._parse_range("between 1.0M and 2.0M") == (1_000_000, 2_000_000)


def test_parse_range_no_match():
    assert scan_alpha._parse_range("Above 100") is None


def test_scan_monotonicity_no_violation_when_axiom_holds(sample_catalog_for_arb):
    """Real Ohio quotes — no violation should be detected after parser fix."""
    arbs = scan_alpha.scan_monotonicity(sample_catalog_for_arb, min_edge_cents=1)
    # No legitimate monotonicity violation in a normal market — verify no
    # spurious arb pops out for the Ohio-style stacked thresholds
    for a in arbs:
        # Each reported arb must satisfy the actual axiom violation
        sell = a["sell_yes"]
        buy = a["buy_yes"]
        assert sell["bid"] > buy["ask"], (
            f"Reported arb has bid({sell['ticker']})={sell['bid']} "
            f"NOT > ask({buy['ticker']})={buy['ask']} — broken arb"
        )


def test_scan_monotonicity_finds_genuine_violation():
    """Construct a synthetic event with a real monotonicity violation and
    verify the scanner reports it correctly."""
    from src.catalog.models import CatalogMarket
    catalog = [
        # Above 100: market thinks 30%
        CatalogMarket(
            ticker="EVT-100", event_ticker="EVT", title="Test event",
            subtitle="Above 100", category="Test", status="open",
            yes_bid=29, yes_ask=31, implied_probability=0.30,
        ),
        # Above 200: market thinks 50% — VIOLATION (less likely event priced higher)
        CatalogMarket(
            ticker="EVT-200", event_ticker="EVT", title="Test event",
            subtitle="Above 200", category="Test", status="open",
            yes_bid=49, yes_ask=51, implied_probability=0.50,
        ),
    ]
    arbs = scan_alpha.scan_monotonicity(catalog, min_edge_cents=1)
    assert len(arbs) == 1
    a = arbs[0]
    # Should sell yes(200) at 49, buy yes(100) at 31 → edge 18
    assert a["sell_yes"]["ticker"] == "EVT-200"
    assert a["buy_yes"]["ticker"] == "EVT-100"
    assert a["edge_cents"] == 49 - 31
