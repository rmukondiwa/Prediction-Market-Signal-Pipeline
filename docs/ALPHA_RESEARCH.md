# Alpha Research — Beyond Cross-Platform Arb

A catalog of edge sources we have NOT yet exploited, with concrete test
instructions for each. Read alongside [STRATEGIES.md](STRATEGIES.md)
(which lists what's been tried) and [USER_MANUAL.md](USER_MANUAL.md)
(which covers the active arb infra).

Each source is rated by **edge-likelihood** (gut estimate based on what
we've observed) and **test cost** (engineering hours + LLM/API spend).

---

## Quick map

| # | Alpha source | Edge likelihood | Test cost | Infra needed |
|---|---|---|---|---|
| 1 | Polymarket within-platform multi-market arb | medium | $0 + 2h | none new |
| 2 | Cross-event implied-curve consistency | medium-high | $0 + 4h | new scanner |
| 3 | Live Kalshi WS book aggregator | medium | $0 + 1d | WS subscriber wiring |
| 4 | Sports-specific game-flow odds | medium-high | $0 + 1d | live data feed |
| 5 | News/headline dislocation | high (short-lived) | $20/mo + 1w | news API + latency tightening |
| 6 | Polymarket whale copy-trade | medium | $0 + 6h | Polygon RPC |
| 7 | Calendar / time-window arb | low-medium | $0 + 4h | refined scanner |
| 8 | Implied volatility crossover (CEX options ↔ Kalshi) | medium | $0 + 3d | Deribit/Coinbase options data |
| 9 | Settlement-timing arb (rare) | very low (short window) | $0 + 1d | event monitoring |
| 10 | Conditional probability stronger grouping | low | $0 + 2h | LLM-grouped axiom check |

We discuss each below with **runnable test instructions**.

---

## 1. Polymarket within-platform multi-market arb

### Thesis

Polymarket events sometimes have multiple binary markets that should
satisfy probability axioms among themselves. E.g., for "Who wins the
2024 Republican primary?" the market lists candidates as separate binary
markets — sum of YES probabilities should ≤ 1.0 minus the "other"
probability.

Polymarket's CLOB doesn't enforce this — operator-curated events vary.
A scanner could find sums >$1 (or partition violations) and surface
free-arb candidates.

### Edge likelihood: **medium**

Polymarket has lower scanner pressure than Kalshi (we've seen `scan_alpha`
return 0 on Kalshi but Polymarket events haven't been swept by us).

### Test instructions

**Step 1 — Build the scanner.** Adapt `scan_alpha.py` partition logic
to Polymarket. Polymarket events have an `events` endpoint:

```bash
curl -s "https://gamma-api.polymarket.com/events?active=true&closed=false&limit=50" \
  | jq '.[] | {id, slug, markets: [.markets[]?.question]}' | head -40
```

For each event, pull the market list, fetch each market's CLOB top-of-book
via:

```python
from src.ingestion.polymarket.clob_client import get_book
for m in event_markets:
    yes_token = m["clobTokenIds"][0]
    book = get_book(yes_token)
    yes_ask = book.top_ask.price
    # ... sum across all markets in the event
```

**Step 2 — Run partition check.** For each event with N binary markets:
```
sum_yes_asks = sum(yes_ask for each market)
if sum_yes_asks < 1.0:
    # Buying every YES leg pays $1 + remaining markets' payouts
    # Free arb candidate
```

### Success metric
- Find at least 1 partition arb in 100 events scanned
- Edge ≥ $0.02 per $1 round-trip after Polygon gas

### Failure mode to watch for
"Other" outcome handling — many Polymarket events are "Person A wins" /
"Person B wins" / etc with implicit "neither" probability. Sum of YES
probabilities <1 is expected, not arb.

**EMPIRICAL CONFIRMATION (2026-05-07):** Ran the scanner on the top 20
Polymarket events by volume. Result: 2 "arbs" found, both false
positives — they were time-bucket events ("Bitcoin hits $150k by June"
/ "by December" with implicit "later") where the partition is
incomplete. Real exhaustive partitions (e.g., "Democratic 2028 nominee"
with 44 candidates) summed to exactly $1.0000 — no arb but confirms
the scanner mechanics work.

**Required next iteration:** add an `is_exhaustive_partition` filter
that confirms the event's markets are mutually exclusive AND
collectively exhaustive. Heuristic: check the event has explicit "Other"
or "None of the above" market, OR that question text doesn't contain
"by [date]" patterns (which always have implicit time-after-last-bucket
state).

### Where to put it
`scripts/scan_polymarket_partition.py` — parallel to `scan_alpha.py`.

---

## 2. Cross-event implied-curve consistency

### Thesis

Kalshi has hierarchical event groups: e.g., `KXBTC15M-26MAY071700` (15-min
window) and `KXBTC-26MAY071700` (1-hour window) and `KXBTCD-26MAY07`
(daily, settles at 5pm).

These should be related: `P(BTC up at 5pm)` ≈ product/integral of P(up
in each 15-min window). When the daily price diverges from the 15-min
strip, there's a mispricing.

### Edge likelihood: **medium-high**

This is the prediction-market analog of "calendar spread" arb in vanilla
options. Different time horizons of the same underlying must satisfy a
no-arb condition. Different LP / market-maker exposure on each duration
can create disconnects.

### Test instructions

**Step 1 — Map the hierarchy.** For each underlying (KXBTC, KXETH, ...),
collect all open markets:

```python
import json
from collections import defaultdict
catalog = json.load(open("data/catalog.json"))

# Group by (underlying, expiration_time)
by_window = defaultdict(list)
for m in catalog:
    t = m["ticker"]
    if t.startswith("KXBTC"):
        # parse expiration from ticker; structure varies by series
        ...
```

**Step 2 — For each (underlying, expiry), compute synthetic 5pm
probabilities.** From the 15-min markets:
```
P_5pm_synth = product over (windows from now to 5pm) of P(window i)
```

This is a rough approximation — assumes intra-window "up" outcomes are
independent (NOT true, but a baseline).

**Step 3 — Compare to actual `KXBTCD-<date>` daily market.** Difference
≥ 5pp is suspicious. Look for direction (synth > daily → daily is
underpricing; vice versa).

### Success metric
- ≥1 deviation ≥10pp surfaced from a single scan
- Confirmed actionable after rules-reading

### Failure mode
"Settlement window" mismatch — KXBTCD might use a different 60-second
average than the integral over 15-min windows. Independence assumption
overestimates volatility risk.

### Where to put it
`scripts/scan_term_structure.py`

---

## 3. Live Kalshi WS book aggregator

### Thesis

The cached catalog (`data/catalog.json`) has degenerate `yes_bid=0,
yes_ask=1` for most markets — there's no live order book. Any structural
arb scanner running on cached data is dead-on-arrival because we can't
detect monotonicity violations without real bids/asks.

The Kalshi WebSocket already streams `orderbook_snapshot` and
`orderbook_delta` events for subscribed tickers. If we maintain a live
in-memory book for, say, the 800 BTCD strike-stack markets, we can
detect monotonicity violations in real time.

### Edge likelihood: **medium**

`scan_alpha` returns 0 hits on cached data. With live data, we'd find
SOMETHING — even if MMs eat them in seconds, fast detection might let
us race for them.

### Test instructions

**Step 1 — Subscribe** the Kalshi WS to all KXBTCD-* markets. Modify
`src/ingestion/kalshi/websocket_client.py` config to include a market
list of strike-stack markets.

**Step 2 — Maintain a live book.** Aggregate `orderbook_delta` events
into a per-ticker dict of `{price_cents: size}` for both yes/no sides.

**Step 3 — Run a monotonicity check** every 100ms:
```python
# For each event group (e.g. KXBTCD-26MAY07):
strikes_sorted_by_threshold = sorted(markets, key=lambda m: m.threshold)
for i in range(len(strikes_sorted)-1):
    if top_bid[i].cents > top_ask[i+1].cents:
        # Violation — buy YES at strike i+1 ask, sell YES at strike i bid
        # Lock in $0.01+ per contract risk-free
        ...
```

**Step 4 — Log every violation** with timestamp + spread. Even if not
auto-tradeable, the LOG tells you whether real-time arb exists at our
latency.

### Success metric
- ≥1 monotonicity violation detected in 24 hours of WS streaming
- Violation persists ≥500ms (longer than our place-order latency)

### Failure mode
Even if violations exist, our REST place-order latency (~200-500ms) means
we lose the race to faster bots. The log tells us whether to invest in
faster execution.

### Where to put it
- Modify `src/ingestion/kalshi/websocket_client.py` for full strike-stack subscription
- New `src/portfolio/order_book.py` — live book aggregator
- New `scripts/scan_live_book.py` — runs the WS + monotonicity check

---

## 4. Sports-specific game-flow odds

### Thesis

Kalshi has live NBA/NHL/MLB/NFL game-flow markets:
- `KXNBAGAME-<gameid>-<team>` → "Will team X win?"
- `KXNBA2HSPREAD-<gameid>-<team>4` → "Will team X cover -4 in 2H?"
- `KXNBATOTAL-<gameid>-206` → "Will combined score exceed 206?"

These markets update in real-time during games. Compared to live odds at
sportsbooks (DraftKings, FanDuel, Pinnacle), Kalshi's prices can diverge
because of the limited Kalshi audience.

### Edge likelihood: **medium-high**

Sportsbooks are professionally-managed and tight; Kalshi is retail-driven
and slower. A live cross-comparison should surface 1-3¢ edges
intermittently.

### Test instructions

**Step 1 — Get live sportsbook odds.** Free APIs include:
- The Odds API (https://the-odds-api.com/) — free tier 500 req/mo, $30/mo for more
- Pinnacle (paid, but the highest-quality reference)
- DraftKings public odds page (scrape, no auth)

**Step 2 — Map Kalshi game IDs to sportsbook game IDs.** Both use
team abbreviations + dates; build a manual mapping or fuzzy-match.

**Step 3 — Compare implied probabilities.**
```
sportsbook_implied = 1 / (decimal_odds * (1 - vig))
kalshi_implied = (yes_bid + yes_ask) / 200
edge = sportsbook_implied - kalshi_implied
```

If `|edge| ≥ 0.03`, it's potentially actionable.

**Step 4 — Scan during live games.** Run every 30 seconds during NBA
prime time. Most edges will appear in the last 5 min of a quarter when
Kalshi traders are slow to update.

### Success metric
- ≥3 detected edges ≥3pp during 1 NBA game
- ≥1 actionable (depth ≥$50 on Kalshi side)

### Failure mode
Sportsbook lines are post-vig; comparing to mid is wrong. Use the no-vig
implied: `implied / (sum of all implieds for the event)`.

Also: timing mismatch. If Kalshi quotes a "halftime over/under" but the
sportsbook just adjusted, you might be reading stale data. Use freshness
timestamps.

### Where to put it
- `src/ingestion/sportsbook/` — new module
- `scripts/scan_sports_odds_arb.py`

---

## 5. News/headline dislocation

### Thesis

When breaking news hits (e.g., "Powell resigns", "Iran missile strike"),
prediction markets re-price slowly. A 5-30 second edge exists between
"news breaks" and "Kalshi book updates".

### Edge likelihood: **high** but extremely short-lived

The trade window is 5-30s; our REST latency is 200-500ms. Marginally
viable but technically demanding.

### Test instructions

**Step 1 — Set up news ingestion.** Options ranked by latency:
- Twitter API (real-time but rate-limited): ~$200/mo for live feed
- Reuters/AP wire RSS: ~5-30s lag
- Newsapi.org: 1-15min lag (too slow for arb)
- Direct exchange feeds (Bloomberg etc.): $24,000+/yr (out of scope)

**Step 2 — LLM classifier on incoming headlines.** For each headline,
ask Gemini: "Does this affect any Kalshi market we hold? Which ones?"

**Step 3 — Race condition trigger.** When the LLM emits a relevant
ticker, immediately submit IOC orders before the book reprices.

**Step 4 — Backtest first.** Use historical news + historical Kalshi
ticks to measure: how many seconds is the typical book-reprice lag?
What's the realistic captured edge after our 500ms latency?

### Success metric
- Event-driven edge captured ≥$0.05 per affected market
- ≥5 events in a representative week

### Failure mode
Most "news" is noise — the LLM filter has high false-positive rate, and
the cost of submitting wrong orders eats the edge.

### Where to put it
- `src/ingestion/news/` — new module
- `scripts/news_arb_live.py` — high-priority loop running alongside main orchestrator

---

## 6. Polymarket whale copy-trade

### Thesis

Polymarket positions are public on the Polygon chain. A scanner can
identify wallets that have been profitable historically (track their PnL
on settled markets) and mirror their trades with a small delay.

### Edge likelihood: **medium**

Polymarket has sophisticated traders. Several documented "whales" with
$100k+ profits over 12 months. Even with a 60-second copy delay, the
edge could be 50-70% of theirs.

### Test instructions

**Step 1 — Inventory recent Polymarket settlements.** Pull the last 90
days of settled markets and their on-chain trades:
```python
import web3
w3 = web3.Web3(web3.HTTPProvider("https://polygon-rpc.com"))
# CTFExchange contract events: OrderFilled, etc.
# Index by trader address; group by event resolution
```

**Step 2 — Compute trader PnL.** For each address, sum (filled_volume ×
realized_outcome_premium - fees - gas).

**Step 3 — Filter for size + win rate.** Wallets with:
- ≥$50k cumulative volume
- Win rate ≥60% on resolved markets
- Active in last 30 days

**Step 4 — Subscribe to those wallets' new positions.** Poll their
positions every 60s; when a new entry appears, evaluate (manual or LLM
filter) and decide whether to mirror.

### Success metric
- Identify 5+ "smart money" wallets with provable edge
- Copy-trade simulator on past 30 days shows positive expectancy

### Failure mode
Survivorship bias — wallets that look good on past data may have lucky
streaks. Need to validate on rolling windows.

Lag — by the time you see the trade, the price has moved.

### Where to put it
- `src/ingestion/polygon/` — RPC client
- `scripts/polymarket_whale_tracker.py`
- `scripts/polymarket_copy_trade.py`

---

## 7. Calendar / time-window arb

### Thesis

Within Kalshi's BTC15M ladder, market `KXBTC15M-26MAY071730` and
`KXBTC15M-26MAY071745` (consecutive 15-min windows) imply BTC's expected
direction in each interval. If implied(1730) is far from implied(1745),
it's a calendar mispricing (assuming markets resolve independently).

### Edge likelihood: **low-medium**

The two windows are independently priced and probably correctly so. But
during low-volume periods (overnight), MMs may not refresh, and stale
quotes diverge from the realized rate.

### Test instructions

**Step 1 — Pull 30 days of historical 15M market data** from
`experiments/decay/data/expanded_universe_meta.json` (resolved only —
provides realized outcomes per window).

**Step 2 — Compute consecutive-window correlation.** For each underlying:
- For each pair of consecutive windows, record (implied_1, implied_2,
  realized_1, realized_2)
- Plot implied(t+1) vs implied(t)
- Check residuals — is there structure?

**Step 3 — Identify outliers.** Windows where implied(t+1) - implied(t)
> X (e.g., 0.10) but the realized continuation rate doesn't match.

**Step 4 — Test forward.** Live-track the next 24 hrs of consecutive
windows; for each pair where the gap is ≥0.10, place a small "fade
the gap" trade and measure realized PnL.

### Success metric
- Statistically significant deviation found in historical analysis (p<0.05)
- Forward-test cumulative PnL ≥0 over 100 windows

### Failure mode
Real volatility justifies the gap. Without an exogenous reason (news
event), the gap is just market efficiency disagreeing with our model.

### Where to put it
- `scripts/research_calendar_arb.py` — historical analysis
- `scripts/scan_calendar_arb_live.py` — forward scanner

---

## 8. Implied volatility crossover (CEX options ↔ Kalshi)

### Thesis

Deribit and Coinbase Derivatives quote BTC options. The implied
distribution from the option chain (e.g., from Black-Scholes inversion
or risk-neutral density) gives us a "true" probability for "BTC > $X
on date Y". We can compare to Kalshi's `KXBTCD-26MAY07-T<X>`.

### Edge likelihood: **medium**

CEX options are deep, tight, and reflect institutional pricing. If Kalshi
disagrees by ≥3pp, that's a real signal.

### Test instructions

**Step 1 — Fetch Deribit option chain.**
```bash
curl -s "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option" \
  | jq '.result[] | select(.expiration_timestamp / 1000 > now)' | head
```

**Step 2 — Compute risk-neutral CDF from option chain.**
```python
# Use butterfly spreads at adjacent strikes
# Or: Breeden-Litzenberger
#   ∂²C/∂K² = e^(rT) × pdf
# Integrate numerically to get CDF(K) = P(S_T < K)
```

**Step 3 — Compare to Kalshi.** For each Kalshi strike `KXBTCD-<date>-T<X>`,
compute `Kalshi_implied_P(BTC > X)` and `option_implied_P(BTC > X)`.

**Step 4 — Edge = option_implied - kalshi_implied.** If ≥3pp, surface.

### Success metric
- ≥10 Kalshi strikes where edge ≥3pp at any moment
- Edge persists ≥1 hr (longer than our place-order latency)

### Failure mode
Different settlement conventions. Deribit settles at index avg over a
window; Kalshi might use a single-print. The "true" probabilities differ.

Also: option chain liquidity. Deep OTM strikes have wide spreads and
the implied CDF is noisy there.

### Where to put it
- `src/ingestion/deribit/` — REST client
- `scripts/research_cex_kalshi_iv.py` — historical comparison
- `scripts/scan_cex_kalshi_arb.py` — live scanner

---

## 9. Settlement-timing arb

### Thesis

When Kalshi settles a market at 5pm but Polymarket's equivalent doesn't
settle until midnight UTC, you can sometimes lock in the Kalshi outcome
and trade Polymarket profitably on certain knowledge.

### Edge likelihood: **very low**

This is rare. Most settlement windows are close enough that no info-gap
exists. Only happens during cross-platform listings with mismatched cut-offs.

### Test instructions

**Step 1 — Identify candidate event pairs.** For each cross-platform
match (from arb v2 scanner), record both venues' resolution timestamps.

**Step 2 — Filter for ≥1 hr settlement gap** AND for events where the
outcome is determined by an event observable to the public before the
later settlement (e.g., a press conference outcome).

**Step 3 — Manual evaluation.** These are rare enough that automated
scanning isn't worth it. Set up an alert when the arb scanner finds a
pair with settlement_gap ≥1 hr.

### Success metric
- 1+ usable settlement-timing arb identified per quarter

### Failure mode
The entire universe is rare. Most candidate edges are illusions.

### Where to put it
- Add `settlement_gap_seconds` to scanner output
- Alert when ≥3600 seconds

---

## 10. Conditional probability (stronger LLM grouping)

### Thesis

`scan_alpha` checks per-event partition violations. But conditional
relationships across events (e.g., P(Trump impeached) ≤ P(Trump
re-elected)) require LLM grouping to identify. We have the LLM
infrastructure but haven't built the grouping module.

### Edge likelihood: **low**

Hard mispricings are rare. Soft inconsistencies are common but require
qualitative judgment to trade.

### Test instructions

**Step 1 — Sample 50 random Kalshi events** and ask Gemini: "Are any of
these market pairs conditionally related (P(A∧B), P(A|B), P(A∨B))?"

**Step 2 — For each pair flagged**, check the prices satisfy the
conditional-bound axiom.

**Step 3 — Surface violations.** Output `(market_a, market_b, axiom_violated, magnitude)`.

### Success metric
- ≥1 conditional-bound violation found per 200 markets sampled
- Magnitude ≥3pp

### Failure mode
LLM grouping has high false-positive rate. Many "related" markets aren't
strictly conditional.

### Where to put it
- `scripts/scan_conditional_axioms.py`

---

## Recommended next-build order

If I had a week, I'd build them in this order — by expected ROI:

| Priority | Build | Reason |
|---|---|---|
| 1 | Polymarket within-platform partition scanner (#1) | Cheapest test, real chance of finding free arbs |
| 2 | Live Kalshi WS book aggregator (#3) | Unblocks every structural-arb scanner; cached catalog is a dead end |
| 3 | Sports odds vs Kalshi NBA scanner (#4) | Has measurable edge in retail-driven Kalshi sports markets |
| 4 | Cross-event implied curve consistency (#2) | Novel, low-competition, intermediate effort |
| 5 | CEX option implied vs Kalshi BTCD (#8) | Highest ceiling but requires Deribit / option pricing know-how |
| 6 | Polymarket whale copy-trade (#6) | Documented edge but lag is hard |

Items 5 (news), 7 (calendar), 9 (settlement timing), 10 (conditional)
are lower priority — niche or hard-to-execute.

---

## Testing protocol (for any new alpha source)

For each candidate, follow this sequence to avoid spending time on
strategies that won't survive:

```
   ┌──────────────────────────────────────────────────────┐
   │  1. Hypothesis-state                                 │
   │     "Edge exists because X. Capacity is $Y/day."     │
   │                                                      │
   │  2. Cheapest possible test                           │
   │     - Pure historical analysis on existing data      │
   │     - No real money, no live infra changes           │
   │                                                      │
   │  3. Statistical validation                           │
   │     - n ≥ 30 examples                                │
   │     - Brier score, hit rate, edge estimate           │
   │     - Compare to a base-rate baseline                │
   │     - Verify edge persists in sub-windows            │
   │                                                      │
   │  4. Forward-test in paper                            │
   │     - 1-2 weeks of paper trading                     │
   │     - Measure live divergence from backtest          │
   │                                                      │
   │  5. Small-stakes live (≤$500)                        │
   │     - 1 week minimum                                 │
   │     - Compare live PnL to forward-test expectations  │
   │                                                      │
   │  6. Scale (only if 5 is positive)                    │
   └──────────────────────────────────────────────────────┘
```

**Each step has a kill criterion:**
- Step 3 fails if the historical Brier > baseline Brier
- Step 4 fails if forward-test paper PnL is negative or trends toward zero
- Step 5 fails if live PnL diverges from forward-test by >25%

If any step fails, **stop**. Don't proceed to a more expensive step
hoping it'll work — that's how the decay strategy ended up costing 3
days of forward-testing for net −$34.

---

## Where to log results

Each alpha source should produce a markdown writeup like
`experiments/decay/docs/STRATEGY_FINDINGS.md` covering:
- Hypothesis and rationale
- Test parameters (n, time window, data source)
- Per-step result with numbers
- Pass/fail for each kill criterion
- Decision: deploy / iterate / archive

Store under `docs/research/<source>.md` for active research, or move to
`experiments/<source>/` once archived.

---

*Research without testing is fiction. Test cheaply, kill fast, scale only what survives.*
