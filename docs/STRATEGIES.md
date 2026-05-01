# Strategy Catalog

Reference doc for every strategy in the codebase: what it does, what it
needs, where it lives, and what status it's in. Companion to
[STRATEGY_FINDINGS.md](STRATEGY_FINDINGS.md) (session results) and
[INFRASTRUCTURE.md](INFRASTRUCTURE.md) (the components).

Status legend:
- 🟢 **Production** — running, generating signals
- 🟡 **Built, validated** — has working backtest, not yet in production
- 🟠 **Built, partial** — code exists but doesn't yet beat baseline
- 🔵 **Researched** — written up but not built
- ⚪ **Identified** — on the list, not started

---

## 🟢 1. Settlement Decay (production)

**Status**: Live in `paper_trade_decay.py` with trail_up entry logic. Backtested
across 4,789 trades/year. Real edge confirmed in live forward test.

**Hypothesis**: Late-stage prediction markets at high implied probability
under-price the safety of the favorite due to favorite-longshot bias —
a well-documented phenomenon in betting market literature ([Snowberg & Wolfers, NBER 2010](https://www.nber.org/system/files/working_papers/w15923/w15923.pdf)).
The bias persists despite many participants because retail bettors
systematically over-bet long shots, leaving small but real premium on the
favorite side near close.

**Mechanics**:
- Buy YES when `mid >= threshold AND time_to_close <= window`
- Buy NO when `mid <= (1 - threshold) AND time_to_close <= window`
- Hold to settlement
- Each trade: 1-3¢ edge on $0.85-0.99 stakes

**Production parameters** (after variant testing):
- `yes_thresholds = [0.75, 0.85, 0.95]` (trail-up ladder)
- `no_thresholds = [0.25, 0.15, 0.05]` (trail-down ladder)
- `max_hours = 12` for daily markets, `max_minutes = 15` for HF
- `max_fraction = 0.05` per fill (because trail_up multiplies fills per market)
- Depth-aware sizing: never take more than 25% of inside ask

**Code**:
- Signal model: [src/signals/settlement_decay.py](../src/signals/settlement_decay.py)
- Live trader: [scripts/paper_trade_decay.py](../scripts/paper_trade_decay.py)
- Backtest: [scripts/backtest_decay.py](../scripts/backtest_decay.py),
  [backtest_decay_hf.py](../scripts/backtest_decay_hf.py),
  [backtest_combined.py](../scripts/backtest_combined.py)
- Variants test: [scripts/backtest_decay_variants.py](../scripts/backtest_decay_variants.py)

**Backtest performance** (audited execution: Kalshi fees + 10% withdrawal +
volume gate + ask-side fills):

| Universe | Sample | N | Win % | Sharpe | Annual P&L |
|---|---|---|---|---|---|
| Daily (sports/weather) | 365d | 386 | 97.4% | +3.08 | $682 |
| HF 15M crypto | 90d | 1,671 | 93.2% | +3.24 | $6,040 |
| HF + trail_up [.75/.85/.95] +NO | 90d | 4,980 | 93.6% | +2.87 | **$8,805** |
| Expanded (BTCD, ETHD, hourly weather) | 90d | 2,669 | 97.4% | +0.14 | $1,528 |
| **Combined** | | **~7,500** | | | **~$11,000** |

**Per-series breakdown** (where the bias actually exists):

| Series | Net $/yr | Verdict |
|---|---|---|
| KXBTCD (daily Bitcoin) | +$771 | ✅ |
| Weather (KXHIGHT*, KXLOWT*) | +$1,200 (combined) | ✅ |
| KXETH (hourly Ethereum) | +$290 | ✅ |
| KXSOLD (daily Solana) | +$145 | ✅ |
| 15M crypto (BTC/ETH/SOL/etc.) | +$6,040 | ✅ |
| **KXBTC** (hourly Bitcoin) | **−$366** | ❌ market efficient |
| **KXETHD** (daily Ethereum) | **−$597** | ❌ market efficient |

**Variants tested** ([scripts/backtest_decay_variants.py](../scripts/backtest_decay_variants.py)):

| Variant | trades/yr | win% | Sharpe | Annual $ |
|---|---|---|---|---|
| Baseline (1 entry/market, YES) | 815 | 93.0% | 2.10 | $2,739 |
| Multi-entry (2-min cooldown) | 1,034 | 92.5% | 1.75 | $2,666 (worse) |
| Last-window only (last 1 min) | 665 | 99.4% | 2.38 | $820 (less $) |
| **Trail-up [.75,.85,.95] +NO** | **4,980** | **93.6%** | **2.87** | **$8,805** ⭐ |
| Trail-up [.70,.80,.90] +NO | 5,370 | 90.3% | 2.46 | $9,487 |

**Capacity**: depth-bound at $200-500 per trade. Strategy maxes out at
$30-50k bankroll regardless of available capital. To scale beyond this,
need additional series or different markets entirely.

**Known issues** (from session audit):
- Trail-up can hit 3x intended per-market exposure when implied gaps past
  multiple thresholds in one tick (HYPE 117ct issue)
- 0.95 threshold may be barely positive-EV — backtest shows it's where
  the market is well-calibrated
- Cross-asset correlation: BTC/ETH/SOL move together, so "diversified
  crypto book" is partly an illusion

---

## 🟡 2. Trail-up Entry Logic

**Status**: Built and validated. Combined with Settlement Decay above.

**Hypothesis**: instead of a single entry per market when implied first
crosses our threshold, take additional entries at progressively higher (or
lower) thresholds. As implied climbs from 0.75 → 0.85 → 0.95, each
crossing represents fresh confirmation with a small incremental edge.

**Why it works**:
- Each threshold step compounds across the universe (4,980 trades vs 815)
- Win rate stays high because higher thresholds have higher implied
  probability (correctly forecasts the actual outcome)
- Small per-trade edges accumulate

**Why other variants don't**:
- **Multi-entry with cooldown**: implied doesn't oscillate above/below a
  single threshold inside a 15-min window — by the time it crosses, it
  usually stays crossed. Cooldown adds nothing.
- **Last-window only**: at very high implied (last 1-2 min), the market is
  well-calibrated. You give up the alpha that lives in the 0.75-0.90 zone.

**Code**: integrated into [src/signals/settlement_decay.py](../src/signals/settlement_decay.py)
and [paper_trade_decay.py](../scripts/paper_trade_decay.py).

---

## 🟠 3. Pure Structural Arbitrage Scanner

**Status**: Built ([scripts/scan_alpha.py](../scripts/scan_alpha.py)),
zero hits in production data — confirmed market makers enforce these
axioms.

**Hypothesis**: Within the same event, prices must satisfy probability
axioms:
- **Monotonicity**: P(value > T1) ≥ P(value > T2) when T1 < T2
- **Partition**: ∑ P(mutually exclusive bins) ≤ 1 (sum of bids)
- **Conditional bounds**: P(A∧B) ≤ min(P(A), P(B))

When violations exist, riskless arbitrage is possible.

**Result**: After fixing parser bugs (k/M/billion suffixes) and a partition
heuristic false positive (Trump late-night posts hour-bins are NOT mutually
exclusive), **zero risk-free arbs across 15,801 liquid markets**. Market
makers enforce these axioms.

**Soft kinks** (mid-curve violations within bid/ask spread) — 5 found in
the catalog, but these are directional plays not riskless arbs:
- Brent crude `>$93` vs `>$93.50` (8pp kink)
- Silver `>$87.99` vs `>$88.99` (7pp)
- Gold `>$5131.99` vs `>$5151.99` (6pp)
- Brent `>$104.99` vs `>$106.99` (5.5pp)
- Coffee `>304.99¢` vs `>309.99¢` (5pp)

**Why no hits**: same scanner that we run is also being run by every other
quant on Kalshi. Free money on the surface gets arbed in seconds.

**Tests**: [tests/test_scan_alpha.py](../tests/test_scan_alpha.py) covers
parser correctness and synthetic genuine-violation detection.

---

## 🟠 4. Cross-Platform Arbitrage (Kalshi ↔ Polymarket)

**Status**: Scanner built ([scripts/scan_cross_platform_arb.py](../scripts/scan_cross_platform_arb.py)).
Research surfaced **$40M+ in arb profits** earned by bots on Polymarket
in the last year. Our v1 scanner found zero clean arbs.

**Hypothesis**: same event priced differently on Kalshi vs Polymarket
gives risk-free arbitrage when:
- Buy YES on platform A + Buy NO on platform B
- Total cost < $1.00 (paid <$1 for guaranteed $1 payout)

**Why our v1 scanner didn't find arbs**: structural mismatch in BTC universe.

| Platform | BTC market structure |
|---|---|
| Kalshi | Narrow daily ranges (`$77,000-77,099.99 on May 1, 2026`) |
| Polymarket | Year-end targets (`reach $80,000 by Dec 31, 2026`) |

These are structurally incompatible. Real cross-platform arb requires:
1. **LLM-based question-semantic matching** (not just keyword + threshold)
2. **Categories where the markets are 1:1** — likely elections, CPI, Fed
   decisions (not crypto where structures differ)
3. **Both-platform execution infrastructure** — Polymarket needs Polygon wallet

**Effort to make real**: 1-2 weeks. Documented as research artifact for now.

---

## 🟠 5. Soft Kink Scanner (directional, not riskless)

**Status**: Built into [scripts/scan_alpha.py](../scripts/scan_alpha.py).
Identifies non-monotone mid-price curves where one side is provably wrong
even if spreads don't strictly invert.

**Hypothesis**: when mid("Above $93.50") > mid("Above $93.00"), one of
those two prints is wrong. Either the lower threshold is underpriced or
the higher is overpriced. Trade the side you have other reasons to trust.

**Output**: 5 candidates currently (commodities markets — Brent, silver,
gold, coffee). Magnitude 5-8pp violations.

**Limitation**: doesn't tell you WHICH side is wrong. Needs a second
signal (LLM reasoning, news, fundamental view) to direction the trade.

---

## 🟠 6. Signal Model Stubs (from Phase 5 spec)

**Status**: 5 signal models scaffolded in `src/signals/`, only `RawLLMSignal`
and `SettlementDecaySignal` actively used in production.

These are designed as **alternative signal models** — each handles a
different LLM failure mode. Most are scaffolded but never trained on real
data.

### 6a. `RawLLMSignal` — baseline 🟡
Pass-through of the LLM's `Edge.kelly_fraction` and `estimated_fair_prob`
without any empirical correction. The baseline every other model must beat.

Code: [src/signals/raw_llm.py](../src/signals/raw_llm.py)

### 6b. `ConsistencyArbSignal` — pure structural 🟠
Scans LLM-grouped markets for axiom violations. The LLM's job here is
purely grouping; calibration is irrelevant.

Code: [src/signals/consistency_arb.py](../src/signals/consistency_arb.py)

Status: built, no live data tested.

### 6c. `CalibratedLLMSignal` — empirical correction 🟠
Fits an isotonic regression on `(LLM_estimated_fair_prob, settled_outcome)`
pairs from resolved history. Applies the calibration map to live LLM
outputs to produce corrected fair-value probabilities.

**This is the highest-unrealized-value signal in the codebase.** We have
1,900+ resolved markets that could train it; just need to run the LLM on
each and pair predictions with outcomes.

Code: [src/signals/calibrated_llm.py](../src/signals/calibrated_llm.py),
trainer at [scripts/fit_calibration.py](../scripts/fit_calibration.py)

Effort to deploy: 2-3 hours + ~$20 in Gemini tokens. Expected upside:
$2-10k/yr if calibration learning works (likely it does — canonical quant move).

### 6d. `BayesianBaseRateSignal` — empirical prior 🟠
Uses LLM retrieval to find analogous resolved markets, computes empirical
base rate, treats as prior, updates with current market price as Bayesian
likelihood. Posterior grounded in actual outcomes.

Code: [src/signals/bayesian_base_rate.py](../src/signals/bayesian_base_rate.py)

Status: built, no production data tested.

### 6e. `CoherenceRegressionSignal` — residual-based 🟠
Predicts each market's price from the prices of LLM-grouped related
markets via ridge regression. Residuals = signal.

Code: [src/signals/coherence_regression.py](../src/signals/coherence_regression.py)

Status: built, no production data tested.

---

## ⚪ 7. ML Direction Predictor (Coinbase + Binance data)

**Status**: Identified, not started. Highest-upside next strategy.

**Hypothesis**: 15-minute crypto markets settle based on Coinbase price.
The price discovery happens on CEX order books, not on Kalshi. By
ingesting Coinbase/Binance L2 order book + trade tape + funding rates,
we can predict 15-min direction and use it as a confidence filter on
top of settlement decay.

**Architecture**:
```
Coinbase/Binance WebSocket → Feature engineering → XGBoost
  - L2 order book              - OFI 1m/5m/15m       trained on resolved
  - Trade tape                  - Realized vol          Kalshi outcomes
  - Funding rates               - Price momentum
  - Mark price                  - Cross-ex basis        ↓
                                                       Predict P(YES)
                                                       Filter trades
```

**Two integration patterns**:

| Pattern | Approach | Upside | Risk |
|---|---|---|---|
| **A: Confidence filter on existing strategy** | ML predicts direction; only take settlement-decay trades that agree | +30-50% to existing strategy | Lower (still grounded in favorite-longshot) |
| **B: Standalone directional ML model** | Pure ML prediction, no anchor | Could 5-10x current returns | Higher (untested signal source, may overfit) |

**Pattern A is the right starting move** — additive, preserves the working
strategy as the floor.

**Effort**: 3-4 week serious project. Realistic upside +$5-15k/year on top
of current $11-15k.

**What to build**:
1. Coinbase + Binance WebSocket ingestion
2. Feature pipeline (OFI, vol, funding, basis)
3. XGBoost trainer on resolved Kalshi outcomes
4. Pattern A integration with `paper_trade_decay.py`
5. Live forward test

---

## ⚪ 8. Polymarket Whale Copy-Trade

**Status**: Identified, not started.

**Hypothesis**: Polymarket positions are public on-chain (Polygon RPC).
Track top-performing wallets, mirror their trades with a lag.

**Effort**: 4-6 hours to build basic version.

**Risk**: lag means you fill at worse prices than the whale. Need careful
backtest before deploying.

---

## ⚪ 9. Weather-Specialist HF Strategy

**Status**: Identified. Per the research, one Polymarket bot reportedly
made $24k on London weather, another $65k across cities.

**Hypothesis**: weather forecast accuracy is well-known. Hourly weather
markets often misprice tail probabilities (extreme highs/lows). Stale
quote elimination + forecast-aligned bets could be additive to settlement
decay.

**Effort**: 1-2 hours per city, build per-city decay parameters and
integrate forecast data feed (NOAA / OpenWeatherMap).

---

## ⚪ 10. News/Sentiment Pipeline

**Status**: Identified. Major prediction-market moves are news-driven
(Fed decisions, election results, geopolitical events). Currently we
ignore news entirely.

**Hypothesis**: integrating Twitter/Reddit sentiment + news headlines
gives ~5-15s lead time on retail-driven moves. For HF crypto markets,
this could be a directional signal.

**Effort**: 1-2 weeks for production-quality pipeline.

---

## ⚪ 11. Market Making (FYI — wrong tool for our latency)

**Status**: Identified, ruled out for our infrastructure.

**Why it's profitable**: continuously post bid + ask around fair value,
capture spread. Per the research, bots making millions doing this on
Polymarket.

**Why we can't**: needs sub-second execution. Our latency is 5-30s. Wrong
tool entirely. If we ever wanted to build this, we'd need:
- Co-located server next to Kalshi/Polymarket exchanges
- Custom WebSocket clients with microsecond-grade latency
- Hardware-accelerated quote updates

Not worth pursuing unless we're going professional.

---

## Strategy combination matrix

For reference, what stacks with what:

| | Settlement Decay | Trail-up | ML Direction | Cross-Platform Arb | Calibrated LLM |
|---|---|---|---|---|---|
| Settlement Decay | self | combined | confidence filter | independent | combined |
| Trail-up | combined | self | confidence filter | independent | combined |
| ML Direction | filter | filter | self | independent | filter |
| Cross-Platform Arb | independent | independent | independent | self | independent |
| Calibrated LLM | combined | combined | filter | independent | self |

"Combined" = both can run on same trade. "Filter" = one gates the other.
"Independent" = different data, different markets, fully diversified.

---

## Recommended ordering for tomorrow+

1. **Fix the two bugs from today's audit** (state.py + paper_trade reporting)
2. **Per-market exposure cap** (today's HYPE 117ct issue)
3. **Train CalibratedLLMSignal** — biggest unrealized value in current code
4. **Build ML Direction Predictor** (Coinbase ingestion → XGBoost)
5. **Cross-Platform Arb v2** with LLM-based question matching
6. **Whale copy-trade + weather specialist** as side strategies
