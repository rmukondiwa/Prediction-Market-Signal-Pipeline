# System Roadmap — Kalshi Quant Trading Engine

## Vision

An end-to-end automated trading system on Kalshi that uses LLM-driven cross-market inference to identify *related* markets and structural relationships, then applies quantitative signal models on top to produce calibrated trading decisions. Sized with quarter-Kelly under hard risk limits, backtest-validated, paper-traded before live, monitored continuously.

## Foundational thesis

Prediction markets are uniquely suited to LLM analysis because every contract is a literal proposition with a binary outcome and an explicit price. This is structurally different from equities/futures, where prices reflect compressed valuations that LLMs cannot reason about cleanly.

But: **the LLM is reliable as a retrieval and structuring layer, not as a probability calibrator.** It can identify causally related markets, decompose conditional structure, and detect axiom violations. It cannot reliably put a calibrated number on an outcome. Therefore the system separates the two roles:

- **LLM layer** (existing Phase 4): retrieves related markets, identifies structural relationships, generates analytical scaffolding
- **Signal model layer** (Phase 5, Stage 7.5): empirical calibration on top of LLM output — consistency arbitrage, isotonic-calibrated LLM probabilities, base-rate Bayesian update, cross-market coherence regression

The trading signal is a `CalibratedEdge` from a signal model, not an `Edge` from the LLM directly.

## Current state — honest assessment

What Raphael shipped through Phase 4 is a **signal generation pipeline**, not a trading system. Roughly 30% of the full stack. There is currently no order placement, no portfolio state, no risk management, no backtest, no paper trading, no live monitoring loop, no settlement tracking, no P&L attribution. The `InferenceReport.suggested_edges[]` is research output — actionable in spirit but not in code.

This document maps the full stack and sequences what's left.

## Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 9: Operations         scheduler · monitoring · alerts      │  TODO
├──────────────────────────────────────────────────────────────────┤
│ Layer 8: Execution          trading client · paper · live        │  TODO
├──────────────────────────────────────────────────────────────────┤
│ Layer 7: Portfolio & Risk   state · risk gates · sizing          │  TODO
├──────────────────────────────────────────────────────────────────┤
│ Layer 6: Validation         backtest framework · cache · metrics │  TODO
├──────────────────────────────────────────────────────────────────┤
│ Layer 5: Signal models      consistency · calibration · base     │  TODO
│                             rate · coherence regression          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 4: Historical Archive snapshotter · catalog · resolutions  │  TODO
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: Cross-market LLM   catalog · embed · retrieve · infer   │  DONE (Phase 4)
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: Single-market LLM  extractor · LLM insight              │  DONE (Phase 2)
├──────────────────────────────────────────────────────────────────┤
│ Layer 1: Ingestion          WS · parser · normalizer · Redis     │  DONE (Phase 1)
└──────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Ingestion (DONE)
Live WebSocket → parser → normalizer → Redis Streams. RSA-PSS auth, reconnect, exponential backoff. See [src/ingestion/kalshi/](../src/ingestion/kalshi/).

### Layer 2 — Single-market signal (DONE)
Redis snapshot → OpenAI structured output → `InsightReport`. See [src/insight/](../src/insight/).

### Layer 3 — Cross-market signal (DONE)
Catalog → embeddings → FAISS → vector search → LLM rerank → cross-market inference → `InferenceReport` with `suggested_edges`, `detected_mispricings`, `derived_probabilities`. See [src/catalog/](../src/catalog/), [src/context/](../src/context/), [src/inference/](../src/inference/).

### Layer 4 — Historical archive (TODO, Stage 6)
Daily snapshots of Redis streams, catalog, and market resolutions to parquet. **Passive — start collecting now even before backtest is built**, because every day delayed is a day of book history we can never recover.

### Layer 5 — Signal models (TODO, Stage 7.5)
Quantitative methods that consume the LLM `InferenceReport` and produce `CalibratedEdge` outputs. Five planned implementations: `RawLLMSignal` (baseline), `ConsistencyArbSignal` (axiom violations), `CalibratedLLMSignal` (isotonic correction), `BayesianBaseRateSignal` (empirical prior + Bayesian update), `CoherenceRegressionSignal` (residuals from cross-market regression). Each handles a different failure mode of the LLM. Likely production signal is an ensemble of two or three.

### Layer 6 — Validation & backtest (TODO, Stage 7)
Replay resolved markets through inference + signal models with a disk-cached LLM. Honest fill simulation. Metrics: Brier score on each signal model's `estimated_fair_prob` vs. `market_baseline_brier`, hit rate by confidence, edge decay, Sharpe, P&L net of fees. Side-by-side comparison of all signal models. **Until at least one signal model beats the market baseline, no live capital.**

### Layer 7 — Portfolio & risk (TODO, Stage 8)
Redis-backed source of truth for cash, positions, working orders, P&L. Risk manager gates every order: per-market cap, total exposure, correlated-bet cap, daily loss limit, kill switch. Quarter-Kelly sizer with risk-manager scaling. Consumes `CalibratedEdge`, not raw LLM `Edge`.

### Layer 8 — Execution (TODO, Stages 9–10)
Kalshi REST/WS trading client (extends existing auth). Order manager with limit-order placement, cancel-replace, partial fill handling. Paper executor (simulated fills against live book) → Live executor (real fills). Paper before live, no exceptions. `Fill` records carry the originating `signal_model` for attribution.

### Layer 9 — Operations (TODO, Phase 6)
Scheduler that runs inference on a watchlist on a cadence. Health checks. Prometheus metrics + dashboard. Alert on anomalous P&L, on inference failures, on auth expiry. Prompt versioning + A/B framework for iterating on the LLM logic.

## Phase plan & sequencing

| Phase | Stage | What | Depends on | Time est. | Risk |
|-------|-------|------|------------|-----------|------|
| 5b | 6 | Forward archive | nothing — start NOW | 2–3 days | low |
| 5a | 7 | Backtest framework + cache + RawLLM baseline | Kalshi candle API | 1 week | medium |
| 5a' | 7.5 | Concrete signal models | 7 framework | 1–2 weeks | medium |
| 5c | 8 | Portfolio + risk | nothing | 4–5 days | low |
| 5d | 9 | Paper executor | 8 + 7.5 + Kalshi trading auth | 1 week | medium |
| 5e | 10 | Live trading | 7.5 results + 9 + good metrics | 1 week | **high** |
| 6 | 11 | Operations layer | 5e | ongoing | medium |

**Critical sequencing**: Stage 6 is independent and should start today. Stages 7 and 8 are independent of each other and can run in parallel. Stage 7.5 needs 7's framework but its individual signal models can be built independently of each other (good for parallelism). Stage 9 needs 8 + at least one signal model from 7.5. Stage 10 should not start until at least one signal model in 7.5 beats `market_baseline_brier` on the backtest sample.

## Open questions

These need answers before some stages can be fully scoped:

1. **Starting capital + risk tolerance.** Sets the per-market cap, total exposure cap, daily loss limit. Without numbers we can only build configurable scaffolding.
2. **Watchlist strategy.** Run inference on every market every hour? On a curated event watchlist? Triggered by news? This is a real product decision — affects compute spend and signal quality.
3. **What time granularity for backtest replay.** Hourly seems right for LLM signals (news-driven, not microstructure). Confirm before building.
4. **Have we accumulated any Redis archive yet?** If not, Stage 7 must rely entirely on Kalshi's candlestick API (lower fidelity). Stage 6 starts the clock for higher-fidelity backtests later.
5. **Kalshi fee structure exact details.** Affects net P&L calculation in backtest and live. Need to read current rate card and bake into models.
6. **Compliance / KYC / account standing for trading.** Not a code question but blocks Stage 10.

## Risks

- **Signal looks good in backtest, fails live.** Most common failure mode for systematic strategies. Mitigations: (1) honest fill modeling, (2) walk-forward validation rather than single in-sample fit, (3) paper-trade for a meaningful window before live.
- **LLM non-determinism.** Even at temperature 0, Groq has small variance. Backtest cache hides this; live re-runs surface it. Track distribution of `estimated_fair_prob` across repeated calls on same input as a sanity check.
- **Catalog look-ahead bias.** Until Stage 6 has accumulated daily catalog snapshots, backtest uses today's catalog for past timestamps. Acknowledged bias; expect signal degradation when fixed.
- **Settlement risk.** Kalshi has settled markets retroactively contested or recategorized in rare cases. Track this — don't assume settlement is permanent until N days post-close.
- **Concentration risk.** LLM tends to find correlated edges (multiple markets on the same event). Without a correlation cap, "5 independent edges" might be one bet with 5x variance. Stage 8 must address this explicitly.
- **Kalshi API changes.** Public API has been stable but not contractually so. Forward archive (Stage 6) gives us insurance — even if Kalshi changes endpoints, our parquet history is ours.
- **Capital risk.** Real money goes in at Stage 10. Hard ceilings, kill switch, daily loss limit, manual approval for first N days. No "set it and forget it" until proven.

## What this roadmap does NOT cover

- ML signal generation beyond LLM (statistical features, gradient boosting on engineered features, etc.) — possible Phase 7
- Multi-exchange (Polymarket, etc.) — out of scope for now
- Front-end / web UI — CLI is fine
- Slippage models more sophisticated than midpoint + parameter — can refine later if needed
- HFT-style microstructure trading — wrong tool for an LLM-driven system anyway
