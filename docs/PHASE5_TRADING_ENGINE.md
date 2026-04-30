# Phase 5: Trading & Backtest Engine

Same shape as the existing `CLAUDE.md`. Read that first if you haven't — this doc reuses its conventions, its model patterns, and its file-by-file specification style.

## What this phase is

Five stages turn the existing LLM-driven retrieval/inference pipeline into a **trading system**, with one defining architectural choice: the LLM is treated as a retrieval and structuring layer, **not** a probability calibrator. Quantitative signal models on top of LLM output produce the actual trading number. The phase delivers a forward-collected data archive, a backtest framework, a library of signal models, a portfolio/risk layer, and a paper-trading executor. Live execution is scoped at the end but gated on backtest results.

## Thesis: LLM as retrieval, quant as calibration

The LLM is reliable at:
- Reading propositional markets and grouping causally related ones
- Decomposing conditional probability structure (P(A|B), temporal dependencies)
- Identifying axiom violations (P(by June) > P(by December) is impossible)
- Generating analytical scaffolding ("here's the relationship, here's where to look")

The LLM is unreliable at:
- Putting calibrated absolute probabilities on outcomes
- Distinguishing strong causation from plausible-sounding co-occurrence
- Estimating tail risk

**Implication for design**: the trading signal is NOT `Edge.estimated_fair_prob` from the LLM. It is the output of a `SignalModel` that takes the LLM's `InferenceReport` as input and produces a `CalibratedEdge` using empirical methods — consistency arbitrage, calibration learning on resolved outcomes, base-rate priors, coherence regression. The backtester's job is to compare signal models head-to-head against the market-implied baseline.

This decoupling lets us swap or A/B test signal models without touching retrieval, and lets the LLM keep its job (semantic structuring) without being asked to do something it's bad at (numerical calibration).

## What already exists (Phase 1–4)

- Live ingestion → Redis Streams ([src/ingestion/](../src/ingestion/))
- Single-market insight ([src/insight/](../src/insight/))
- Catalog + embeddings + FAISS ([src/catalog/](../src/catalog/))
- Cross-market retrieval, rerank, structured inference ([src/context/](../src/context/), [src/inference/](../src/inference/))
- CLI: `python -m scripts.build_index`, `python infer.py TICKER`

## Architecture

```
COLD:     Kalshi REST → catalog → embed → FAISS index            (existing, daily)
HOT:      Redis snapshot + index → retrieve → rerank → infer     (existing, per-call)

NEW PASSIVE (always running):
ARCHIVE:  Redis streams + catalog + resolutions → parquet/day    (Stage 6)

NEW ACTIVE:
SIGNAL:   InferenceReport → SignalModel(1..N) → CalibratedEdge   (Stage 7 / 7.5)
BACKTEST: archive (or candles) → cached inference →              (Stage 7)
          signal model(s) → fill sim → portfolio replay → metrics

TRADE:    CalibratedEdge → risk gate → sizer →                   (Stages 8–9)
          order manager → paper/live executor → portfolio state
```

Signal models sit between the LLM and the trading machinery as a clean abstraction. Same event-driven principle as Phase 4: every stage emits a typed artifact downstream consumers can replace.

## Stages in this phase

| Stage | Scope | Independence |
|-------|-------|-------------|
| 6 | Forward-collected archive | Independent; **start first** |
| 7 | Backtest framework + cache + `SignalModel` protocol + RawLLM baseline | Independent of 6 |
| 7.5 | Concrete signal models (consistency, calibrated, Bayesian, coherence) | Needs 7 framework |
| 8 | Portfolio + risk | Independent |
| 9 | Paper-trading executor | Needs 8 + 7.5 |
| 10 | Live trading | Gated on backtest results — **scoped, not built in this phase** |

---

## Stage 6: Forward-collected archive

**Why first**: every day not collecting is a day of book history we can't recover. Independent of every other stage. Can run in production tonight.

### Files to create

**`src/storage/__init__.py`** — empty.

**`src/storage/snapshotter.py`**

Two async functions:

- `drain_streams(redis_client, since: dict[str, str], output_dir: Path) -> dict[str, str]`
  Reads each Redis Stream (`market_events`, `trade_events`, `orderbook_events`) from `since[stream_name]` to `+`. Writes to `output_dir/{stream}.parquet`. Returns updated `since` dict (last id read per stream) for the next checkpoint. Use `pyarrow` for parquet write — supports append.
- `run_snapshot(redis_config, archive_root: Path = Path("data/archive"))`
  Daily entry point. Loads checkpoint from `archive_root/.checkpoint.json`. Determines today's directory: `archive_root/YYYY-MM-DD/`. Calls `drain_streams`. Writes new checkpoint. Logs counts per stream.

**`src/storage/catalog_archive.py`**

`snapshot_catalog(base_url: str, archive_root: Path)` — calls existing `build_catalog()` from `src/catalog/fetcher.py`, writes to `archive_root/YYYY-MM-DD/catalog.parquet` (Pydantic dump → list of dicts → pyarrow Table → parquet).

**`src/storage/resolution_tracker.py`**

`track_resolutions(base_url: str, archive_root: Path)` — for each ticker observed in any prior catalog snapshot, query `GET /markets/{ticker}` and record:
```
ResolutionRecord(ticker, status, result, settlement_value, close_time, settled_time, last_checked)
```
Append to `archive_root/resolutions.parquet`. Skip tickers already settled. This is a daily delta — cheap.

**`scripts/archive_daily.py`**

```
Usage: python -m scripts.archive_daily

Steps:
  1. Snapshot Redis streams since last checkpoint
  2. Snapshot catalog
  3. Update resolution status for known tickers
  4. Log summary: streams=N catalog=M resolved_today=K
```

Schedule via cron or systemd timer at e.g. 02:00 UTC.

### Acceptance criteria

- [ ] After 7 days running, `data/archive/` contains 7 dated subdirectories
- [ ] Each subdirectory has `market_events.parquet`, `trade_events.parquet`, `orderbook_events.parquet`, `catalog.parquet`
- [ ] `data/archive/resolutions.parquet` grows daily as markets settle
- [ ] Re-running on the same day is idempotent

### Dependencies

```
pyarrow>=15.0.0
pandas>=2.1.0
```

---

## Stage 7: Backtest framework + signal model protocol

**Goal**: a backtest engine that takes any `SignalModel` and produces comparable metrics. Includes the cache, the replayer, the fill simulator, the metrics module, and the `SignalModel` protocol with one baseline implementation (`RawLLMSignal`). Stage 7.5 adds the additional signal models.

### Files to create

**`src/backtest/__init__.py`** — empty.

**`src/backtest/cache.py`** — disk-cached LLM responses.

```python
class InferenceCache:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, model: str, temperature: float, prompt_version: str) -> str:
        payload = json.dumps(
            {"p": prompt, "m": model, "t": temperature, "v": prompt_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, prompt, model, temperature, prompt_version) -> dict | None: ...
    def put(self, prompt, model, temperature, prompt_version, response: dict) -> None: ...
```

Storage: `cache_dir/{first_two_chars_of_hash}/{full_hash}.json`. Sharded to avoid 100k files in one directory.

Each entry: `{"prompt": ..., "model": ..., "temperature": ..., "prompt_version": ..., "response": ..., "tokens_in": ..., "tokens_out": ..., "stored_at": ...}`.

**Cache wrapper changes** in `src/context/reranker.py` and `src/inference/engine.py`:
- Add a `PROMPT_VERSION` constant in each module. Bump on prompt changes.
- Accept optional `cache: InferenceCache | None = None`.
- When `cache` is provided: check before LLM call; store after. Force `temperature=0`.
- Live code passes `None` and behaves exactly as today.

### Signal model protocol

**`src/signals/__init__.py`** — empty.

**`src/signals/models.py`**

```python
class CalibratedEdge(BaseModel):
    """Trading signal output by a SignalModel. Distinct from `Edge`
    (which is LLM-asserted): probabilities here are intended to be
    empirically grounded."""
    ticker: str
    title: str
    side: str                    # "yes" | "no"
    estimated_fair_prob: float   # calibrated by the signal model
    current_implied_prob: float
    edge_pp: float               # estimated_fair - implied (probability points)
    confidence: float            # 0..1, model-specific scoring
    kelly_fraction: float        # quarter-Kelly, clamped 0..0.25
    thesis: str
    source_signal_model: str     # for attribution
    metadata: dict = {}          # model-specific extras

class HistoricalContext(BaseModel):
    """Read-only access to historical data signal models may need."""
    archive_root: Path
    resolutions: dict[str, ResolutionRecord] = {}  # lazy-loaded
    candles: dict[str, list[Candle]] = {}          # lazy-loaded
```

**`src/signals/protocol.py`**

```python
class SignalModel(Protocol):
    name: str

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        ...
```

Stateless protocol. Models that need parameters or trained state load them at construction time.

**`src/signals/raw_llm.py`** — Stage 7 baseline; pass-through.

```python
class RawLLMSignal:
    name = "raw_llm"

    def signals(self, focus, context, llm_report, history) -> list[CalibratedEdge]:
        # Translate each LLM Edge into a CalibratedEdge with no transformation.
        # estimated_fair_prob copied from LLM. confidence mapped from "low/medium/high".
        # This is the baseline every other signal model must beat.
        ...
```

### Backtest core

**`src/backtest/models.py`**

```python
class BacktestConfig(BaseModel):
    start_date: datetime
    end_date: datetime
    granularity: str = "1h"
    universe: list[str] | None
    starting_capital: float = 10_000.0
    fill_model: str = "midpoint"
    slippage_per_contract: float = 0.01
    fee_per_contract: float = 0.07   # placeholder; verify Kalshi rate card
    risk_limits: RiskLimits
    signal_models: list[str]         # by name; multiple → side-by-side comparison

class SimulatedFill(BaseModel):
    ticker: str
    side: str
    contracts: int
    price: float
    fee: float
    timestamp: datetime
    signal_model: str                # which model produced the edge
    edge_thesis: str

class SignalModelMetrics(BaseModel):
    name: str
    n_decisions: int
    n_fills: int
    pnl_realized: float
    pnl_unrealized: float
    by_confidence: dict[str, dict]
    brier_score: float               # on signal model's calibrated probs
    sharpe: float
    max_drawdown: float
    edge_decay_curve: list[tuple[int, float]]
    fills: list[SimulatedFill]

class BacktestReport(BaseModel):
    config: BacktestConfig
    market_baseline_brier: float     # Brier of "trust market price" baseline
    per_signal: dict[str, SignalModelMetrics]
```

**`src/backtest/replayer.py`**

`build_snapshot_at(ticker, t, candle_data) -> MarketSnapshot` — reconstructs a `MarketSnapshot` using Kalshi candlestick API (close ± half-spread approximation). Documented bias; refine with Stage 6 archive once ≥60 days accumulated.

`load_catalog_at(t, archive_root) -> list[CatalogMarket]` — Stage 6 daily snapshots if available, else current catalog with warning.

`fetch_candles(ticker, start, end) -> list[Candle]` — wraps `GET /markets/{ticker}/candlesticks` with `retry_with_backoff()`.

**`src/backtest/fill_simulator.py`**

```python
class FillSimulator:
    def __init__(self, model: str, slippage: float, fee: float): ...

    def simulate(self, edge: CalibratedEdge, snapshot: MarketSnapshot,
                 t: datetime) -> SimulatedFill | None:
        # midpoint: fill at mid + slippage on take side
        # trade_match: scan trades within ±5min of t for cross
        # return None if no fill
```

Start with midpoint. Implement `trade_match` once Stage 6 tape is reliable.

**`src/backtest/metrics.py`**

Pure functions:
- `brier_score(predicted_probs, outcomes) -> float` — on the signal model's `estimated_fair_prob`, NOT the LLM's
- `market_baseline_brier(implied_probs, outcomes) -> float` — must beat this to claim alpha
- `hit_rate_by_confidence(fills, outcomes) -> dict[str, float]`
- `sharpe(daily_returns) -> float` (annualized assuming 365 trading days — prediction markets trade weekends)
- `max_drawdown(equity_curve) -> float`
- `edge_decay(fills, outcomes, hours: list[int]) -> list[tuple[int, float]]`

**`src/backtest/runner.py`**

```python
async def run_backtest(
    config: BacktestConfig,
    signal_models: dict[str, SignalModel],
    cache: InferenceCache,
) -> BacktestReport:
    universe = config.universe or load_resolved_markets_in_window(...)
    history = HistoricalContext(archive_root=Path("data/archive"))

    # One isolated portfolio per signal model
    portfolios = {
        name: PortfolioState.empty(config.starting_capital, env=f"bt:{run_id}:{name}")
        for name in signal_models
    }
    risk = RiskManager(config.risk_limits)
    sizer = QuarterKellySizer()
    simulator = FillSimulator(config.fill_model, config.slippage_per_contract, config.fee_per_contract)

    for ticker in universe:
        for t in walk_timesteps(ticker, config.start_date, config.end_date, config.granularity):
            snapshot = build_snapshot_at(ticker, t, ...)
            catalog = load_catalog_at(t, ...)
            context = await retrieve_and_rerank(snapshot, catalog, cache=cache)
            llm_report = await run_inference(snapshot, context, cache=cache)

            # Run each signal model in isolation
            for name, model in signal_models.items():
                edges = model.signals(snapshot, context, llm_report, history)
                for ce in edges:
                    decision = risk.check(ce, portfolios[name])
                    if not decision.approved:
                        continue
                    contracts = sizer.size(ce, portfolios[name].cash, decision, current_price=...)
                    fill = simulator.simulate(ce, snapshot, t)
                    if fill is not None:
                        portfolios[name].apply(fill)
                portfolios[name].mark_to_market(t, current_prices=...)

        for name, pf in portfolios.items():
            pf.settle(ticker, settlement_value=fetch_settlement(ticker), t=...)

    return assemble_report(config, portfolios, signal_models)
```

Same LLM call sequence drives all signal models — the cache makes parallel comparison cheap.

**`scripts/run_backtest.py`**

```
Usage:
  python -m scripts.run_backtest --config configs/default_backtest.yaml
                                 --signal-models raw_llm,calibrated_llm,consistency_arb
                                 --start 2026-01-01 --end 2026-03-31
                                 --output reports/bt_20260430.json

Loads config, instantiates each named signal model, runs backtest, writes
BacktestReport JSON, prints comparison table:

  Model              | Brier  | Sharpe | MaxDD  | Trades | PnL
  market_baseline    | 0.215  | -      | -      | -      | -
  raw_llm            | 0.241  | -0.12  | -8.4%  | 47     | -$214
  calibrated_llm     | 0.208  | +0.34  | -3.1%  | 47     | +$182
  consistency_arb    | 0.198  | +1.21  | -0.5%  |  6     | +$48
```

**`scripts/fetch_resolutions.py`** — backfill helper. Pulls settlement data without depending on Stage 6 having been running.

### Reproducibility requirements (non-negotiable)

1. **Temperature = 0** for all LLM calls when running through the cache.
2. **`PROMPT_VERSION` constants** in `reranker.py` and `engine.py`, included in cache key.
3. **Seeded RNG** for any randomization (e.g., reranker candidate shuffle), keyed on snapshot timestamp.
4. **Pinned model strings** in cache key. Don't rely on Groq aliases that may route to different model versions.
5. **Signal models are stateless or load deterministic state** at construction. No randomness in signals().

### Acceptance criteria

- [ ] `python -m scripts.run_backtest --signal-models raw_llm` produces a `BacktestReport`
- [ ] Report contains `market_baseline_brier` and `per_signal[raw_llm]` with all metrics
- [ ] Re-running with same cache produces identical results
- [ ] First-run cache miss rate ~100%; second-run hit rate ~100%
- [ ] Backtest covers ≥30 settled markets, ≥50 simulated fills total

### Dependencies

```
pandas>=2.1.0   (already in Stage 6)
scikit-learn>=1.3.0   (for Stage 7.5 calibration)
```

---

## Stage 7.5: Concrete signal models

**Goal**: four signal models that each handle calibration without trusting the LLM's absolute probabilities. Each is a thesis about *how* to extract real signal from LLM retrieval.

### Files to create

**`src/signals/consistency_arb.py`**

`ConsistencyArbSignal` — scans LLM-grouped markets for axiom violations.

Checks:
- **Monotonicity**: P(by T1) > P(by T2) when T1 < T2 (deadline-stacked markets)
- **Total probability**: ∑P(mutually exclusive Aᵢ) > 1
- **Conditional bounds**: P(A∧B) > min(P(A), P(B)) — a market on a joint event priced higher than either marginal

When a violation is detected, emit two `CalibratedEdge`s — short the overpriced, long the underpriced — both with high confidence (the edge is structural, not predictive).

The LLM's job here is purely grouping: "these N markets cover the same event tree." Calibration is irrelevant. This model produces the cleanest signals but rare events — expect <5% of inferences to trigger anything.

**`src/signals/calibrated_llm.py`**

`CalibratedLLMSignal` — fits an isotonic regression on `(LLM_estimated_fair_prob, settled_outcome)` pairs from resolved history. Applies the calibration map to new LLM outputs.

```python
class CalibratedLLMSignal:
    name = "calibrated_llm"

    def __init__(self, calibration_map_path: Path):
        # load fitted IsotonicRegression from disk
        ...

    def signals(self, focus, context, llm_report, history) -> list[CalibratedEdge]:
        for edge in llm_report.suggested_edges:
            calibrated_prob = self.iso.predict([edge.estimated_fair_prob])[0]
            # only trade if calibrated edge still exists after correction
            ...
```

Training script: `scripts/fit_calibration.py` — loads resolved markets from `data/archive/resolutions.parquet`, runs LLM (with cache) on each at multiple historical timestamps, pairs predictions with outcomes, fits isotonic regression, saves to `data/calibration_map.pkl`. Re-fit periodically.

This is the canonical move: empirical correction of a biased estimator. Likely the strongest single model.

**`src/signals/bayesian_base_rate.py`**

`BayesianBaseRateSignal` — uses LLM retrieval to find analogous *resolved* markets, computes empirical base rate, treats it as prior, updates with current market price.

```python
class BayesianBaseRateSignal:
    name = "bayesian_base_rate"

    def signals(self, focus, context, llm_report, history) -> list[CalibratedEdge]:
        analogues = find_analogous_resolved(focus, context, history)  # uses semantic similarity
        if len(analogues) < 5:
            return []   # not enough prior data
        base_rate = sum(a.outcome for a in analogues) / len(analogues)
        # Bayesian update: posterior ∝ prior × likelihood
        # likelihood = market_price as estimator with noise model
        posterior = bayesian_update(prior=base_rate, market_implied=focus.implied_prob)
        # emit CalibratedEdge if posterior diverges from market by ≥ threshold
        ...
```

Posterior is grounded in *actual outcomes*, not LLM speculation. Best when there are dense analogues (recurring market types like "will X price be above Y by date").

**`src/signals/coherence_regression.py`**

`CoherenceRegressionSignal` — predicts focus market price from related-market prices via OLS or ridge regression. Residuals = signal.

```python
class CoherenceRegressionSignal:
    name = "coherence_regression"

    def signals(self, focus, context, llm_report, history) -> list[CalibratedEdge]:
        if len(context) < 4:
            return []
        # X = related markets' prices over recent window
        # y = focus market's price over same window
        # fit ridge, predict expected focus price, compare to actual
        # residual = predicted - actual = signal direction
        ...
```

Captures "everything moved except this one" cases. The LLM's role: tell us which markets to put in the regression matrix.

### Acceptance criteria

- [ ] All four models implement the `SignalModel` protocol
- [ ] `RawLLMSignal` baseline runs in backtest and produces fills
- [ ] At least one signal model (`CalibratedLLMSignal` or `ConsistencyArbSignal`) beats `market_baseline_brier` on the backtest sample
- [ ] `scripts/fit_calibration.py` produces a saved isotonic map; loading it in `CalibratedLLMSignal` is idempotent
- [ ] Comparison table in `scripts/run_backtest.py` output ranks signal models by Brier score

### Dependencies

```
scikit-learn>=1.3.0   (already in Stage 7)
```

---

## Stage 8: Portfolio + risk management

**Why now**: needed before any executor (paper or live) places orders. Independent of Stages 6–7.5.

### Files to create

**`src/portfolio/__init__.py`** — empty.

**`src/portfolio/models.py`**

```python
class Position(BaseModel):
    ticker: str
    side: str
    contracts: int
    avg_cost: float
    opened_at: datetime
    last_updated: datetime

class WorkingOrder(BaseModel):
    order_id: str
    ticker: str
    side: str
    contracts: int
    limit_price: float
    placed_at: datetime
    status: str
    filled_contracts: int = 0

class Fill(BaseModel):
    fill_id: str
    order_id: str | None
    ticker: str
    side: str
    contracts: int
    price: float
    fee: float
    timestamp: datetime
    signal_model: str   # NEW: track which model originated this fill

class RiskLimits(BaseModel):
    max_per_market_usd: float = 500.0
    max_total_exposure_usd: float = 5_000.0
    max_correlated_exposure_usd: float = 1_500.0
    daily_loss_limit_usd: float = 500.0
    max_kelly_fraction: float = 0.25
    kill_switch: bool = False

class RiskDecision(BaseModel):
    approved: bool
    reason: str
    scale_factor: float = 1.0
```

**`src/portfolio/state.py`** — Redis-backed source of truth.

Operations (atomic via WATCH/MULTI or Lua):
- `get_snapshot()`, `apply_fill(fill)`, `add_working_order(order)`, `update_order(...)`, `mark_to_market(prices)`, `settle(ticker, settlement_value, t)`

Redis layout:
```
portfolio:{env}:cash                   STRING (cents)
portfolio:{env}:positions              HASH ticker → JSON Position
portfolio:{env}:working_orders         HASH order_id → JSON WorkingOrder
portfolio:{env}:fills                  STREAM (audit log, includes signal_model)
portfolio:{env}:realized_pnl           STRING (cents)
portfolio:{env}:daily_pnl:YYYY-MM-DD   STRING (cents)
```

`{env}` is `paper`, `live`, or `bt:<run_id>:<signal_model>` for backtest isolation.

**`src/portfolio/risk.py`**

```python
class RiskManager:
    def __init__(self, limits: RiskLimits): ...

    def check(self, edge: CalibratedEdge, portfolio: PortfolioSnapshot,
              context_markets: list[ContextMarket]) -> RiskDecision:
        # kill_switch
        # daily_loss_limit
        # per-market cap
        # total exposure cap
        # correlated exposure cap (shared event_ticker; LLM-flagged correlations)
        # cap kelly_fraction at limits.max_kelly_fraction
        # return reject or scale_factor < 1.0
        ...
```

Correlation heuristics:
1. Same `event_ticker` → treat as one position. Max one open edge per event.
2. Multiple `CalibratedEdge`s in current batch with overlapping `thesis` keywords → cap aggregate exposure.

**`src/portfolio/sizer.py`**

```python
class QuarterKellySizer:
    def size(self, edge: CalibratedEdge, available_cash: float,
             decision: RiskDecision, current_price: float) -> int:
        # cash_to_deploy = available_cash * edge.kelly_fraction * decision.scale_factor
        # contracts = floor(cash_to_deploy / current_price)   # contracts pay $1 at settlement
        # clamp to [0, max_contracts_per_order]
        return contracts
```

### Acceptance criteria

- [ ] `PortfolioState` round-trips through Redis
- [ ] Risk gate rejects per-market cap breaches with clear reason string
- [ ] `kill_switch=True` blocks all orders
- [ ] Daily loss limit halts new orders mid-day when crossed
- [ ] Sizer never returns more contracts than cash supports
- [ ] Backtest runs with different `bt:<run_id>:<signal_model>` namespaces don't interfere

---

## Stage 9: Paper-trading executor

**Goal**: wire `CalibratedEdge` from a chosen signal model through portfolio + risk + Kalshi trading client to simulated fills against the live order book. Surfaces auth/integration bugs without capital. Generates a forward dataset of (signal, outcome) pairs.

### Files to create

**`src/execution/__init__.py`** — empty.

**`src/execution/kalshi_trading_client.py`**

Extends RSA-PSS auth from `src/ingestion/kalshi/websocket_client.py` to authenticated REST endpoints (balance, positions, orders, place, cancel) and the private fill WS.

**Verify exact endpoints against current Kalshi API docs at build time.** Do not hand-write paths from this doc. Consider wrapping `kalshi-python` SDK if it's stable enough.

**`src/execution/models.py`**

```python
class OrderRequest(BaseModel):
    ticker: str
    side: str
    contracts: int
    order_type: str       # "limit" | "market"
    limit_price: float | None
    time_in_force: str    # "ioc" | "gtc"
    client_order_id: str

class OrderResult(BaseModel):
    accepted: bool
    order_id: str | None
    error: str | None
    raw_response: dict
```

**`src/execution/order_manager.py`**

Translates `CalibratedEdge` + sizer output into `OrderRequest`. Limit-price strategy: post passive at mid, reprice after N seconds, cross spread if signal still fresh. Configurable. Listens to fill WS to update `WorkingOrder` status.

**`src/execution/paper_executor.py`**

Same interface as `LiveExecutor` but never sends real orders. Simulates fills against the **live** book (real-time, not historical):
- Limit at price P on yes-side: poll book; fill when ask ≤ P
- Persists synthetic fill to `PortfolioState env=paper` with `signal_model` field set
- Tracks slippage vs. naive midpoint

**`scripts/paper_trade.py`**

```
Usage:
  python -m scripts.paper_trade --watchlist configs/watchlist.yaml
                                --signal-model calibrated_llm
                                --interval 1h
                                --max-positions 5

Loop every interval:
  for ticker in watchlist:
    snapshot = extract from live Redis
    catalog = current
    context = retrieve_and_rerank (live, no cache)
    llm_report = run_inference (live, no cache)
    edges = signal_model.signals(...)
    for ce in edges:
      decision = risk.check(ce, portfolio)
      if approved:
        contracts = sizer.size(...)
        order_manager.place_paper_order(...)
  portfolio.mark_to_market(...)
  log [paper:{signal_model}] cash=$X positions=N daily_pnl=$Y
```

Run for ≥2 weeks before considering Stage 10. Run multiple instances in parallel with different `--signal-model` flags to compare in production.

### Acceptance criteria

- [ ] Paper executor places ≥10 simulated orders without auth errors
- [ ] Fill notifications update `PortfolioState` correctly
- [ ] Paper cash + position values reconcile with sum of fills
- [ ] Settlement of a paper position increments `realized_pnl` correctly
- [ ] Daily summary log shows `[paper:{signal_model}] cash=$X positions=N daily_pnl=$Y`
- [ ] Two paper-trade processes with different signal models run concurrently without state collision

### Dependencies

```
# possibly:
kalshi-python>=...
```

---

## Stage 10: Live trading (NOT BUILT IN THIS PHASE)

Build only after:
1. Stage 7 backtest produces a signal model with `brier_score < market_baseline_brier` on a meaningful sample, and positive Sharpe
2. Stage 9 paper-trading runs cleanly for ≥2 weeks with positive or break-even P&L on the chosen signal model
3. Risk limits, kill switch, daily loss limit all verified under paper conditions

`LiveExecutor` parallels `PaperExecutor` calling real `POST /portfolio/orders`. `scripts/live_trade.py` adds:
- Manual confirmation on first N trades
- Hard cap on starting capital
- Auto-disable on daily loss limit
- Health checks (Kalshi reachable, Redis reachable, recent inference success rate)
- Locked to one signal model — no live multi-model A/B; that's what paper is for

## Conventions (same as Phase 4)

- Modern type hints: `str | None`, `list[str]`
- `@dataclass` with `field(default_factory=...)` for config
- `Pydantic BaseModel` for all data schemas
- `from src.utils.logging import get_logger`. Never `print()`
- `asyncio` for all I/O
- Wrap HTTP with `retry_with_backoff()`
- Empty `__init__.py` files

## Environment variables to add

```
# Stage 6
ARCHIVE_ROOT=data/archive

# Stage 7
BACKTEST_CACHE_DIR=data/backtest_cache
BACKTEST_PROMPT_VERSION=v1

# Stage 7.5
CALIBRATION_MAP_PATH=data/calibration_map.pkl

# Stage 8 / 9
PORTFOLIO_ENV=paper
RISK_MAX_PER_MARKET_USD=500
RISK_MAX_TOTAL_EXPOSURE_USD=5000
RISK_DAILY_LOSS_LIMIT_USD=500
KILL_SWITCH=false

# Stage 9
KALSHI_TRADING_API_KEY=...
KALSHI_TRADING_PRIVATE_KEY=...
```

## Dependencies to add to requirements.txt

```
pyarrow>=15.0.0
pandas>=2.1.0
scikit-learn>=1.3.0
# kalshi-python>=...   # consider in Stage 9
```

## Key design decisions

**LLM is retrieval + structuring, not calibration.** The defining choice. The LLM identifies which markets matter and how they relate; quantitative signal models on top produce the trading number. Decoupling lets us A/B test signal models independently and replace poor calibrators without touching retrieval. `InferenceReport` becomes a structured input, not a trading signal.

**`SignalModel` protocol is the central abstraction.** Backtest, paper trade, and live trade all consume `CalibratedEdge` from a `SignalModel`. The library of models in Stage 7.5 is extensible — any future model (gradient boosting on engineered features, etc.) plugs in by implementing the protocol.

**Cache is backtest-only, opt-in via injection.** Live and backtest share LLM call sites, but cache parameter defaults to `None`. Live behavior unchanged. Live almost never hits the cache (snapshots change every WS tick), and stale cache hits in production aren't worth the marginal savings.

**Forward archive starts immediately, used later.** Stage 6 has no consumer until Stage 7.5 wants higher-fidelity replay or signal models need historical features — but the data must accumulate from now to be available later.

**Resolved markets first for backtest.** Settlement = ground truth. Universe naturally bounded. Extend to in-flight markets with MTM after the first signal model beats baseline.

**Portfolio state in Redis, not in memory.** Survives restarts. Single source across paper, live, backtest with namespace isolation. Audit log via Redis Stream — and the `signal_model` field on Fill enables post-trade attribution.

**Paper before live.** The bug list that surfaces only in real execution is long: auth scopes, idempotency, rate limits, partial fills, fill-notification timing, weekend behavior. Paper at Stage 9 catches almost all without capital risk.

**Quarter-Kelly is conservative enough.** Already clamped to `[0.0, 0.25]` per Phase 4. Risk manager scales further down only for limit breaches, never "just to be safe."

**Correlation cap is the silent killer.** LLM finding "5 high-confidence edges" on related markets is one position with 5x variance. Risk manager treats shared `event_ticker` as one. Future: cluster by thesis similarity and cap aggregate cluster exposure.

**Stage 7.5 models are complementary, not redundant.** Each handles a different failure mode of the LLM:
- `RawLLMSignal` — baseline
- `ConsistencyArbSignal` — ignores LLM probabilities entirely; pure structural arbitrage
- `CalibratedLLMSignal` — corrects LLM probability bias empirically
- `BayesianBaseRateSignal` — replaces LLM probability with empirical base rate
- `CoherenceRegressionSignal` — uses LLM only for grouping; signal from price coherence

Likely outcome: ensemble of two or three of these is the production signal, with `RawLLMSignal` retained as a baseline for monitoring drift.

## Acceptance criteria for the phase

- [ ] `data/archive/YYYY-MM-DD/` accumulates daily (Stage 6)
- [ ] `python -m scripts.run_backtest` produces a `BacktestReport` comparing ≥2 signal models against `market_baseline_brier` (Stages 7 + 7.5)
- [ ] At least one signal model in 7.5 beats baseline Brier on the backtest sample
- [ ] Backtest is reproducible: same inputs + cache → identical outputs
- [ ] Portfolio state in Redis survives process restart (Stage 8)
- [ ] Risk manager rejects/scales correctly across all configured limits (Stage 8)
- [ ] `python -m scripts.paper_trade --signal-model X` runs and places paper orders that update portfolio state, with the `signal_model` field on every fill (Stage 9)
- [ ] All error paths produce structured log lines, not stack traces
- [ ] No live-trading code path reachable without `PORTFOLIO_ENV=live` AND `--allow-live` CLI flag

## What this phase explicitly defers

- Stage 10 live execution (gated)
- Phase 6 operations layer (scheduler, monitoring, dashboard, prompt-versioning framework)
- Sophisticated fill modeling beyond midpoint + slippage
- Multi-leg / spread orders
- Multi-exchange support
- Ensemble weighting of signal models (start with single-model selection; ensemble after individual models are validated)
