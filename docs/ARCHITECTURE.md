# Architecture

Layer-by-layer technical reference for the prediction-market arbitrage
infrastructure. Read this when you need to **modify** the system. For
end-user usage, see [USER_MANUAL.md](USER_MANUAL.md).

---

## Design principles

1. **Strategy-agnostic infrastructure.** The portfolio, risk, and
   execution layers don't know what strategy is calling them. The
   archived decay strategy lives in `experiments/decay/`; the live arb
   orchestrator (`scripts/run_arb_live.py`) lives in the main tree.
   Either could be swapped without touching the lower layers.

2. **Async-shaped I/O.** All exchange interactions, state writes, and
   alerts are `async`. The `StateBackend` Protocol mandates async even
   for in-memory backends so swapping in Redis is trivial.

3. **Pydantic everywhere for schemas.** `MarketEvent`, `TradeEvent`,
   `Position`, `Fill`, `OrderRequest`, `InferenceReport` — every typed
   data structure is Pydantic. Modern type hints (`str | None`,
   `list[str]`) throughout.

4. **Backend-agnostic state.** The `PortfolioState` class takes any
   `StateBackend` (Protocol). `InMemoryBackend` for tests + paper;
   `RedisBackend` for live. Same code works in both.

5. **No magic globals.** Configuration is loaded explicitly via
   `KalshiConfig()` / `RiskConfig()` / etc. Tests construct their own
   instances; production wires from `.env` via `python-dotenv`.

6. **Loud safety gates on side-effects.** Both live trading clients
   print a startup banner. The `--live` flag in any wrapper requires
   typed `I CONFIRM` before proceeding. Risk gates return contracts=0
   with a debug dict, never raising silently.

---

## Layer 1: Configuration

`src/config/`

```
KalshiConfig          ← .env (KALSHI_*)
  ├─ api_key_id
  ├─ private_key_pem  ← KALSHI_PRIVATE_KEY or _PATH
  ├─ ws_url, rest_base_url
  └─ market_tickers, channels, reconnect params

RedisConfig           ← .env (REDIS_*)
  └─ host, port, db, password
     .url property
```

Config classes are `@dataclass` with `field(default_factory=...)` for
env-var resolution. They raise `ValueError` if required fields are missing.

---

## Layer 2: Data ingestion

```
                        ┌──────────────────────────────┐
                        │  Kalshi WS                   │
                        │  src/ingestion/kalshi/       │
                        │   websocket_client.py (165)  │
                        │   message_parser.py   ( 80)  │
                        │   normalizer.py       (120)  │
                        └─────────────┬────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │ raw JSON                     │ ParsedMessage                 │ NormalizedEvent
       ▼                              ▼                              ▼
   websockets                    MessageType                    MarketEvent
   library                       enum                           TradeEvent
                                                                 OrderBookEvent
```

**Auth:** RSA-PSS signed headers. The signing pattern reused for the
REST trading client.

**Reconnection:** `retry_with_backoff` from `src/utils/retry.py`. Pings
every 20 seconds.

**Polymarket:**
- `src/ingestion/polymarket/clob_client.py` — read-only CLOB books
- The Gamma API (market metadata) is queried directly via stdlib
  `urllib.request` in scanners; no abstraction layer needed.

---

## Layer 3: Catalog + vectors

```
            ┌─────────────────────────────────┐
            │  scripts/build_index.py         │
            └────────┬────────────────────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
    fetcher.py   embedder.py vector_store.py
        │           │            │
   /markets        OpenAI       FAISS
   /events         OR Gemini    IndexFlatIP
   REST           via OpenAI
                  base_url
                     │
                     ▼
                data/catalog.json    (5 MB, 15.8k markets)
                data/vectors.faiss   (32 MB, 512-dim)
                data/vectors_meta.json (4 MB)
```

**Provider selection** (in `embedder._default_model()`):

```python
def _default_model() -> str:
    base = os.getenv("OPENAI_BASE_URL", "")
    if "generativelanguage.googleapis.com" in base:
        return "gemini-embedding-001"
    return "text-embedding-3-small"
```

Both providers respect the `dimensions=512` parameter, so the FAISS
index is provider-agnostic at the storage level.

**Throttling:** Gemini free tier is ~30 RPM at the embedding endpoint.
The embedder paces 4 seconds between batches at `batch_size=100`. Retry
on HTTP 429 with exponential backoff.

---

## Layer 4: LLM stack

```
              ┌──────────────────────────────────────┐
              │  Focus market (MarketSnapshot)       │
              └───────────────┬──────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────┐
              │  src/context/retriever.py            │
              │  embed → FAISS.search(k=30)         │
              └───────────────┬──────────────────────┘
                              │ list[CandidateMarket]
                              ▼
              ┌──────────────────────────────────────┐
              │  src/context/reranker.py             │
              │  Gemini chat: rate relevance 0-10    │
              │  + extract causal "relationship"     │
              └───────────────┬──────────────────────┘
                              │ list[ContextMarket]
                              ▼
              ┌──────────────────────────────────────┐
              │  src/inference/engine.py             │
              │  Gemini chat (or GPT-4o):            │
              │   1. Consistency check               │
              │   2. Conditional probabilities       │
              │   3. Cross-market divergence         │
              │   4. Stale pricing                   │
              └───────────────┬──────────────────────┘
                              │ InferenceReport
                              ▼
              ┌──────────────────────────────────────┐
              │  src/signals/*.py                    │
              │  6 signal models:                    │
              │   raw_llm, calibrated_llm,           │
              │   consistency_arb, bayesian,         │
              │   coherence_regression               │
              │  All return list[CalibratedEdge]     │
              └──────────────────────────────────────┘
```

**Inference engine model selection** (in `src/inference/engine.py`):
auto-selects `gemini-2.0-flash` if `OPENAI_BASE_URL` points at Gemini
compat, else `gpt-4o`. Per-call override via env or the runner's `--model`.

**Calibration** (`src/signals/calibrated_llm.py`):
- Loads `data/calibration_map.pkl` (sklearn IsotonicRegression)
- Maps raw LLM `estimated_fair_prob` → calibrated probability
- Trained by `scripts/train_calibration_gemini.py`
- Today's training: +1.88% Brier improvement on 178 crypto markets

---

## Layer 5: Portfolio + risk

```
   ┌──────────────────────────────────────────────────┐
   │  PortfolioState                                  │
   │  src/portfolio/state.py                          │
   │                                                  │
   │  apply_fill(Fill)        → cash debit + position │
   │  settle(ticker, value)   → realized PnL          │
   │  list_positions()                                │
   │  get_cash() / get_realized_pnl()                 │
   │  snapshot(prices)        → PortfolioSnapshot     │
   └────────────┬─────────────────────────────────────┘
                │
       ┌────────┴───────────────┐
       ▼                        ▼
 InMemoryBackend           RedisBackend
   (tests, paper)         (live; survives restart)

   ┌──────────────────────────────────────────────────┐
   │  RiskGates                                       │
   │  src/portfolio/risk_gates.py                     │
   │                                                  │
   │  compute_size(ask, depth, state, cfg, tier)     │
   │      → (contracts, debug_info)                   │
   │  drawdown_size_multiplier(realized, limit)      │
   │      → 1.0 / 0.5 / 0.25 / 0.0                   │
   │  kill_switch_state(state, cfg)                  │
   │      → (halt: bool, reason: str)                │
   └──────────────────────────────────────────────────┘
```

**The `apply_fill` flow:**

```
apply_fill(fill)
   │
   ├─ audit log: xadd to fills stream
   │
   ├─ cash: -= contracts × price + fee
   │
   ├─ if no existing position:
   │     create Position with total_fees=fill.fee
   │
   ├─ elif same side:
   │     update avg_cost = weighted average
   │     total_fees += fill.fee
   │
   └─ else (opposite side):
         _handle_opposite_side_fill(...)
            │
            ├─ closing = min(pos.contracts, fill.contracts)
            │
            ├─ realized_pnl += closing × (1−fill.price − pos.avg_cost)
            │                   − closing_side_fee − closed_entry_fee
            │
            ├─ if remaining_existing > 0: shrink position
            ├─ elif leftover_fill > 0: open opposite-side position
            └─ else: position closed flat
```

This is the "leftover-drop fix" + "fee-aware ledger" from the day-3
refactor.

**The `compute_size` flow:**

```
compute_size(ask_price, ask_size, state, cfg, tier_idx)
   │
   ├─ if kill_switch_state: return (0, "halted")
   │
   ├─ multiplier = drawdown_size_multiplier(realized, daily_limit)
   │   if multiplier == 0: return (0, "ramp_zero")
   │
   ├─ by_bankroll = state.cash × cfg.max_fraction / ask_price
   │
   ├─ by_depth = ask_size × cfg.depth_take_fraction
   │
   ├─ tier_cap_usd = cfg.max_per_market_usd × TIER_CAP_FRACTIONS[tier_idx]
   │   remaining_market = max(0, tier_cap_usd − state.existing_market_notional)
   │   by_market = remaining_market / ask_price
   │
   ├─ remaining_asset = max(0, cfg.max_per_asset_usd − state.existing_asset_notional)
   │   by_asset = remaining_asset / ask_price
   │
   ├─ raw = min(by_bankroll, by_depth, by_market, by_asset, max_per_order)
   │
   ├─ contracts = int(raw × multiplier)
   │
   └─ return (max(0, contracts), debug_info{binding: ...})
```

---

## Layer 6: Execution

```
   ┌────────────────────────────────────────────────┐
   │  KalshiTradingClient (Protocol)                │
   │  src/execution/kalshi_trading_client.py        │
   │                                                │
   │  ├─ Stub:  records orders in memory            │
   │  └─ Live:  REST POST /portfolio/orders         │
   │             with RSA-PSS auth                  │
   └────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────┐
   │  PolymarketTradingClient (Protocol)            │
   │  src/execution/polymarket_trading_client.py    │
   │                                                │
   │  ├─ Stub:  records orders in memory            │
   │  └─ Live:  py-clob-client + EIP-712 signing    │
   │             POST clob.polymarket.com/order      │
   └────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────┐
   │  Multi-leg coordinator                         │
   │  src/execution/multi_leg.py                    │
   │                                                │
   │  legs = [Leg(place=..., unwind=...), ...]      │
   │  execute_legs(legs, max_legging_window_ms)     │
   │     │                                          │
   │     ├─ asyncio.gather all legs concurrently    │
   │     ├─ if any leg fails: unwind all that       │
   │     │                    succeeded             │
   │     └─ return MultiLegResult                   │
   └────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────┐
   │  Reconciler                                    │
   │  src/execution/reconciliation.py               │
   │                                                │
   │  observe(expected, actual) → FillDivergence    │
   │     ├─ qty delta, price delta, fee delta       │
   │     ├─ severity = normal / moderate / high     │
   │     └─ score++ if not normal                   │
   │  should_halt() → bool when score ≥ threshold   │
   └────────────────────────────────────────────────┘
```

---

## Layer 7: Orchestration

```
       ┌──────────────────────────────────────────────────────────┐
       │             scripts/supervisor.py                         │
       │             (crash-restart with exp backoff)              │
       └────────────────────┬─────────────────────────────────────┘
                            │ subprocess
                            ▼
       ┌──────────────────────────────────────────────────────────┐
       │             scripts/run_arb_live.py                       │
       │                                                           │
       │   while True:                                             │
       │     1. kill_switch_state(state, cfg) → maybe sleep        │
       │     2. run_scanner() → list of arb candidates             │
       │     3. for each candidate:                                │
       │           maybe_trade_arb(...)                            │
       │             │                                             │
       │             ├─ pre-trade risk gate                        │
       │             ├─ build legs                                 │
       │             ├─ execute_legs (multi-leg coord)             │
       │             ├─ reconciler.observe                         │
       │             └─ alert on divergence                        │
       │     4. asyncio.sleep(interval_seconds)                    │
       └──────────────────────────────────────────────────────────┘
```

---

## Layer 8: Backtest

`src/backtest/` (largely unchanged from earlier phases)

```
   replayer.py    — chronological event replay
   fill_simulator.py — depth-walk fill simulation
   cache.py       — LLM response cache
   metrics.py     — Sharpe, win rate, drawdown
   runner.py      — orchestrates: replay → signal → fill → metrics
   models.py      — BacktestConfig, BacktestResult
```

Used for backtesting **any** strategy. The archived decay strategy uses
this; a future arb strategy would too.

---

## File-by-file inventory

```
src/
├── backtest/             —  ~520 lines, 30 tests
│   ├── cache.py          —  InferenceCache, deterministic by prompt+model
│   ├── fill_simulator.py —  Walks orderbook depth, computes effective price
│   ├── metrics.py        —  Sharpe / win rate / per-market breakdown
│   ├── models.py         —  BacktestConfig, BacktestResult schemas
│   ├── replayer.py       —  Event replay in chronological order
│   └── runner.py         —  Orchestrates the full backtest pipeline
│
├── catalog/              —  ~290 lines
│   ├── embedder.py       —  Provider-agnostic Gemini/OpenAI embeddings
│   ├── fetcher.py        —  Async catalog build via Kalshi REST
│   ├── models.py         —  CatalogMarket Pydantic
│   ├── store.py          —  JSON persistence (orjson if available)
│   └── vector_store.py   —  FAISS IndexFlatIP wrapper
│
├── config/
│   ├── kalshi_config.py  —  Loads .env, resolves RSA private key
│   └── redis_config.py   —  Connection params + .url property
│
├── context/              —  ~180 lines
│   ├── models.py         —  CandidateMarket, ContextMarket
│   ├── retriever.py      —  embed focus + FAISS search
│   └── reranker.py       —  Gemini relevance scoring
│
├── execution/            —  ~830 lines, 22 tests
│   ├── kalshi_trading_client.py    —  Stub + Live REST clients
│   ├── multi_leg.py                —  Concurrent leg coordinator
│   ├── models.py                   —  OrderRequest, OrderResult
│   ├── order_manager.py            —  Async order lifecycle
│   ├── paper_executor.py           —  Generic paper executor
│   ├── polymarket_trading_client.py—  Stub + Live CLOB clients
│   └── reconciliation.py           —  Severity-scored divergence
│
├── inference/            —  ~300 lines
│   ├── engine.py         —  LLM call w/ structured prompt
│   └── models.py         —  Mispricing, Edge, InferenceReport
│
├── ingestion/
│   ├── kalshi/           —  WS client, parser, normalizer
│   └── polymarket/
│       └── clob_client.py—  CLOB book + price reads, no auth
│
├── insight/              —  legacy single-market generator
├── models/               —  Event Pydantic schemas
├── portfolio/            —  ~530 lines, 26 tests
│   ├── models.py         —  Position (with total_fees), WorkingOrder, Fill
│   ├── risk.py           —  Pre-trade risk decision (legacy)
│   ├── risk_gates.py     —  Strategy-agnostic gate stack (NEW)
│   ├── sizer.py          —  Quarter-Kelly sizer
│   └── state.py          —  PortfolioState + InMemoryBackend + RedisBackend
│
├── publisher/
│   └── event_publisher.py—  Routes events to Redis Streams
│
├── signals/              —  ~680 lines, 50 tests
│   ├── bayesian_base_rate.py   —  LLM-found analogues + Bayesian update
│   ├── calibrated_llm.py       —  Isotonic regression calibration
│   ├── coherence_regression.py —  Ridge regression on context prices
│   ├── consistency_arb.py      —  Axiom violation detector
│   ├── models.py               —  CalibratedEdge, HistoricalContext
│   ├── protocol.py             —  SignalModel Protocol + quarter_kelly()
│   └── raw_llm.py              —  Pass-through baseline
│
├── storage/              —  ~280 lines, 17 tests
│   ├── catalog_archive.py
│   ├── resolution_tracker.py
│   └── snapshotter.py
│
└── utils/
    ├── alerts.py         —  4-sink alerting (file + Discord + Slack + Pushover)
    ├── logging.py        —  Structured JSON logger
    └── retry.py          —  Async exponential backoff

scripts/
├── archive_daily.py
├── benchmark_pipeline.py
├── build_index.py                   —  Cold-path index builder
├── fetch_resolutions.py
├── fit_calibration.py               —  Legacy calibration fit
├── measure_llm_signal_edge.py       —  End-to-end signal edge measurement (NEW)
├── paper_trade.py                   —  Generic LLM-signal paper trader
├── run_arb_live.py                  —  LIVE ARB ORCHESTRATOR (NEW)
├── run_backtest.py
├── run_scanners_loop.py             —  Scanner orchestrator
├── scan_alpha.py                    —  Within-Kalshi structural arb
├── scan_cross_platform_arb.py       —  v1 keyword-matched
├── scan_cross_platform_arb_llm.py   —  v2 LLM-matched (NEW)
├── supervisor.py                    —  Watchdog (NEW)
├── train_calibration_gemini.py      —  Calibration trainer (NEW)
├── validate_llm_signal.py
└── validate_signals_integration.py  —  Smoke harness (NEW)

experiments/decay/  (gitignored — strategy archive)
```

---

## Testing strategy

| Test category | Location | Count |
|---|---|---|
| Backtest (cache/replayer/sim/metrics/runner) | `tests/test_backtest_*.py` | 30 |
| Execution (order manager, paper executor, multi-leg, reconciler) | `tests/test_execution_*.py`, `test_multi_leg_*.py`, `test_reconciliation.py` | 22 |
| Live client smoke (Kalshi) | `tests/test_live_client_smoke.py` | 4 |
| Portfolio (state with fee-aware regressions, risk, sizer) | `tests/test_portfolio_*.py`, `test_risk_gates.py` | 26 |
| Risk gates (extracted) | `tests/test_risk_gates.py` | 11 |
| Signal models (one test file each) | `tests/test_signals_*.py` | 50 |
| Storage (catalog archive, resolution tracker, snapshotter) | `tests/test_storage_*.py` | 17 |
| Scanner (scan_alpha) | `tests/test_scan_alpha.py` | 10 |
| Orchestrator | `tests/test_run_arb_live.py` | 4 |
| Other | `tests/test_signals_*.py` (etc) | 5 |
| **Total** | | **179** |

Run all:
```bash
pytest tests/ -q
# 179 passed in ~2s
```

---

## Extending the system

### Adding a new signal model

1. Implement `SignalModel` Protocol in `src/signals/<your_signal>.py`:
   ```python
   class YourSignal:
       name = "your_signal"
       def signals(self, focus, context, llm_report, history) -> list[CalibratedEdge]:
           ...
   ```
2. Add unit tests in `tests/test_signals_<your_signal>.py`
3. Register in `scripts/measure_llm_signal_edge.py` for edge measurement
4. Wire into a paper trader (e.g., `scripts/paper_trade.py`)

### Adding a new exchange

1. Implement a Trading Client following `KalshiTradingClient` Protocol
2. Add a CLOB read client in `src/ingestion/<exchange>/`
3. Build a stub for tests
4. Wire into the multi-leg coordinator (it's already exchange-agnostic)
5. Update `scan_cross_platform_arb_llm.py` matching logic if cross-listed

### Adding a new alert sink

In `src/utils/alerts.py`, add an `_async_<sink>(msg)` function and add it
to the `tasks` list in `alert_async()`. The file sink always runs;
sink errors are logged but never raise.

---

*Architecture is the code, code is the docs. When in doubt, grep.*
