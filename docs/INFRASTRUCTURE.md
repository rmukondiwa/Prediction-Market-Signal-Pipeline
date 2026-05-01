# Infrastructure Reference

Complete inventory of code, data, scripts, and external dependencies.
Companion to [STRATEGIES.md](STRATEGIES.md) and [SYSTEM_ROADMAP.md](SYSTEM_ROADMAP.md).

## High-level architecture

```
┌────────────────────────────────────────────────────────────┐
│ External APIs                                               │
│  - Kalshi REST (public)        api.elections.kalshi.com    │
│  - Kalshi WebSocket (auth)     trading-api.kalshi.com      │
│  - Polymarket Gamma (public)   gamma-api.polymarket.com    │
│  - Polymarket CLOB (public)    clob.polymarket.com         │
│  - OpenAI API (paid)           api.openai.com              │
│  - Gemini API (free tier)      generativelanguage...       │
│  - Groq API (specced, not used) api.groq.com               │
└────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────┐
│ Source code (src/)                                          │
│  ingestion → models → publisher → Redis Streams            │
│  catalog → embedder → FAISS vector store                   │
│  insight ← extractor → LLM (single-market)                 │
│  context ← retriever → reranker → LLM (cross-market)       │
│  inference (LLM cross-market reasoning)                    │
│  signals ← protocol + 5 signal models                      │
│  backtest (cache, replayer, fill_simulator, runner)        │
│  storage (snapshotter, catalog_archive, resolution_tracker)│
│  portfolio (state, risk, sizer)                            │
│  execution (kalshi_trading_client, order_manager,          │
│             paper_executor)                                │
└────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────┐
│ Scripts (scripts/)                                          │
│  Data fetchers, backtests, scanners, paper trader          │
└────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────┐
│ Data (data/)                                                │
│  catalog.json, vectors.faiss, *_universe_meta.json,        │
│  *_candles.json, archive/                                   │
│                                                             │
│ Reports (reports/)                                          │
│  Backtest outputs, alpha scans                              │
│                                                             │
│ Logs (logs/)                                                │
│  paper_decay.jsonl (live forward-test events)              │
└────────────────────────────────────────────────────────────┘
```

## Source code modules (`src/`)

### `src/ingestion/kalshi/`
**Phase 1**, pre-existing. Live WebSocket client for Kalshi market data.
- `websocket_client.py` — RSA-PSS authenticated WS connection, reconnect with backoff
- `message_parser.py` — Raw JSON → `ParsedMessage` with typed `MessageType` enum
- `normalizer.py` — `ParsedMessage` → `MarketEvent` / `TradeEvent` / `OrderBookEvent`

### `src/models/`
**Phase 1**. Exchange-agnostic Pydantic event models.
- `market_event.py` — yes_bid/ask, volume, open_interest
- `trade_event.py` — yes_price/no_price, count, taker_side
- `orderbook_event.py` — yes_levels, no_levels, snapshot/delta

### `src/publisher/`
**Phase 1**. Routes normalized events to Redis Streams via XADD.
- `event_publisher.py` — Pydantic → JSON → Redis stream

### `src/catalog/`
**Phase 4**. Market universe + embedding pipeline.
- `models.py` — `CatalogMarket` (ticker, event_ticker, title, prices)
- `fetcher.py` — Paginate `/markets`, dedup `event_tickers`, fetch `/events/{X}`
- `store.py` — JSON serialization (orjson if available)
- `embedder.py` — OpenAI text-embedding-3-small at 512 dimensions
- `vector_store.py` — FAISS IndexFlatIP with L2 normalization (cosine similarity)

### `src/insight/`
**Phase 2**. Single-market LLM insight.
- `models.py` — `MarketSnapshot`, `LLMInsight`, `InsightReport`
- `extractor.py` — Async Redis read → `MarketSnapshot`
- `generator.py` — OpenAI structured output (`gpt-5-mini` via Responses API)

### `src/context/`
**Phase 4**. Cross-market retrieval + reranking.
- `models.py` — `CandidateMarket` (post-vector), `ContextMarket` (post-rerank)
- `retriever.py` — OpenAI embed query → FAISS top-k → `CandidateMarket[]`
- `reranker.py` — `gpt-4o-mini` JSON-mode rerank with relevance scores;
  added `cache: InferenceCache | None` parameter for backtest reproducibility

### `src/inference/`
**Phase 4**. Cross-market LLM reasoning.
- `models.py` — `Edge`, `Mispricing`, `DerivedProbability`, `InferenceReport`
- `engine.py` — `gpt-4o` JSON-mode inference; computes quarter-Kelly per edge;
  same `cache` parameter as reranker

### `src/signals/` ⭐ Phase 5 addition
Signal model layer — quantitative methods on top of LLM output.
- `models.py` — `CalibratedEdge`, `HistoricalContext`
- `protocol.py` — `SignalModel` Protocol; `quarter_kelly()` helper
- `raw_llm.py` — `RawLLMSignal` (pass-through baseline)
- `consistency_arb.py` — `ConsistencyArbSignal` (axiom violation detector)
- `calibrated_llm.py` — `CalibratedLLMSignal` (isotonic regression correction)
- `bayesian_base_rate.py` — `BayesianBaseRateSignal` (empirical prior + update)
- `coherence_regression.py` — `CoherenceRegressionSignal` (residual signal)
- `settlement_decay.py` — `SettlementDecaySignal` (the production strategy)

### `src/backtest/` ⭐ Phase 5 addition
Backtest framework with sound execution model.
- `models.py` — `BacktestConfig`, `SimulatedFill`, `Candle`, `BacktestReport`
- `cache.py` — `InferenceCache` (sha256-keyed disk store for LLM responses)
- `replayer.py` — `build_snapshot_at(t)` reconstruction, candle helpers
- `fill_simulator.py` — `FillSimulator` (midpoint or trade-match models)
- `metrics.py` — Brier score, Sharpe, max drawdown, hit-rate-by-confidence
- `runner.py` — `BacktestRunner` orchestrator

### `src/storage/` ⭐ Phase 5 addition
Forward-collected archive for higher-fidelity future backtests.
- `snapshotter.py` — Async Redis Streams → daily parquet
- `catalog_archive.py` — Daily catalog snapshot, `load_catalog_for_date()`
- `resolution_tracker.py` — Settlement polling for known tickers

### `src/portfolio/` ⭐ Phase 5 addition
State + risk + sizing.
- `models.py` — `Position`, `WorkingOrder`, `Fill`, `RiskLimits`, `RiskDecision`,
  `PortfolioSnapshot`
- `state.py` — `PortfolioState` (Redis or InMemory backend); ⚠️ has known bug in
  `_handle_opposite_side_fill` (leftover contracts on full close are dropped)
- `risk.py` — `RiskManager` (kill switch, daily loss limit, exposure caps)
- `sizer.py` — `QuarterKellySizer` (side-aware contract count from kelly_fraction)

### `src/execution/` ⭐ Phase 5 addition
Order placement layer.
- `models.py` — `OrderRequest`, `OrderResult`
- `kalshi_trading_client.py` — `KalshiTradingClient` Protocol + `Stub`
  implementation (records intent, never sends real orders)
- `order_manager.py` — `OrderManager` (translates `CalibratedEdge` → `OrderRequest`)
- `paper_executor.py` — `PaperExecutor` (full pipeline: edge → risk → sizer → order)

### `src/config/`
- `kalshi_config.py` — `@dataclass` from env vars (api_key_id, private_key, URLs, tickers)
- `redis_config.py` — `@dataclass` from env vars (host, port, db, stream names)

### `src/utils/`
- `logging.py` — `get_logger()` returns structured key=value logger
- `retry.py` — `retry_with_backoff()` async exponential backoff with jitter

## Scripts (`scripts/`)

### Cold-path / setup
- `build_index.py` — Fetch full Kalshi catalog → embed → FAISS index → disk
- `archive_daily.py` — Daily cron: snapshot Redis streams + catalog + resolutions

### Universe builders (Phase 5)
- `fetch_decay_candles.py` — Daily-decay universe (vol≥100, lifetime≥24h)
- `fetch_hf_candles.py` — 15M crypto universe (1-min candles)
- `fetch_expanded_universe.py` — Daily/hourly crypto + hourly weather
- `fetch_expanded_candles.py` — Smart-period candle fetcher
- `fetch_resolutions.py` — Backfill settlement data for backtests

### Backtests (Phase 5)
- `run_backtest.py` — Generic backtest runner (uses `BacktestRunner`)
- `backtest_decay.py` — Hours-based decay backtest (with sizing analysis)
- `backtest_decay_hf.py` — Minutes-based HF decay backtest (Kelly analysis)
- `backtest_decay_variants.py` — Trail-up vs multi-entry vs last-window comparison
- `backtest_combined.py` — Per-universe combined report

### Live / forward
- `paper_trade.py` — Generic paper trader (LLM-driven, pre-existing)
- `paper_trade_decay.py` — Settlement-decay forward test ⭐ active
- `benchmark_pipeline.py` — End-to-end LLM pipeline latency benchmark

### Scanners
- `scan_alpha.py` — Pure structural arb (monotonicity + partition + soft kinks)
- `scan_cross_platform_arb.py` — Kalshi vs Polymarket arb scanner

### Trainers
- `fit_calibration.py` — Isotonic regression for `CalibratedLLMSignal` (synthetic data only so far)

## Tests (`tests/`)

22 test files covering most of Phase 5:

- `test_backtest_*.py` — cache, fill_simulator, metrics, replayer, runner
- `test_execution_*.py` — order_manager, paper_executor
- `test_portfolio_*.py` — state, risk, sizer
- `test_signals_*.py` — all 5 signal models + settlement_decay
- `test_storage_*.py` — snapshotter, catalog_archive, resolution_tracker
- `test_scan_alpha.py` — parser correctness + synthetic violation detection

Run: `.venv/bin/python -m pytest tests/`

⚠️ **Missing tests** (uncovered):
- `state.py _handle_opposite_side_fill` (the known bug)
- `paper_trade_decay.py` end-to-end
- `scan_cross_platform_arb.py`

## Data (`data/`)

| File | Size | What |
|---|---|---|
| `catalog.json` | 4.8 MB | All 15.8k Kalshi markets w/ prices, titles |
| `vectors.faiss` | 32 MB | FAISS index of catalog embeddings (512-dim) |
| `vectors_meta.json` | 4.1 MB | Per-vector metadata (matches FAISS row order) |
| `decay_universe_meta.json` | small | 472 settled markets (year-old daily decay) |
| `decay_candles.json` | medium | hourly+daily candles per market |
| `hf_universe_meta.json` | small | 983 settled 15M crypto markets (90 days) |
| `hf_candles.json` | small | 1-min candles for 437 of them |
| `expanded_universe_meta.json` | small | 1212 markets (BTCD, ETHD, SOLD, weather) |
| `expanded_candles.json` | medium | smart-period candles for 942 of them |

Gitignored: `archive/`, `backtest_cache/`, `calibration_map.pkl`

## Reports (`reports/`)

JSON outputs from backtest/scanner runs:

- `alpha.json` — `scan_alpha` results
- `decay_backtest.json`, `hf_decay_backtest.json`, `combined_backtest.json` — backtest sweeps
- `cross_platform_arb.json` — research artifact (no real arbs found)
- `synth_*.json` — early synthetic-data backtest runs

## Logs (`logs/`)

- `paper_decay.jsonl` — JSONL stream of every event from the live paper trader
  (start, fill, settled, signal_skipped, tick_summary, tick_error)

## External APIs in use

### Public (no auth)
- **Kalshi REST** — `api.elections.kalshi.com/trade-api/v2`
  - `/markets?status=*&min_close_ts=*&max_close_ts=*` — paginated market list
  - `/markets/{ticker}` — full market detail (status, result, settlement_value)
  - `/events?series_ticker=X&status=open` — ⭐ events by series (faster than paginating all markets)
  - `/events/{event_ticker}` — markets at top level of response (NOT nested under "event")
  - `/series/{series}/markets/{ticker}/candlesticks` — OHLC at period_interval (1, 60, 1440)
- **Polymarket Gamma** — `gamma-api.polymarket.com`
  - `/markets?active=true&closed=false&order=volumeNum` — by volume
  - Requires User-Agent header
- **Polymarket CLOB** — `clob.polymarket.com`
  - `/markets?limit=N` — page of markets with order book metadata

### Auth required (paid)
- **OpenAI** — for embeddings + reranker + inference (env: `OPENAI_API_KEY`)
- **Gemini** — alternative LLM, used in benchmark + future ML (env: `GEMINI_API_KEY`)
- **Kalshi WebSocket** — RSA-PSS auth for live ingestion (env: `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY`)
- **Kalshi Trading API** — for live orders (NOT yet integrated; current trader uses stub)

## Runtime

- **Python**: 3.12 in `.venv/`
- **Dependencies**: `requirements.txt` (faiss-cpu, pyarrow, pandas, scikit-learn,
  pytest, pytest-asyncio, fakeredis, aiohttp, openai, redis, etc.)
- **Backend storage**: `InMemoryBackend` (paper trader) or Redis (production-ready)
- **Process model**: single Python process per script; paper trader runs as
  long-lived asyncio loop polling every 30s

### Currently running (typical session)

1. `paper_trade_decay.py` — long-lived loop, 30s tick cadence, public Kalshi reads
2. Background `Monitor` on `logs/paper_decay.jsonl` — emits notifications on fills/settlements
3. (When debugging) ad-hoc Python scripts via `.venv/bin/python -m scripts.X`

## Git state

- **Branch**: `feature/phase5-trading-engine` (off `main`)
- **Latest commit** (as of session end): `5f6fad6` — post-audit correction
- **Not pushed to remote** — local only

## Key environment variables

```
# Kalshi WebSocket (Phase 1 ingestion)
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY=...    # raw PEM with \n escapes
KALSHI_PRIVATE_KEY_PATH=  # alternative: path to .pem file
KALSHI_WS_URL=wss://api.elections.kalshi.com/trade-api/ws/v2
KALSHI_REST_URL=https://api.elections.kalshi.com/trade-api/v2
KALSHI_MARKET_TICKERS=ticker1,ticker2

# Redis (Phase 1 publisher + Phase 5 portfolio state)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# OpenAI (Phase 4 LLM)
OPENAI_API_KEY=sk-...

# Gemini (alternative LLM, used in benchmark)
GEMINI_API_KEY=AIzaSy...

# Phase 5 archive root
ARCHIVE_ROOT=data/archive

# Phase 5 backtest
BACKTEST_CACHE_DIR=data/backtest_cache
BACKTEST_PROMPT_VERSION=v1
```

## Known bugs / gotchas

1. **`src/portfolio/state.py _handle_opposite_side_fill`**: leftover contracts
   from a fill that fully closes an opposite-side position are silently dropped.
   Should open a new position with `abs(remaining)` contracts on the new side.
2. **`paper_trade_decay.py` session reporting**: tick_summary doesn't include
   `realized_pnl` from intermediate opposite-side closures. Source of truth for
   session P&L is `state.get_realized_pnl()`, not summed settlement events.
3. **Trail-up exposure aggregation**: when implied gaps past multiple
   thresholds in one tick, all fire and per-market exposure can hit 3x intended.
4. **`infer.py` typo**: imports `extract_snapshot` but actual function is
   `extract_latest_snapshot`. The Redis path errors when `--no-redis` is omitted.
   Pre-existing bug, didn't fix.
5. **Cross-platform arb scanner**: matching is keyword-based, finds many false
   positives. Needs LLM-based question matching for v2.

## Documentation map

| Doc | Purpose |
|---|---|
| [PHASE5_TRADING_ENGINE.md](PHASE5_TRADING_ENGINE.md) | Original Phase 5 spec (planned architecture) |
| [SYSTEM_ROADMAP.md](SYSTEM_ROADMAP.md) | High-level architectural overview |
| [STRATEGY_FINDINGS.md](STRATEGY_FINDINGS.md) | What we built + learned + bugs found in session |
| [STRATEGIES.md](STRATEGIES.md) | Catalog of every strategy: built / researched / future |
| [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | This doc — code/data/scripts/APIs reference |
| `CLAUDE.md` (root) | Project instructions for Claude (codebase conventions) |
