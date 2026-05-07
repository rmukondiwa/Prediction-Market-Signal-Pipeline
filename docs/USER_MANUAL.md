# Prediction Market Signal Pipeline — User Manual

**Version:** 1.0 · **Last updated:** 2026-05-07 · **Tests passing:** 179/179

---

## Table of Contents

1. [What this system does](#1-what-this-system-does)
2. [Setup](#2-setup)
3. [System map](#3-system-map)
4. [Common workflows](#4-common-workflows)
   - [4.1 Refresh the catalog + vector index](#41-refresh-the-catalog--vector-index)
   - [4.2 Run the arb scanners](#42-run-the-arb-scanners)
   - [4.3 Train the LLM calibration map](#43-train-the-llm-calibration-map)
   - [4.4 Paper-trade arbitrage](#44-paper-trade-arbitrage)
   - [4.5 Live-trade arbitrage](#45-live-trade-arbitrage)
5. [Component reference](#5-component-reference)
6. [Risk management](#6-risk-management)
7. [Operational concerns](#7-operational-concerns)
8. [Cost reference](#8-cost-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

---

## 1. What this system does

This is **infrastructure for arbitrage trading on prediction markets**. The
core thesis is that the same real-world event is sometimes priced
differently across Kalshi and Polymarket, and that semantically-related
markets within a single venue can violate axioms (monotonicity, partition).

The pipeline detects those mispricings, validates them, sizes positions
under a multi-layer risk framework, and executes simultaneously on both
venues with auto-unwind if either leg fails.

**What it is good for:**
- Cross-platform arbitrage (Kalshi ↔ Polymarket)
- Within-platform structural scanning (riskless arb, soft kinks)
- LLM-assisted semantic matching of questions across venues
- Paper trading + live trading with full risk gates and observability
- Backtest research with deterministic replay

**What it is *not* designed for:**
- Directional speculation (the archived decay strategy in
  [experiments/decay/](../experiments/decay/) is a research artifact only)
- High-frequency market making (sub-second latency)
- Settlement timing arbitrage (resolved by exchange, not exploitable)

---

## 2. Setup

### 2.1 Python environment

Python 3.12+ required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:
```bash
pytest tests/ -q
# → 179 passed
```

### 2.2 Environment variables

Copy `.env.example` to `.env` and fill in. Minimum for read-only operation:

```bash
# LLM provider — Gemini via OpenAI-compat endpoint
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=AIzaSy...               # same key, used by OpenAI SDK
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Add for live Kalshi trading:
```bash
KALSHI_API_KEY_ID=<your-key-id>
KALSHI_PRIVATE_KEY_PATH=/path/to/kalshi_key.pem
# OR
KALSHI_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\n...\n-----END...
```

Add for live Polymarket trading:
```bash
POLYMARKET_PRIVATE_KEY=0x...           # Polygon EOA with USDC + allowances
POLYMARKET_FUNDER=0x...                # optional, delegated funder
POLYMARKET_API_KEY=...                 # optional L2 creds
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
```

Add for live alerts (any/all optional):
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/.../...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/.../.../...
PUSHOVER_TOKEN=...                     # high-severity push notifications
PUSHOVER_USER=...
```

Add for persistent state (recommended for live):
```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://localhost:6379/0     # optional shorthand
```

### 2.3 One-time external setup

For **Polymarket live trading** (skip if you only want Kalshi):

1. Create a Polygon wallet (MetaMask is fine)
2. Deposit USDC on Polygon — **native USDC, not bridged USDC.e**
3. Approve allowances per [Polymarket allowance docs](https://docs.polymarket.com/developers/CLOB/allowances):
   - USDC → `CTFExchange` contract
   - `ConditionalTokens` → `CTFExchange`
   - Same for `NEG_RISK` variants if you'll trade neg-risk markets
4. Export the wallet's private key into `POLYMARKET_PRIVATE_KEY`

For **Redis** (recommended for live):
```bash
brew install redis      # macOS
redis-server &
```

For **alert webhooks** (recommended for live):
- Discord: server settings → integrations → webhooks → new webhook
- Slack: api.slack.com → create app → enable incoming webhooks
- Pushover: pushover.net → register, create app, copy token + user key

---

## 3. System map

### 3.1 High-level architecture

```
                              ┌────────────────────────────────────────┐
                              │            CONFIGURATION               │
                              │  KalshiConfig · RedisConfig · .env     │
                              └────────────────┬───────────────────────┘
                                               │
   ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
   │                                           │                                           │
   ▼                                           ▼                                           ▼
┌─────────────────┐               ┌─────────────────────┐                     ┌────────────────────┐
│  INGESTION      │               │   STORAGE / CATALOG │                     │  RESEARCH          │
│                 │               │                     │                     │                    │
│ Kalshi WS       │ ──events────▶ │ Redis Streams       │ ──snapshot────────▶ │ resolution_tracker │
│ (RSA-PSS auth)  │               │ event_publisher     │                     │ catalog_archive    │
│                 │               │                     │                     │ snapshotter        │
│ Polymarket      │ ──books────▶  │ catalog.json        │                     │                    │
│ Gamma + CLOB    │               │ vectors.faiss       │                     │ backtest replayer  │
└─────────────────┘               │ (Gemini-embedded,   │                     │ + fill simulator   │
                                  │  15,801 mkts × 512) │                     └────────────────────┘
                                  └──────────┬──────────┘
                                             │
   ┌─────────────────────────────────────────┼──────────────────────────────────────────────┐
   │                                         │                                              │
   ▼                                         ▼                                              ▼
┌────────────────────┐         ┌──────────────────────┐                       ┌───────────────────┐
│  SCANNERS          │         │  LLM STACK           │                       │  EXECUTION CLIENTS│
│                    │         │                      │                       │                   │
│ scan_alpha         │         │ retriever (FAISS)    │                       │ KalshiClient      │
│ scan_xplatform v1  │         │ reranker (Gemini)    │                       │  ├── stub         │
│ scan_xplatform v2  │ ◀────── │ inference engine     │                       │  └── live (REST)  │
│ (LLM-matched)      │         │                      │                       │                   │
│ run_scanners_loop  │         │ 6 signal models:     │                       │ PolymarketClient  │
│ (orchestrator)     │         │  raw · calibrated    │                       │  ├── stub         │
└────────────────────┘         │  consistency_arb     │                       │  └── live (CLOB)  │
                               │  bayesian · coherence│                       └───────────────────┘
                               └──────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────────┐
                              │       PORTFOLIO + RISK              │
                              │                                     │
                              │  PortfolioState (fee-aware)         │
                              │   ├── InMemoryBackend (paper/test)  │
                              │   └── RedisBackend (live)           │
                              │                                     │
                              │  RiskGates (8 layers):              │
                              │   per-fill · per-market · per-asset │
                              │   per-tier · drawdown ramp          │
                              │   daily kill · drawdown trail       │
                              │                                     │
                              │  Reconciler (severity-scored)       │
                              └────────────────────┬────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────┐
                              │       ORCHESTRATION                 │
                              │                                     │
                              │  run_arb_live  ─┐                   │
                              │                 ├─ multi-leg coord  │
                              │                 ├─ alerts (4 sinks) │
                              │                 └─ supervisor       │
                              └─────────────────────────────────────┘
```

### 3.2 Layer inventory

| Layer | Path | Lines | Tests |
|---|---|---|---|
| Configuration | `src/config/` | ~110 | — |
| Kalshi ingestion | `src/ingestion/kalshi/` | ~365 | covered via integration |
| Polymarket ingestion | `src/ingestion/polymarket/` | ~115 | smoke ✅ |
| Event models | `src/models/` | ~150 | — |
| Event publishing | `src/publisher/` | ~80 | — |
| Storage | `src/storage/` | ~280 | 17 ✅ |
| Catalog + FAISS | `src/catalog/` | ~290 | — |
| Context retrieval | `src/context/` | ~180 | — |
| Inference engine | `src/inference/` | ~300 | — |
| Signal models | `src/signals/` | ~680 | 50 ✅ |
| Portfolio | `src/portfolio/` | ~530 | 26 ✅ |
| Execution | `src/execution/` | ~830 | 22 ✅ |
| Insight (legacy) | `src/insight/` | ~210 | — |
| Backtest | `src/backtest/` | ~520 | 30 ✅ |
| Utilities | `src/utils/` | ~280 | smoke ✅ |
| Scripts (entry points) | `scripts/` | ~3,500 | smoke + 4 ✅ |
| **Total active** | | **~8,500** | **179** |
| Archived (decay) | `experiments/decay/` | ~1,950 | — |

### 3.3 The data flow for one arbitrage trade

```
       ┌──────────────────────────────────────────────────────┐
       │                                                      │
       │   1. SCANNER — every 5 min, run_scanners_loop        │
       │                                                      │
       │      ┌─────────────────────────────────────┐         │
       │      │ scan_cross_platform_arb_llm:        │         │
       │      │  • Pull Polymarket Gamma API        │         │
       │      │  • Filter joke/farm markets         │         │
       │      │  • Embed both sides via Gemini      │         │
       │      │  • Cosine top-K matches             │         │
       │      │  • LLM verifies identity (strict)   │         │
       │      │  • Threshold post-filter            │         │
       │      │  • Compute arb math w/ CLOB prices  │         │
       │      └────────────────┬────────────────────┘         │
       │                       ▼                              │
       │      reports/cross_platform_arb_llm.json              │
       │                                                      │
       └──────────────────────┬───────────────────────────────┘
                              │ verified arbs
                              ▼
       ┌──────────────────────────────────────────────────────┐
       │                                                      │
       │   2. ORCHESTRATOR — run_arb_live                     │
       │                                                      │
       │     for each verified arb above min_edge:            │
       │                                                      │
       │       a) Pre-flight RISK GATES                       │
       │          • Daily loss kill switch                    │
       │          • Drawdown trail                            │
       │          • Per-fill / market / asset / tier caps    │
       │          ► size = min(...) × ramp_multiplier        │
       │          ► if size==0, skip this arb                 │
       │                                                      │
       │       b) Build LEGS                                  │
       │          • Leg A: buy YES on Kalshi @ ask            │
       │          • Leg B: buy NO  on Polymarket @ ask        │
       │            (or symmetrically the other direction)    │
       │          • Each leg has place + cancel + unwind cb   │
       │                                                      │
       │       c) MULTI-LEG COORDINATOR                       │
       │                                                      │
       │          asyncio.gather([Leg A, Leg B])              │
       │              │                                       │
       │              ├─ both succeed → done, $1 payout       │
       │              │                                       │
       │              └─ one fails → unwind succeeded leg     │
       │                              + alert                 │
       │                                                      │
       │       d) RECONCILER — score divergence               │
       │          • qty delta, price slippage, fees, latency  │
       │          • severity = normal / moderate / high       │
       │          • cumulative score → halt threshold         │
       │                                                      │
       │       e) STATE UPDATE                                │
       │          • PortfolioState.apply_fill(...)            │
       │          • Cash, position, fees, realized PnL        │
       │          • Persisted to Redis if live                │
       │                                                      │
       │       f) ALERT on any high-severity event            │
       │          • file sink (always)                        │
       │          • Discord / Slack / Pushover                │
       │                                                      │
       └──────────────────────────────────────────────────────┘
```

---

## 4. Common workflows

### 4.1 Refresh the catalog + vector index

The catalog (15,801 markets) and FAISS index (Gemini embeddings, 512-dim)
are cached on disk. Refresh them when you want to scan against current
market data.

**Initial build:**
```bash
python -m scripts.build_index
```

**Skip the network fetch** (faster, re-uses cached `data/catalog.json`):
```bash
python -m scripts.build_index --skip-fetch --no-filter
```

Build time: ~7 min for 15.8k markets at Gemini's free-tier rate (15 RPM).
Cost: ~$0.02 per full rebuild.

**Output:**
- `data/catalog.json` — market metadata (~5 MB)
- `data/vectors.faiss` — FAISS IndexFlatIP, 32 MB
- `data/vectors_meta.json` — per-vector metadata (~4 MB)

### 4.2 Run the arb scanners

#### One-shot scans

**Within-Kalshi structural arbs** (monotonicity + partition + soft kinks):
```bash
python -m scripts.scan_alpha --output reports/alpha.json
```

**Cross-platform v1** (regex/keyword matching — finds ~0):
```bash
python -m scripts.scan_cross_platform_arb
```

**Cross-platform v2** (LLM-matched with CLOB prices — recommended):
```bash
python -m scripts.scan_cross_platform_arb_llm \
  --poly-limit 100 --kalshi-prefixes KXBTCD,KXETHD,KXBTC,KXBTCY \
  --top-k 3 --max-verify 40 --sim-threshold 0.70
```

#### Continuous orchestrator

Run all scanners on a schedule, alert on hits:
```bash
python -m scripts.run_scanners_loop &
```

Schedule:
- `scan_alpha` every 30 min
- `scan_cross_platform_arb` every 60 min

Output: `logs/scanner_hits.jsonl`. An `ALERT: RISKLESS_ARB_FOUND` line
appears for any genuine riskless arb (none observed yet — MMs eat them).

### 4.3 Train the LLM calibration map

The `CalibratedLLMSignal` learns a map from raw LLM probabilities to
calibrated probabilities by fitting isotonic regression on resolved
markets. Today's training on 178 crypto markets shows +1.88% Brier
improvement over baseline.

**Run training:**
```bash
python -m scripts.train_calibration_gemini --n 200 --crypto-only \
  --concurrency 2 --output data/calibration_map.pkl
```

Cost: ~$0.02. Time: ~5 min at Gemini free-tier RPM.

**Output:**
- `data/calibration_map.pkl` — fitted IsotonicRegression
- `data/calibration_pairs.jsonl` — raw training data (resumable)

### 4.4 Paper-trade arbitrage

```bash
python -m scripts.run_arb_live
```

Runs with stub clients on both venues. Submits no real orders. Surfaces
candidate arbs with sizing + dry-run results.

Output goes to `logs/arb_live.jsonl`.

You can adjust thresholds:
```bash
python -m scripts.run_arb_live \
  --bankroll 1000 --max-trade-usd 100 \
  --min-edge 0.03 --interval-seconds 180
```

### 4.5 Live-trade arbitrage

⚠️ **Real money on Kalshi + Polymarket. Read everything first.**

#### Pre-flight checklist

- [ ] Kalshi account funded with intended amount
- [ ] Kalshi API key + private key in `.env`
- [ ] Polygon wallet funded with USDC
- [ ] CTF + USDC allowances approved on Polygon
- [ ] `POLYMARKET_PRIVATE_KEY` in `.env`
- [ ] Webhook alerts configured (Discord/Slack/Pushover)
- [ ] Redis running (`redis-server &`)
- [ ] Have read this manual through to § 6 (Risk management)
- [ ] Have run the paper version for at least one full session
- [ ] Have a kill plan (how do I stop everything if alerts blow up?)

#### Recommended invocation

Always run under the supervisor for crash protection:

```bash
python -m scripts.supervisor --max-restarts 5 --backoff-base 5 -- \
  python -m scripts.run_arb_live --live \
    --bankroll 500 \
    --daily-loss-limit 30 \
    --drawdown-limit 20 \
    --min-edge 0.02 \
    --max-trade-usd 50 \
    --interval-seconds 300 \
    --use-redis
```

The orchestrator will print a confirmation banner. **Type `I CONFIRM`
exactly** to start. Anything else aborts.

#### What happens during a run

```
[T+0]    confirm_live() prompt — operator types "I CONFIRM"
[T+1]    Connect to Kalshi REST + Polymarket CLOB
[T+2]    Initialize Redis-backed PortfolioState (or load existing)
[T+3]    Alert: "Live arb orchestrator starting" (warning severity)
[T+5]    First scanner run (subprocess: scan_cross_platform_arb_llm)
[T+30]   Scanner result: N verified arbs, M above min_edge
[T+30]   For each candidate:
           pre-trade: kill switch check, size via risk gates
           place: 2 legs concurrent, dispersion <2s
           reconcile: severity score, halt at threshold=5
           alert: any failure or divergence
           state: persist to Redis
[T+330]  Sleep until next interval
[T+330+] Loop forever (until kill switch trips, supervisor max restarts,
         or operator SIGINT)
```

#### How to stop

- **Graceful:** `kill -TERM <pid>` on the supervisor. It forwards SIGTERM
  to the child, child finishes current scan/trade, then exits.
- **Hard:** `kill -KILL <pid>` on both supervisor and child. Open
  positions remain on the exchange — close them manually if needed.

---

## 5. Component reference

### 5.1 Kalshi trading client

[src/execution/kalshi_trading_client.py](../src/execution/kalshi_trading_client.py)

Two implementations behind a Protocol:

```python
class KalshiTradingClientStub:
    """Records orders in memory. Used by paper trading + tests."""

class KalshiTradingClientLive:
    """REST POST /portfolio/orders with RSA-PSS auth.
    Dollar→cent conversion clamped to [1,99].
    Side-aware: yes_price for buy-yes, no_price for buy-no.
    IOC support via expiration_ts.
    Errors return OrderResult(accepted=False, error=...).
    """
```

**Auth:** same RSA-PSS pattern as the WebSocket client. Each request
signs `<timestamp_ms><METHOD><path>` and sends three headers:
`KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`.

### 5.2 Polymarket trading client

[src/execution/polymarket_trading_client.py](../src/execution/polymarket_trading_client.py)

```python
class PolymarketTradingClientStub:
    """Symmetric to Kalshi stub."""

class PolymarketTradingClientLive:
    """Wraps py-clob-client.
    EIP-712 signed orders, posted to clob.polymarket.com/order.
    Read-only L2 creds derived from L1 key on first use.
    Loud startup banner."""
```

The `OrderRequest` translation between Kalshi and Polymarket flavor:

```python
from src.execution.polymarket_trading_client import from_kalshi_order
poly_req = from_kalshi_order(kalshi_order_request, token_id="...")
```

### 5.3 Polymarket CLOB read

[src/ingestion/polymarket/clob_client.py](../src/ingestion/polymarket/clob_client.py)

```python
from src.ingestion.polymarket.clob_client import get_book, get_market_books

# For one outcome token
book = get_book(token_id="139156893172690...")
print(book.top_bid.price, book.top_bid.size)
print(book.top_ask.price, book.top_ask.size)
print(book.spread)

# For a Polymarket market dict (Gamma API)
yes_book, no_book = get_market_books(market)
```

No auth required for read endpoints.

### 5.4 Risk gates

[src/portfolio/risk_gates.py](../src/portfolio/risk_gates.py)

Strategy-agnostic. Pass current state + config, get back allowed contract
count + binding-constraint info.

```python
from src.portfolio.risk_gates import RiskConfig, RiskState, compute_size

cfg = RiskConfig(
    max_fraction_per_fill=0.05,    # 5% of bankroll per fill
    depth_take_fraction=0.25,       # take ≤25% of inside ask
    max_per_market_usd=500.0,       # any one ticker
    max_per_asset_usd=1000.0,       # any one underlying (BTC/ETH/...)
    daily_loss_limit_usd=300.0,     # hard halt
    drawdown_limit_usd=200.0,       # peak-to-trough trail
)
state = RiskState(
    cash=4500, realized_pnl=-50, peak_realized_pnl=100,
    existing_market_notional=200, existing_asset_notional=400,
)

contracts, info = compute_size(
    ask_price=0.85, ask_size=1000,
    state=state, cfg=cfg, threshold_tier=2,
)
print(contracts, info["binding"])  # "depth" / "bankroll" / "per_market_tier" / "per_asset"
```

The 8 risk layers are detailed in [§ 6 Risk management](#6-risk-management).

### 5.5 Multi-leg coordinator

[src/execution/multi_leg.py](../src/execution/multi_leg.py)

Submit N legs concurrently, auto-unwind succeeded legs if any fail.

```python
from src.execution.multi_leg import Leg, execute_legs

legs = [
    Leg(name="kalshi-yes", venue="kalshi",
        place=lambda: kalshi.place_order(req_a),
        cancel=kalshi.cancel_order,
        unwind=lambda: kalshi.place_order(opposite_market_order),
        notional=50.0),
    Leg(name="poly-no", venue="polymarket",
        place=lambda: poly.place_order(req_b),
        cancel=poly.cancel_order,
        unwind=lambda: poly.place_order(opposite_token_order),
        notional=50.0),
]
result = await execute_legs(legs, max_legging_window_ms=2_000.0)

if not result.all_succeeded:
    print(f"Failed: {[l.name for l in result.failed_legs]}")
    print(f"Unwound: {[l.name for l in result.unwound_legs]}")
print(f"Dispersion: {result.place_dispersion_ms:.1f}ms")
```

### 5.6 Reconciliation

[src/execution/reconciliation.py](../src/execution/reconciliation.py)

```python
from src.execution.reconciliation import Reconciler, ExpectedFill, ActualFill

rec = Reconciler()  # default config

expected = ExpectedFill(ticker="X", side="yes", contracts=100,
                          price=0.85, fee_estimate=0.5)
# ... place order, parse response into ActualFill ...
actual = ActualFill(ticker="X", side="yes", contracts=92,
                     price=0.86, fee=0.55)
divergence = rec.observe(expected, actual)

if divergence.severity == "high":
    # alert + halt
    pass
if rec.should_halt():
    # cumulative divergence score crossed threshold
    pass
```

### 5.7 Live alerting

[src/utils/alerts.py](../src/utils/alerts.py)

```python
from src.utils.alerts import alert, alert_async

# Sync (file sink + best-effort webhook if loop running)
alert("Kill switch tripped", severity="high",
      context={"realized_pnl": -302, "limit": 300})

# Async (preferred from async code)
await alert_async("Position opened",
                  severity="info",
                  context={"ticker": "X", "contracts": 50})
```

Severity levels:
- `info` → file sink only (`logs/alerts.jsonl`)
- `warning` → file + Discord + Slack
- `high` → file + Discord + Slack + Pushover (if configured)

### 5.8 Watchdog supervisor

[scripts/supervisor.py](../scripts/supervisor.py)

```bash
python -m scripts.supervisor --max-restarts 10 --backoff-base 5 -- \
  python -m scripts.run_arb_live --live ...
```

- Restarts the child on non-zero exit
- Exponential backoff (5s, 8.5s, 14.4s, ...) capped at 5 min
- Resets backoff after `--healthy-run-seconds` (default 120s)
- Alerts on every restart (severity=warning)
- Alerts on max-restarts-reached (severity=high)

### 5.9 Portfolio state

[src/portfolio/state.py](../src/portfolio/state.py)

```python
from src.portfolio.state import PortfolioState, InMemoryBackend, RedisBackend

# Paper / tests
state = PortfolioState(InMemoryBackend(), env="paper_arb")
await state.initialize(starting_capital=500.0)

# Live
state = PortfolioState(
    RedisBackend(url="redis://localhost:6379/0"),
    env="live_arb",
)

# Read
cash = await state.get_cash()
realized = await state.get_realized_pnl()
positions = await state.list_positions()

# Write
await state.apply_fill(fill)              # respects fee-aware ledger
pnl = await state.settle(ticker, value=1.0, t=now)
```

The Redis backend cap streams at 10k entries each (auto-trim).

---

## 6. Risk management

The 8-layer risk stack, applied in this order to every fill decision:

```
   ┌────────────────────────────────────────────────────────────┐
   │  Daily kill switch       — realized PnL ≤ -daily_limit?    │
   │  Drawdown trail          — peak - current ≥ trail_limit?   │
   │      ▼ if either YES, halt entirely                        │
   ├────────────────────────────────────────────────────────────┤
   │  Drawdown ramp           — multiplier ∈ {1, 0.5, 0.25, 0}  │
   │      based on % of daily limit consumed                    │
   ├────────────────────────────────────────────────────────────┤
   │  Per-fill bankroll cap   — ≤ 5% of cash                    │
   │  Depth cap               — ≤ 25% of inside ask              │
   │  Per-tier cumulative     — 30% / 60% / 100% × per_market   │
   │  Per-market cumulative   — sum of fills on one ticker      │
   │  Per-asset cumulative    — sum across same-underlying       │
   │      ▼ contracts = min(all of these) × ramp_multiplier     │
   └────────────────────────────────────────────────────────────┘
```

**Drawdown ramp brackets:**

| Realized PnL as % of daily limit | Multiplier |
|---|---|
| ≥ 0% (i.e., flat or up) | 1.00 |
| 0 to −33% of limit | 1.00 |
| −33% to −66% | 0.50 |
| −66% to −99% | 0.25 |
| ≥ −100% (kill) | 0.00 |

Example: with `daily_loss_limit_usd=300` and current `realized_pnl=-150`,
ramp multiplier = 0.50. A fill that would normally be 100 contracts
becomes 50.

**Per-tier sizing** (for trail-up strategies):

| Trail-up tier | Cumulative cap (% of `max_per_market_usd`) |
|---|---|
| Tier 0 (early threshold, e.g. 0.75) | 30% |
| Tier 1 (mid, 0.85) | 60% (so +30% over tier 0) |
| Tier 2 (terminal, 0.95) | 100% (so +40% over tier 1) |

This is back-loaded: the smallest entry happens at the most reversal-prone
threshold. Empirically saved ~$300 in the archived decay forward test.

**Reconciler halt:**

Each fill divergence (qty, price, fee, latency) gets a severity
(`normal`/`moderate`/`high`). Cumulative score increments by 1 (moderate)
or 2 (high). Reaching `cfg.high_score_threshold` (default 5) signals the
caller should halt — live live diverges from paper enough that you can't
trust it.

---

## 7. Operational concerns

### 7.1 Deployment topology

```
   ┌─────────────────────────────────────────────────────────────┐
   │                       OPERATOR LAPTOP                        │
   │                                                              │
   │   ┌─────────────────┐   ┌─────────────────────────────┐     │
   │   │ Redis (local)   │   │ supervisor.py               │     │
   │   │ port 6379       │◀──┤   ↓ subprocess              │     │
   │   └─────────────────┘   │ run_arb_live (live mode)    │     │
   │                         │   ├─ Kalshi REST            │     │
   │                         │   ├─ Polymarket CLOB        │     │
   │                         │   └─ Gemini for verification│     │
   │                         └─────────┬───────────────────┘     │
   │                                   │ alerts                  │
   │                                   ▼                         │
   └────────────────────────────┬──────────────────────────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  │                            │
                  ▼                            ▼
         ┌────────────────┐         ┌──────────────────────┐
         │ Discord        │         │ Pushover (mobile)    │
         │ Slack          │         │ for high severity    │
         └────────────────┘         └──────────────────────┘
```

For 24/7 operation, run on a small VPS instead of laptop. Add a systemd
unit that invokes the supervisor; redirect supervisor stdout/stderr to
a logrotate-managed file.

### 7.2 Logs

| Path | Content |
|---|---|
| `logs/arb_live.jsonl` | Per-trade results from `run_arb_live` |
| `logs/scanner_hits.jsonl` | Scanner orchestrator output (every run) |
| `logs/alerts.jsonl` | All alerts, all severities (file sink always writes) |
| `reports/cross_platform_arb_llm.json` | Latest scanner output (overwritten) |
| `reports/llm_signal_edge.json` | Latest LLM signal edge measurement |

All paths gitignored. Rotate manually or add `logrotate` for long runs.

### 7.3 State backups (Redis)

Live state lives in Redis. Snapshot daily:

```bash
redis-cli BGSAVE
# Then copy /usr/local/var/db/redis/dump.rdb (path depends on install)
```

If you lose Redis state, the orchestrator will boot up with `cash=0` —
won't trade until you re-initialize. The audit log in `logs/arb_live.jsonl`
can reconstruct positions if needed.

### 7.4 What the alerts mean

| Alert title | Severity | Action |
|---|---|---|
| Live arb orchestrator starting | warning | Confirm intended |
| Kill switch active — skipping arb | warning | Investigate (are losses real?) |
| Arb dry-run — would have placed | info | Informational only (paper) |
| Arb leg failed — exposure may exist | **high** | **Check positions immediately** |
| Reconciler score crossed halt threshold | **high** | Pause, audit recent fills |
| Child process crashed | warning/high | Supervisor handles; check logs |
| Supervisor giving up — max restarts | high | Manual intervention required |

---

## 8. Cost reference

### 8.1 LLM costs (Gemini)

| Operation | Cost | Notes |
|---|---|---|
| Embed 15,801 markets (catalog rebuild) | ~$0.02 | Free tier covers it |
| LLM signal edge measurement (n=30) | ~$0.10 | Per run |
| Calibration training (n=200) | ~$0.02 | Per run |
| Cross-platform arb scanner (per cycle) | ~$0.005 | n=30 verifications |
| **Daily LLM cost at 1 arb scan / 5 min** | **~$1.50** | 288 scans/day |

### 8.2 Trading costs

**Kalshi:** maker/taker fees per [Kalshi fee schedule](https://kalshi.com/docs/exchange-fees).
Roughly: 7% × p × (1−p) per contract. At p=0.85, ~$0.0089/contract.

**Polymarket:** zero protocol fees on most markets. You pay USDC gas on
Polygon for transactions (~$0.001-0.01 per order). Plus the wide-market
"taker spread" if crossing the book.

For an arbitrage trade with ~2¢ edge per dollar:
- Kalshi fee: ~0.5-1¢ depending on prices
- Polymarket gas: <0.1¢
- Net edge captured: ~1¢ per $1 of round-trip

### 8.3 Infrastructure

| Resource | Monthly |
|---|---|
| VPS (1 vCPU, 1 GB RAM, sufficient for the orchestrator) | $5-10 |
| Redis (local, on the same VPS) | $0 |
| Discord/Slack webhooks | $0 |
| Pushover | one-time $5 |
| **Total ops** | **~$10/mo** |

---

## 9. Troubleshooting

### 9.1 "openai.OpenAIError: API key not set"

Your `.env` is not loaded or `OPENAI_API_KEY` is missing.

```bash
cat .env | grep OPENAI_API_KEY
# If empty:
echo "OPENAI_API_KEY=$GEMINI_API_KEY" >> .env
echo "OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/" >> .env
```

### 9.2 "FAISS index dim mismatch"

The cached `data/vectors.faiss` was built with a different embedding model
than you're querying with. Rebuild:

```bash
mv data/vectors.faiss data/vectors_old.faiss.bak
mv data/vectors_meta.json data/vectors_meta_old.json.bak
python -m scripts.build_index --skip-fetch --no-filter
```

### 9.3 "HTTP 429" from Gemini

Rate limit hit (free tier is ~15 RPM at chat, ~30 RPM at embed). The
embedder retries with exponential backoff up to 5 attempts. If it's still
failing, lower `--concurrency` to 1 or upgrade your Gemini tier.

### 9.4 Polymarket order rejected with "insufficient allowance"

You haven't approved USDC or CTF allowances on Polygon. See § 2.3 setup
or [Polymarket allowance docs](https://docs.polymarket.com/developers/CLOB/allowances).

### 9.5 Kalshi orders accepted but never fill

Probably an IOC order outside the book. Try widening the price by 1-2¢
or switching to GTC for slower-moving markets. The live client uses IOC
by default (matches the decay strategy convention).

### 9.6 "Child crashed" alerts every minute

Supervisor backs off on quick crashes. If crashing immediately every time,
it's a config error — supervisor halts after 5 consecutive quick crashes
and alerts (severity=high). Check `logs/arb_live.jsonl` for the last
successful tick and work backwards.

### 9.7 Reconciler keeps tripping halt

Live divergence is too high. Common causes:
- Limit price too tight (price slippage above warn threshold)
- Order partially filled (qty divergence)
- Hidden fees (e.g., NEG_RISK markets on Polymarket)

Investigate, adjust thresholds in `ReconcilerConfig`, or improve order
type choice (FOK on illiquid markets, GTC on liquid).

---

## 10. Glossary

- **Arb / arbitrage:** trades that produce risk-free profit from a
  mispricing. True arb is rare; most "arbs" we surface need careful rules
  reading to confirm both legs resolve identically.
- **CLOB:** Central Limit Order Book. Polymarket uses an off-chain CLOB
  with on-chain settlement.
- **Conditional Tokens Framework (CTF):** Polymarket's on-chain
  binary-outcome token system. Each market has a YES token and a NO
  token; one settles to $1, the other to $0.
- **Drawdown trail:** kill switch that halts when peak realized PnL
  minus current realized PnL exceeds a threshold. Locks in gains.
- **EIP-712:** Ethereum standard for typed structured data signing.
  Polymarket uses it for order authorization.
- **Fee-aware ledger:** portfolio state that tracks entry fees on each
  position and subtracts them from realized PnL at settlement. Without
  this, realized PnL overstates by the fee total.
- **InferenceReport:** the structured output of the inference engine —
  consistency analysis, derived probabilities, mispricings, suggested
  edges. Consumed by signal models.
- **Multi-leg coordinator:** submits N legs of a trade concurrently and
  unwinds successful legs if any leg fails. Bounds legging risk.
- **Quarter-Kelly:** sizing convention where bet size is 25% of the full
  Kelly fraction. Robust to estimation error in win-probability.
- **RSA-PSS:** signature scheme used by Kalshi for API auth.
- **Trail-up:** entry strategy where the system commits more size as
  implied probability climbs through successive thresholds. Used (and
  archived) in the decay strategy.
- **Tier-0 reversal:** failure mode where a trail-up entry at the earliest
  threshold (lowest confidence) gets fully filled, then the underlying
  reverses before the higher-threshold entries can validate. The
  back-loaded tier sizing in `risk_gates` is the fix.

---

*End of manual. Questions or unclear sections: open an issue or grep this
doc by keyword. The codebase has 179 unit tests as living examples — start
with `tests/test_run_arb_live.py` for the orchestrator's behavior under
the integration smoke.*
