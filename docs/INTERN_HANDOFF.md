# Intern Handoff — Prediction Market Signal Pipeline

Welcome. This document explains what the system is, the core concepts behind it, and how to get it running. It is written for someone who knows Python but may not have a background in prediction markets or quantitative trading.

After reading this, read [ARCHITECTURE.md](ARCHITECTURE.md) for technical depth and [USER_MANUAL.md](USER_MANUAL.md) for operating the system day-to-day.

---

## What is a prediction market?

A prediction market is a betting exchange where contracts pay out $1 if a specific event happens and $0 if it doesn't. The market price of a contract therefore reflects the crowd's implied probability of that event occurring.

Example: "Will the Federal Reserve cut rates by July?" trades at 0.62, meaning the market believes there is a 62% chance of a rate cut.

Two platforms matter here:
- **Kalshi** — US-regulated, operates via REST + WebSocket API
- **Polymarket** — crypto-native, runs on Polygon blockchain, uses a Central Limit Order Book (CLOB)

---

## What this system does

The system looks for **mispricings** — cases where the price on a market is wrong, or where two related markets contradict each other in a way that creates a risk-free profit.

There are two types of opportunity the system hunts:

### 1. Cross-platform arbitrage
The same real-world event is sometimes listed on both Kalshi and Polymarket. If the implied probabilities diverge enough to cover fees, you can buy YES on one platform and NO on the other. One of them pays out $1, so you lock in profit regardless of the outcome.

Example: "BTC above $100k on Dec 31" is at 0.62 on Kalshi and 0.68 on Polymarket. Buying NO on Polymarket (at 0.32) + YES on Kalshi (at 0.62) costs 0.94. The payout is $1 no matter what BTC does. Edge: 6¢ per $1.

### 2. Within-platform structural arbitrage
A single platform's markets can violate basic logic:
- **Monotonicity:** "BTC above $80k by June" can't be cheaper than "BTC above $80k by December" — June is a stricter condition.
- **Partition:** "BTC in range $80k–$90k" + "BTC in range $90k–$100k" + ... should sum to 1.0. If they sum to 0.87, something is mispriced.

These are called **riskless arbs** — they don't depend on any prediction being right.

---

## How the system finds these opportunities

### Step 1 — Build a catalog and vector index (cold path, runs daily)

The system fetches all ~15,000 active markets from Kalshi's REST API and stores them in `data/catalog.json`. It then generates a **vector embedding** for each market title using an LLM embedding model (Gemini or OpenAI). These 512-dimensional vectors are stored in a FAISS index at `data/vectors.faiss`.

**What is an embedding?** A numeric representation of text that places semantically similar sentences near each other in vector space. "Will the Fed cut rates in June?" and "Federal Reserve rate decision by summer" end up close together even though they share no keywords.

**What is FAISS?** Facebook's open-source library for fast approximate nearest-neighbour search over millions of vectors. At 15k markets, it returns the 30 most-similar markets to any query in microseconds.

Run the cold path:
```bash
python3 -m scripts.build_index
```

### Step 2 — Scan for cross-platform matches (hot path, runs every 5 min)

The scanner (`scripts/scan_cross_platform_arb_llm.py`) pulls the top Polymarket markets, embeds their titles, and uses FAISS to find Kalshi markets with similar embeddings. An LLM (Gemini) then verifies each candidate pair to confirm they cover the exact same event — preventing false positives from semantically-close-but-legally-different questions.

### Step 3 — Retrieve context and run inference

For a given market, the system finds the 30 most-related Kalshi markets (via FAISS), then asks an LLM reranker (Gemini) to score each one for causal relevance. The top-scoring context markets are passed to the inference engine, which reasons across the group to find:
- Axiom violations (monotonicity, partition)
- Implied conditional probabilities (P(B|A) from co-priced markets)
- Stale pricing (one market's price hasn't moved while its peers have)

Output is a structured `InferenceReport` with specific price estimates and suggested trades.

### Step 4 — Size positions with risk gates

Before placing any order, the system calculates how many contracts to buy using an 8-layer risk stack:
- **Daily loss kill switch** — halts all trading if realized PnL hits the daily limit
- **Drawdown trail** — halts if gains revert past a peak-to-trough threshold
- **Drawdown ramp** — reduces position size proportionally as losses accumulate
- **Per-fill cap** — never more than 5% of bankroll in a single order
- **Per-market cap** — limit total exposure to any one ticker
- **Per-asset cap** — limit exposure to the same underlying (e.g., total BTC-related exposure)
- **Depth cap** — don't take more than 25% of the inside ask in one order
- **Tier-graduated sizing** — enter small at early thresholds, larger at high-confidence levels

The sizing formula is **quarter-Kelly**: `kelly_fraction * 0.25`. Full Kelly maximizes growth in theory but assumes perfect probability estimates. Quarter-Kelly halves variance while retaining most of the compounding benefit.

### Step 5 — Execute simultaneously on both venues

The multi-leg coordinator fires both legs of a trade concurrently with `asyncio.gather`. If either leg fails, it automatically unwinds the leg that succeeded. Maximum legging window is 2 seconds.

---

## System architecture at a glance

```
data/catalog.json          ← 15k Kalshi markets, fetched daily
data/vectors.faiss         ← FAISS index, 512-dim Gemini embeddings
         │
         ▼
src/context/retriever.py   ← FAISS search: top-30 similar markets
src/context/reranker.py    ← LLM reranker: score relevance 0-10
src/inference/engine.py    ← LLM inference: find mispricings
src/signals/*.py           ← 6 signal models → CalibratedEdge list
         │
         ▼
src/portfolio/risk_gates.py   ← 8-layer risk stack → allowed contracts
src/execution/multi_leg.py    ← concurrent leg execution + auto-unwind
scripts/run_arb_live.py       ← main orchestration loop
scripts/supervisor.py         ← watchdog with exponential backoff restarts
```

---

## Key source files to read first

| File | Why |
|---|---|
| [src/catalog/models.py](../src/catalog/models.py) | `CatalogMarket` — the core data type everything else uses |
| [src/signals/models.py](../src/signals/models.py) | `CalibratedEdge` — the output type of all signal models |
| [src/portfolio/risk_gates.py](../src/portfolio/risk_gates.py) | Understand this before touching live trading |
| [src/execution/multi_leg.py](../src/execution/multi_leg.py) | Core of execution — leg coordination + unwind |
| [scripts/run_arb_live.py](../scripts/run_arb_live.py) | Entry point for the live arb loop |
| [infer.py](../infer.py) | Entry point for on-demand inference |

---

## Project conventions

- **No `print()`.** Use `from src.utils.logging import get_logger; logger = get_logger(__name__)` everywhere.
- **Pydantic for all data schemas.** If you create a new structured type, make it a `BaseModel`.
- **`asyncio` for all I/O.** HTTP calls, Redis reads, file writes that block — all async.
- **`retry_with_backoff()`** wraps any HTTP call that could rate-limit. Import from `src/utils/retry.py`.
- **Modern type hints.** Write `str | None`, `list[str]`, not `Optional[str]`, `List[str]`.
- **`@dataclass`** for config objects. `BaseModel` for data objects.
- **Tests live in `tests/`.** 179 tests. Run them with `pytest tests/ -q`.

---

## Running the tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
pytest tests/ -q
# Expected: 179 passed in ~2s
```

If tests fail before you've changed anything, check that `fakeredis` and `pytest-asyncio` installed correctly.

---

## Deployment

There are two deployment modes: **local** (for development and one-off runs) and **Docker on EC2** (for 24/7 operation).

### Local deployment

**Prerequisites:**
- Python 3.12+
- Redis (`brew install redis` on macOS)
- A `.env` file with API keys (copy `.env.example` to `.env` and fill in)

**Minimum `.env` for read-only / paper trading:**
```
# LLM — use Gemini via OpenAI-compat endpoint (cheapest, free tier available)
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=AIzaSy...            # same key, different env var name
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

**Add for live Kalshi trading:**
```
KALSHI_API_KEY_ID=<your-key-id>
KALSHI_PRIVATE_KEY_PATH=keys/<your-key-file>.txt
```

**Add for live Polymarket trading:**
```
POLYMARKET_PRIVATE_KEY=0x...        # Polygon EOA with USDC
```

**One-time setup for Polymarket:**
1. Create a Polygon wallet (MetaMask works)
2. Deposit native USDC on Polygon (not bridged USDC.e)
3. Approve USDC → `CTFExchange` allowance and `ConditionalTokens` → `CTFExchange` allowance per Polymarket docs
4. Export private key into `POLYMARKET_PRIVATE_KEY`

**Build the market index (required before running any inference):**
```bash
python3 -m scripts.build_index
# Produces data/catalog.json, data/vectors.faiss, data/vectors_meta.json
# Takes ~7 min at Gemini free-tier rate limits
```

**Run inference on a single ticker:**
```bash
python3 infer.py KXBTCD-26DEC31-B100000 --no-redis --dry-run
# --no-redis: use catalog prices instead of live Redis snapshot
# --dry-run: run retrieval + reranking only, skip LLM inference
```

**Run the arb scanner in paper mode:**
```bash
python3 -m scripts.run_arb_live
# Runs with stub clients — no real orders placed
```

**Run arb scanner live (real money):**
```bash
redis-server &
python3 -m scripts.supervisor --max-restarts 5 --backoff-base 5 -- \
  python3 -m scripts.run_arb_live --live \
    --bankroll 500 \
    --daily-loss-limit 30 \
    --min-edge 0.02 \
    --max-trade-usd 50 \
    --use-redis
# Type "I CONFIRM" when prompted
```

---

### Docker deployment (recommended for 24/7)

The project ships a `Dockerfile` and `docker-compose.yml`. One image (`pmsp:latest`) runs all services. Services are split by Compose profile so you only start what you need.

**Docker Compose profiles:**

| Profile | What starts |
|---|---|
| `core` | Redis + Kalshi WebSocket ingestion |
| `trading` | Redis + arb scanners + order book scanner + paper trader |
| `infra` | Scheduler (daily `build_index` cron at 06:00 UTC) |
| `all` | Everything above |

**`.env` for Docker** (two values differ from local):
```
REDIS_HOST=redis          # NOT localhost — use Docker service name
KALSHI_PRIVATE_KEY_PATH=keys/<your-key-file>.txt
```

**Build the image:**
```bash
docker compose build
```

**Start services:**
```bash
# Core only (data collection)
docker compose --profile core up -d

# Core + trading (scanners + paper trader)
docker compose --profile core --profile trading up -d

# Everything including daily scheduler
docker compose --profile all up -d
```

**Check service health:**
```bash
docker compose ps
# All services should show "Up"

docker compose logs -f ingestion    # Kalshi WS stream
docker compose logs -f scanners     # arb scanner output
tail -f logs/scanner_hits.jsonl     # alert log
```

---

### EC2 deployment (the live instance)

The system runs 24/7 on an AWS EC2 `t3.small` (2 vCPU, 2 GB RAM) in `us-east-2`. SSH access:

```bash
ssh -i <path-to-your-key.pem> ubuntu@<EC2_PUBLIC_IP>
# Get the IP from the AWS console
```

**Important:** do **not** run `python3 -m scripts.build_index` on EC2. The t3.small doesn't have enough RAM for the full embed of 15k markets and will OOM (exit code 137). Build locally, then upload:

```bash
# Run from your local machine:
scp -i <path-to-your-key.pem> \
  data/catalog.json \
  data/vectors.faiss \
  data/vectors_meta.json \
  ubuntu@<EC2_PUBLIC_IP>:~/Prediction-Market-Signal-Pipeline/data/
```

**Updating the code on EC2:**
```bash
cd ~/Prediction-Market-Signal-Pipeline
git pull
docker compose build
docker compose --profile core --profile trading up -d
```

**If a service keeps crashing:**
```bash
docker compose logs --tail 30 <service-name>
```

| Error | Fix |
|---|---|
| `FileNotFoundError: data/catalog.json` | Upload from local (see above) |
| `ConnectionError: connecting to localhost:6379` | Set `REDIS_HOST=redis` in `.env` |
| `FileNotFoundError: keys/<file>.txt` | Re-upload the key file |
| Exit code 137 (OOM) | Don't run `build_index` on EC2 |

---

## Alert system

The system emits structured alerts at three severity levels:

| Severity | Destinations |
|---|---|
| `info` | `logs/alerts.jsonl` only |
| `warning` | file + Discord + Slack |
| `high` | file + Discord + Slack + Pushover mobile push |

Configure webhooks in `.env`:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
```

**Alerts that require action:**

| Alert | Meaning | What to do |
|---|---|---|
| `Arb leg failed — exposure may exist` | One leg filled, other didn't | Check open positions on both exchanges manually |
| `Reconciler score crossed halt threshold` | Fill prices/qty diverging from expectations | Investigate last few fills, adjust order type or thresholds |
| `Supervisor giving up — max restarts` | Process crashed 5+ times in a row | Check `logs/arb_live.jsonl` for the root cause |
| `Kill switch active` | Daily loss limit hit | Review PnL, decide whether to reset for the day |

---

## What's not done yet (known gaps)

- **Live backtesting data is sparse.** The EC2 instance snapshots prices daily. After 30+ days it will have enough history for `run_backtest.py` to give meaningful results. Until then, use `measure_llm_signal_edge.py` for signal validation.
- **Calibration map is trained only on crypto markets (178 samples).** Training on a broader market universe would improve the `CalibratedLLMSignal` accuracy.
- **No automated P&L reporting.** Positions and fills are in Redis. Add a reporting script that reads them and emails/posts a daily summary if you want visibility without SSH access.
- **Polymarket NEG_RISK markets behave differently** from regular binary markets (they share collateral across a group). The execution layer doesn't handle them specially yet — be cautious about trading them live.

---

## Glossary

- **CLOB** — Central Limit Order Book. Polymarket's off-chain order matching system with on-chain settlement.
- **CTF / Conditional Tokens Framework** — Polymarket's on-chain binary outcome token system. YES token settles to $1, NO to $0.
- **Embedding** — a fixed-size numeric vector representing text semantics. Similar texts produce nearby vectors.
- **FAISS** — Facebook AI Similarity Search. Efficient nearest-neighbour library used to find related markets.
- **Implied probability** — `(yes_bid + yes_ask) / 2`. What the market prices the event's chance of happening.
- **Quarter-Kelly** — position size = 25% of the full Kelly criterion fraction. Reduces variance while preserving most compounding benefit.
- **RSA-PSS** — signature scheme Kalshi requires for API auth. Private key lives in `keys/` (never committed).
- **Riskless arb** — a trade that profits regardless of outcome, exploiting logical inconsistencies between related markets.
- **Structural arb** — synonym for riskless arb; refers specifically to monotonicity or partition violations within one platform.

---

*For questions about the codebase: read the 179 tests in `tests/` first — they are the most accurate documentation of expected behavior. The tests for `run_arb_live` (`tests/test_run_arb_live.py`) are the best starting point for understanding how the live orchestration loop is supposed to work.*
